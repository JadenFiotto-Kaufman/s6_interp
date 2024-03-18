"""
Train a linear probe that tries to predict the N+1 tokens ahead. 
--direct option indicates whether to train probe to predict vocabulary logits after softmax.
Otherwise, probe is trained to predict the hidden state of that token at the last layer \

If N=0, next token prediction (imitate logits at last layer for this token). 
If not --direct, then it's the same as tuned lens. 
"""
import os
import torch
import accelerate

from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List
from copy import deepcopy
from datasets import load_dataset
from torch.utils.data import DataLoader

from nnsight.models.Mamba import MambaInterp
from nnsight.tracing.Proxy import Proxy

from mamba_ssm.ops.triton.layernorm import rms_norm_fn

accelerator = accelerate.Accelerator(log_with="wandb")

global_step = 0 

class LinearFutureLens(torch.nn.Module):
    def __init__(self, layers: List, d_model: int, vocab_size: int, direct=True) -> None:
        super().__init__()

        if direct:
            translator = torch.nn.Linear(d_model, vocab_size, bias=True)
        else:
            translator = torch.nn.Linear(d_model, d_model, bias=True)

        self.layer_translators = torch.nn.ModuleList(
            [deepcopy(translator) for _ in range(len(layers) - 1)]
        )

def kl_loss(outputs, labels, direct=True, model=None):
    if not direct: 
        # do "logit lens" on inputs if they are hidden states 
        assert model is not None 
        labels = torch.nn.functional.log_softmax(decode(labels, model), dim=-1)
        outputs = torch.nn.functional.log_softmax(decode(outputs, model), dim=-1)
    
    return torch.sum(
        labels.exp() * (labels - outputs), dim=-1
    ).mean()

# does "logit lens" on a hidden state
def decode(hidden_states, model):
    norm_f = model.backbone.norm_f

    decoded = hidden_states.node.graph.add(
        target=rms_norm_fn,
        args=[hidden_states, norm_f.weight, norm_f.bias],
        kwargs={
            "eps": norm_f.eps,
            "residual": None,
            "prenorm": False,
            "residual_in_fp32": True,
        }
    )

    return model.lm_head(decoded)


def val_epoch(dataloader, model, lens, max_length, direct, N):
    log_losses = None
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val:"): # disable=not accelerator.is_local_main_process

            with model.trace() as tracer:
                with tracer.invoke(
                    batch["text"], scan=True, truncation=True, max_length=max_length
                ) as invoker:
                    if direct:
                        logits = model.lm_head.output.save() # (batch_size, seq_len, vocab_size)
                        labels = logits[:, N:, :]
                    else:
                        last_output = (
                            model.backbone.layers[-1].output[0]
                            + model.backbone.layers[-1].output[1]
                        ) # (batch_size, seq_len, model_dim)
                        labels = last_output[:, N:, :]

                    losses = []

                    for layer_idx, layer in enumerate(model.backbone.layers[:-1]):
                        hidden_state = layer.output[0] + layer.output[1]
                        hidden_states = hidden_states[:, :-N, :]

                        pred_states = lens.layer_translators[layer_idx](hidden_state).save()

                        loss = kl_loss(pred_states, labels, direct=direct, model=model)
                        # loss = torch.nn.functional.mse_loss(
                        #     predicted_hidden_state, last_output
                        # )

                        losses.append(loss.save())

                    total_loss = sum(losses).save()


            if log_losses is None:
                log_losses = {f"val_loss_layer_{layer_idx}" : [loss.value] for layer_idx, loss in enumerate(losses)}
                log_losses["val_total_loss"] = [total_loss.value]

            else:

                for layer_idx, loss in enumerate(losses):
                    log_losses[f"val_loss_layer_{layer_idx}"].append(loss.value)
                    log_losses["val_total_loss"].append(total_loss.value)

        log_losses = {key: torch.tensor(value).mean() for key, value in log_losses.items()}

        accelerator.log(log_losses, step=global_step)

def train_epoch(dataloader, optimizer, model, lens, max_length, direct, N):
    for batch in tqdm(dataloader, desc="Train:"): # , disable=not accelerator.is_local_main_process
        optimizer.zero_grad()

        with model.trace() as tracer:
            with tracer.invoke(
                batch["text"], scan=True, truncation=True, max_length=max_length
            ) as invoker:
                if direct:
                    logits = model.lm_head.output.save() # (batch_size, seq_len, vocab_size)
                    labels = logits[:, N:, :]
                else:
                    last_output = (
                        model.backbone.layers[-1].output[0]
                        + model.backbone.layers[-1].output[1]
                    ) # (batch_size, seq_len, model_dim)
                    labels = last_output[:, N:, :]

                losses = []

                for layer_idx, layer in enumerate(model.backbone.layers[:-1]):
                    hidden_states = layer.output[0] + layer.output[1]
                    hidden_states = hidden_states[:, :-N, :]
                    
                    pred_states = lens.layer_translators[layer_idx](hidden_states).save()
                    
                    # pass in raw logits
                    loss = kl_loss(pred_states, labels, direct=direct, model=model)
                    # loss = torch.nn.functional.mse_loss(pred_states, labels)

                    losses.append(loss.save())
                    # break

                total_loss = sum(losses).save()
            
        # total_loss.backward()
        print(total_loss)
        accelerator.backward(total_loss)

        log_losses = dict()

        for layer_idx, loss in enumerate(losses):
            log_losses[f"loss_layer_{layer_idx}"] = loss.value

        log_losses["total_loss"] = total_loss.value

        accelerator.log(log_losses)

        optimizer.step()

        if accelerator.is_local_main_process:
            global global_step
            global_step += 1


def train(repo_id, direct, lr, seed, batch_size, epochs, dataset, max_length, N):
    accelerator.init_trackers(
        project_name="mamba-futurelens",
        init_kwargs={"wandb": {"entity" : "sfeucht"}},
        config={
            "epochs": epochs,
            "model": repo_id,
            "seed": seed,
            "learning_rate": lr,
            "batch_size": batch_size,
            "dataset": dataset,
            "max_length": max_length,
        },
    )

    torch.manual_seed(seed)

    # Load with nnsight
    VOCAB_SIZE = 50280 
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b", padding_side="left"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = MambaInterp(repo_id, tokenizer=tokenizer, dispatch=True, device="cuda:0")

    # half precision
    model.to(torch.bfloat16)

    # Freeze parameters of model
    # for param in model.local_model.parameters():
    for param in model.parameters():
        param.requires_grad = False

    # Dimension size of hidden states through nnsight
    # d_model_hidden_states = model.backbone.layers[0].output_shape[0][-1]
    d_model_hidden_states = model.config.d_model

    # Tuned lens is just ModuleList of linear layers
    lens = LinearFutureLens(model.backbone.layers, d_model_hidden_states, VOCAB_SIZE, direct)

    # Minipile dataset
    dataset = load_dataset(dataset, data_dir="data")

    train_dataset = dataset["test"]
    val_dataset = dataset["validation"]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(lens.parameters(), lr=lr)

    (
        optimizer,
        train_dataloader,
        val_dataloader,
        lens
    ) = accelerator.prepare(
        optimizer, train_dataloader, val_dataloader, lens
    )

    try:
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")

            train_epoch(train_dataloader, optimizer, model, lens, max_length, direct, N)

            accelerator.wait_for_everyone()
            val_epoch(val_dataloader, model, lens, max_length, direct, N)
            accelerator.wait_for_everyone()

            s = "direct" if direct else "indirect"
            torch.save(lens.state_dict(), "checkpoints/futurelens_linear_" + s + ".ckpt")
            accelerator.save_model(lens, f"tunedlens_{epoch}")

        accelerator.end_training()

    except Exception as e:
        accelerator.save_model(lens, f"tunedlens_{epoch}_exception")

        accelerator.end_training()

        raise e


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="state-spaces/mamba-2.8b")
    parser.add_argument("--direct", action=argparse.BooleanOptionalAction)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--seed", default=8, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--epochs", default=16, type=int)
    parser.add_argument("--dataset", default="JeanKaddour/minipile")
    parser.add_argument("--max_length", default=20, type=int)
    parser.add_argument("--N", default=2, type=int)

    #repo_id, direct, lr, seed, batch_size, epochs, dataset, max_length
    train(**vars(parser.parse_args()))