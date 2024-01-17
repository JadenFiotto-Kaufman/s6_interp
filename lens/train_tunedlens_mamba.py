from copy import deepcopy
from typing import List

import accelerate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import wandb
from nnsight.models.Mamba import Mamba

accelerator = accelerate.Accelerator(log_with="wandb")

global_step = 0

def val_epoch(dataloader, model, lens, max_length):
    log_losses = None
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Val:", disable=not accelerator.is_local_main_process)):

            with model.forward(validate=False, inference=True) as runner:
                with runner.invoke(
                    batch["text"], scan=False, truncation=True, max_length=max_length
                ) as invoker:
                    last_output = (
                        model.backbone.layers[-1].output[0]
                        + model.backbone.layers[-1].output[1]
                    )

                    losses = []

                    for layer_idx, layer in enumerate(model.backbone.layers[:-1]):
                        hidden_state = layer.output[0] + layer.output[1]

                        predicted_hidden_state = (
                            lens.layer_translators[layer_idx](hidden_state) + hidden_state
                        )

                        loss = torch.nn.functional.mse_loss(
                            predicted_hidden_state, last_output
                        )

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


def train_epoch(dataloader, optimizer, model, lens, max_length):
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Train:", disable=not accelerator.is_local_main_process)):
        optimizer.zero_grad()

        with model.forward(validate=False, inference=False) as runner:
            with runner.invoke(
                batch["text"], scan=False, truncation=True, max_length=max_length
            ) as invoker:
                last_output = (
                    model.backbone.layers[-1].output[0]
                    + model.backbone.layers[-1].output[1]
                )

                losses = []

                for layer_idx, layer in enumerate(model.backbone.layers[:-1]):
                    hidden_state = layer.output[0] + layer.output[1]

                    predicted_hidden_state = (
                        lens.layer_translators[layer_idx](hidden_state) + hidden_state
                    )

                    loss = torch.nn.functional.mse_loss(
                        predicted_hidden_state, last_output
                    )

                    losses.append(loss.save())

                total_loss = sum(losses).save()

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


def train(lr, model, seed, batch_size, epochs, dataset, max_length):
    accelerator.init_trackers(
        project_name="my_project",
        config={
            "epochs": epochs,
            "model": model,
            "seed": seed,
            "learning_rate": lr,
            "batch_size": batch_size,
            "dataset": dataset,
            "max_length": max_length,
        },
    )

    torch.manual_seed(seed)

    # Load with nnsight
    model = Mamba(model, dispatch=True)

    # Real mamba model is at model.local_model
    # half precision
    model.local_model = model.local_model.to(torch.bfloat16)

    # Freeze parameters of model
    for param in model.local_model.parameters():
        param.requires_grad = False

    class TunedLens(torch.nn.Module):
        def __init__(self, layers: List, d_model: int) -> None:
            super().__init__()

            translator = torch.nn.Linear(d_model, d_model, bias=True)
            translator.weight.data.zero_()
            translator.bias.data.zero_()

            self.layer_translators = torch.nn.ModuleList(
                [deepcopy(translator) for _ in range(len(layers) - 1)]
            )

    # Dimension size of hidden states  through nnsight
    d_model_hidden_states = model.backbone.layers[0].output_shape[0][-1]

    # Tuned lens is just ModuleList of linear layers
    lens = TunedLens(model.backbone.layers, d_model_hidden_states).to("cuda")

    # Minipile dataset
    dataset = load_dataset(dataset, data_dir="data")

    train_dataset = dataset["test"]
    val_dataset = dataset["validation"]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(lens.parameters(), lr=lr)

    (
        model.local_model,
        optimizer,
        train_dataloader,
        val_dataloader,
    ) = accelerator.prepare(
        model.local_model, optimizer, train_dataloader, val_dataloader
    )

    try:
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")

            train_epoch(train_dataloader, optimizer, model, lens, max_length)

            accelerator.wait_for_everyone()
            val_epoch(val_dataloader, model, lens, max_length)
            accelerator.wait_for_everyone()

            accelerator.save_model(lens, f"tunedlens_{epoch}")

        accelerator.end_training()

    except Exception as e:
        accelerator.save_model(lens, f"tunedlens_{epoch}_exception")

        accelerator.end_training()

        raise e


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--lr", default=0.03, type=float)
parser.add_argument("--model", default="state-spaces/mamba-1.4b")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--batch_size", default=30, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--dataset", default="JeanKaddour/minipile")
parser.add_argument("--max_length", default=20, type=int)

train(**vars(parser.parse_args()))
