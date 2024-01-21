from copy import deepcopy
from typing import List

import accelerate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import lovely_tensors as lt 
lt.monkey_patch()

import wandb
from nnsight.models.Mamba import Mamba
from mamba_ssm.ops.triton.layernorm import rms_norm_fn

accelerator = accelerate.Accelerator(log_with="wandb")
accelerator.init_trackers(project_name="mamba_lenses")

global_step = 0 

model = Mamba("state-spaces/mamba-1.4b", device="cuda", dispatch=True)

# Real mamba model is at model.local_model
# half precision
# model.local_model = model.local_model.to(torch.bfloat16)

# Freeze parameters of model
for param in model.local_model.parameters():
    param.requires_grad = False

# give just the last token of each example in batch (bsz, vocab_size)
def kl_divergence(labels, outputs):
    return torch.sum(
        labels.exp() * (labels - outputs), dim=-1
    ).mean()

# func that does the last decode step after hook for either logitlens or tunedlens
# lens=None for logit lens, or give it a TunedLens and it'll do that 
def decode(output, layer=0, lens=None):
    hidden_states = output[0] + output[1]

    if lens is not None and layer != len(lens.layer_translators):
        hidden_states = lens.layer_translators[layer](hidden_states) + hidden_states

    norm_f = model.local_model.backbone.norm_f

    decoded = hidden_states.node.graph.add(
        target=rms_norm_fn,
        args=[hidden_states, norm_f.weight, norm_f.bias],
        kwargs={
            "eps": norm_f.eps,
            "residual": None,
            "prenorm": False,
            "residual_in_fp32": True,
        },
    )

    return model.lm_head(decoded)


class TunedLens(torch.nn.Module):
    def __init__(self, layers: List, d_model: int) -> None:
        super().__init__()

        translator = torch.nn.Linear(d_model, d_model, bias=True)
        translator.weight.data.zero_()
        translator.bias.data.zero_()

        self.layer_translators = torch.nn.ModuleList(
            [deepcopy(translator) for _ in range(len(layers) - 1)]
        )


def test(dataloader, model, tuned_lens, max_length):
    log_losses = None
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Test:", disable=not accelerator.is_local_main_process)):

            with model.forward(validate=False, inference=True) as runner:
                with runner.invoke(
                    batch["text"], scan=False, truncation=True, max_length=max_length
                ) as invoker:
                    # get output probs by using "logit lens" on the last layer 
                    model_pred_probs = torch.nn.functional.log_softmax(
                        decode(model.backbone.layers[-1].output)[:, -1, :]
                    )

                    logit_kls = []
                    tuned_kls = []

                    for layer_idx, layer in enumerate(model.backbone.layers):
                        
                        logitlens_logprobs = torch.nn.functional.log_softmax(
                            decode(layer.output, layer=layer_idx), dim=-1 # (bsz, seq_len, vocab)
                        )[:, -1, :].save()
                        
                        logit_kls.append(
                            kl_divergence(model_pred_probs, logitlens_logprobs).save()
                        )

                        tunedlens_logprobs = torch.nn.functional.log_softmax(
                            decode(layer.output, layer=layer_idx, lens=tuned_lens), dim=-1
                        )[:, -1, :].save() # (bsz, 50280)
                        
                        tuned_kls.append(
                            kl_divergence(model_pred_probs, tunedlens_logprobs).save()
                        )

            logit_kls = [t.value.cpu() for t in logit_kls]
            tuned_kls = [t.value.cpu() for t in tuned_kls]
            
            if log_losses is None:
                log_losses = {f"val_logit_kl_layer_{layer_idx}" : [kl] for layer_idx, kl in enumerate(logit_kls)}
                for layer_idx, kl in enumerate(tuned_kls):
                    log_losses[f"val_tuned_kl_layer_{layer_idx}"] = [kl]    
            else:
                for layer_idx, kl in enumerate(logit_kls):
                    log_losses[f"val_logit_kl_layer_{layer_idx}"].append(kl)
                for layer_idx, kl in enumerate(tuned_kls):
                    log_losses[f"val_tuned_kl_layer_{layer_idx}"].append(kl)

        log_losses = {key: torch.tensor(value).mean() for key, value in log_losses.items()}
        accelerator.log(log_losses, step=global_step)
    
    return log_losses


# Dimension size of hidden states  through nnsight
d_model_hidden_states = model.backbone.layers[0].output_shape[0][-1]

# Tuned lens is just ModuleList of linear layers
lens = TunedLens(model.backbone.layers, d_model_hidden_states).to("cuda")
lens = accelerate.load_checkpoint_and_dispatch(lens, '/share/u/jadenfk/wd/tunedlens_34/model.safetensors')

# Minipile dataset
dataset = load_dataset("JeanKaddour/minipile", data_dir="data")
val_dataset = dataset["validation"]
val_dataloader = DataLoader(val_dataset, batch_size=30, shuffle=False)

(model.local_model, val_dataloader) = accelerator.prepare(model.local_model, val_dataloader)
 
logged_kls = test(val_dataloader, model, lens, 20)
accelerator.wait_for_everyone()

logit_kls = [logged_kls[f"val_logit_kl_layer_{i}"] for i, _ in enumerate(model.backbone.layers)]
tuned_kls = [logged_kls[f"val_tuned_kl_layer_{i}"] for i, _ in enumerate(model.backbone.layers)]

import matplotlib.pyplot as plt 
import numpy as np
x = [i for i, _ in enumerate(model.backbone.layers)]
plt.figure(figsize=(15,8))
plt.plot(x, np.array(logit_kls), label="Logit Lens")
plt.plot(x, np.array(tuned_kls), label="Tuned Lens")
plt.legend()
plt.xticks(x)
plt.xlabel("Layer")
plt.ylabel("KL Divergence (nats)")
plt.savefig("compare.png")


