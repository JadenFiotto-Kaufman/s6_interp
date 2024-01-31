import plotly.express as px
import torch
from transformers import AutoTokenizer

from nnsight.models.Mamba import MambaInterp
from nnsight.util import fetch_attr


def main(repo_id, prompt, layer_module_path, abs):
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b", padding_side="left"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = MambaInterp(repo_id, device="cuda", tokenizer=tokenizer)
    with model.invoke(prompt) as invoker:
        input = invoker.input

        values = [
            fetch_attr(
                model.backbone.layers[layer_idx], layer_module_path
            ).output.save()
            for layer_idx in range(len(model.backbone.layers))
        ]

    values = torch.concatenate([value.value for value in values])

    if abs:

        values = values.abs()

    values = values.mean(dim=1)

    if values.ndim == 3:

        values = values.mean(dim=-1)

    values = values.cpu()

    clean_tokens = [model.tokenizer.decode(token) for token in input["input_ids"][0]]
    token_labels = [f"{token}_{index}" for index, token in enumerate(clean_tokens)]

    def vis(values, token_labels, name):

        fig = px.imshow(
            values,
            color_continuous_midpoint=0.0,
            color_continuous_scale="RdBu",
            labels={"x": "Token", "y": "Layer"},
            x=token_labels,
            title=name,
        )

        fig.write_image(f"{name}.png")

    vis(values, token_labels, layer_module_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("prompt")
    parser.add_argument("--repo_id", default="state-spaces/mamba-2.8b")
    parser.add_argument("--layer_module_path", default="mixer.ssm.discA")
    parser.add_argument("--abs", default=False, action="store_true")

    main(**vars(parser.parse_args()))
