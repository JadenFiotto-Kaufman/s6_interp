import os

import plotly.express as px
import torch
from transformers import AutoTokenizer

from nnsight import util
from nnsight.models.Mamba import MambaInterp
from nnsight.tracing.Proxy import Proxy


def main(prompt, out_path, repo_id, absv, softmax):
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b", padding_side="left"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = MambaInterp(repo_id, device="cuda", tokenizer=tokenizer)

    with model.trace(prompt) as tracer:
        layer_values = []
        layer_magnitude_values = []

        for layer_idx, layer in enumerate(model.backbone.layers):
            x = layer.mixer.ssm.input[0][0].save()

            C = layer.mixer.ssm.input[0][4].save()

            discA = layer.mixer.ssm.discA.output.save()
            discB = layer.mixer.ssm.discB.output.save()

            layer_magnitude_values.append((layer.output[0] - layer.output[1]).norm(dim=2).save())

            # Compute Bx for all tokens before.
            Bx = torch.einsum("bdln,bdl->bdln", discB, x)

            source_values = []

            # Iterate through target tokens.
            for target_token_idx in range(x.shape[2]):
                target_values = []

                # Iterate through source tokens.
                for source_token_idx in range(x.shape[2]):
                    # If source is after target token, it can't "see" it ofc.
                    if target_token_idx < source_token_idx:
                        value = -float("inf")
                    else:
                        # Multiply together all As between source and target.
                        discA_multistep = torch.prod(
                            discA[:, :, source_token_idx + 1 : target_token_idx + 1],
                            dim=2,
                        )

                        # Apply the multistep A to the Bx from source.
                        discABx = discA_multistep * Bx[:, :, source_token_idx]

                        # Apply C from target.
                        # This sums over all 'd', but we might want a separate attention map per d.
                        value = (
                            torch.einsum(
                                "bdn,bn->b", discABx, C[:, :, target_token_idx]
                            )
                            .item()
                            .save()
                        )

                    target_values.append(value)

                source_values.append(target_values)

            layer_values.append(source_values)

    # Convert to values and combine to one tensor (n_layer, n_tokens, n_tokens)

    def post(proxy):

        value = proxy.value

        if absv:

            value = abs(value)

        return value

    values = util.apply(layer_values, post, Proxy)
    values = torch.tensor(values).detach()

    if softmax:
        values = values.softmax(dim=2)
    else:
        values[values == -float('inf')] = 0
        values = values / values.sum(dim=2, keepdim=True)

    clean_tokens = [
        model.tokenizer.decode(token) for token in tracer._invoker.inputs[0]["input_ids"][0]
    ]
    token_labels = [f"{token}_{index}" for index, token in enumerate(clean_tokens)]

    def vis(values, token_labels, name, out_path):
        fig = px.imshow(
            values,
            color_continuous_midpoint=0.0,
            color_continuous_scale="RdBu",
            labels={"y": "Target Token", "x": "Source Token"},
            x=token_labels,
            y=token_labels,
            title=name,
        )

        fig.write_image(os.path.join(out_path, f"{name}.png"))

    os.makedirs(out_path, exist_ok=True)

    for layer_idx in range(values.shape[0]):
        vis(values[layer_idx], token_labels, f"layer_{layer_idx}", out_path)

    vis(values.mean(dim=0), token_labels, f"layer_mean", out_path)


    ssm_values = util.apply(layer_magnitude_values, lambda x : x.value, Proxy)
    ssm_values = torch.concatenate(ssm_values).detach().cpu()
    fig = px.imshow(
        ssm_values,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"y": "Layer", "x": "Token"},
        x=token_labels,
        y=list(range(ssm_values.shape[0])),
        title="norm",
    )

    fig.write_image(os.path.join(out_path, f"norm.png"))

    breakpoint()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("prompt")
    parser.add_argument("--out_path", default="./attn")
    parser.add_argument("--repo_id", default="state-spaces/mamba-2.8b")
    parser.add_argument("--absv", action="store_true", default=False)
    parser.add_argument("--softmax", action="store_true", default=False)


    main(**vars(parser.parse_args()))
