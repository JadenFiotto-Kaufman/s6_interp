from transformers import AutoTokenizer
from nnsight.models.Mamba import MambaInterp
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-neox-20b", padding_side="left"
)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = MambaInterp("state-spaces/mamba-2.8b", device="cuda", tokenizer=tokenizer)

with model.trace("The Eiffel Tower is in the city of") as invoker:

    input = invoker.input
    bbb = model.backbone.layers[0].mixer.ssm.input[0][0].save()
    values = [model.backbone.layers[layer_idx].output.save() for layer_idx in range(len(model.backbone.layers))]
breakpoint()
clean_tokens = [model.tokenizer.decode(token) for token in input['input_ids'][0]]
token_labels = [f"{token}_{index}" for index, token in enumerate(clean_tokens)]


def vis(values, token_labels, name):

    import plotly.express as px

    values = torch.concatenate([value.value for value in values]).abs().mean(dim=1).mean(dim=-1).cpu()



    fig = px.imshow(
        values,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": "Position", "y": "Layer"},
        x=token_labels,
        title=name,
    )

    fig.write_image(f"{name}.png")

    



vis(values, token_labels, 'discA')

