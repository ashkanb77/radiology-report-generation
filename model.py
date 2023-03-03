from torch import nn
from transformers import BlipForConditionalGeneration


class CaptioningModule(nn.Module):
    def __init__(self, model_path):
        super(CaptioningModule, self).__init__()
        self.model = BlipForConditionalGeneration.from_pretrained(model_path)

    def forward(self, input_ids, pixel_values):
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids
        )
