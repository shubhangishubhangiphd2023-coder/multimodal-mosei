from transformers import ViTModel
import torch.nn as nn

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224"
        )

    def forward(self, x):
        return self.model(pixel_values=x).last_hidden_state
