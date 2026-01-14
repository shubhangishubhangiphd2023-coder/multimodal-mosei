from transformers import Wav2Vec2Model
import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

    def forward(self, x):
        return self.model(**x).last_hidden_state
