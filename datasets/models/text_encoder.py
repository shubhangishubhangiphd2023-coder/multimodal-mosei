from transformers import BertModel
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, x):
        return self.model(**x).last_hidden_state
