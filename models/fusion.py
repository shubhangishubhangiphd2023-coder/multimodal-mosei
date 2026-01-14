import torch.nn as nn
from .text_encoder import TextEncoder
from .audio_encoder import AudioEncoder
from .vision_encoder import VisionEncoder
from .mult import MulT

class End2EndMulT(nn.Module):
    def __init__(self, num_emotions=6):
        super().__init__()

        self.text = TextEncoder()
        self.audio = AudioEncoder()
        self.vision = VisionEncoder()
        self.fusion = MulT()

        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions)
        )

    def forward(self, text, audio, vision):
        t = self.text(text)
        a = self.audio(audio)
        v = self.vision(vision)

        fused = self.fusion(t, a, v)
        pooled = fused.mean(1)

        return self.head(pooled)
