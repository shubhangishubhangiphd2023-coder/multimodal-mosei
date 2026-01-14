import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, 8, batch_first=True
        )

    def forward(self, q, k, v):
        return self.attn(q, k, v)[0]


class MulT(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.t_a = CrossAttention(dim)
        self.t_v = CrossAttention(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, t, a, v):
        t = self.t_a(t, a, a)
        t = self.t_v(t, v, v)
        return self.norm(t)
