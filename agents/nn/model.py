"""Set transformer with a pointer policy and value head.

Token embeddings sum kind, IDs, numeric projections, and value-scaled dynamic-variable
embeddings. Positions are encoded only through categorical IDs from encode.py.
The policy reads action pointers and the global token; the value head reads the global
token and mean token embedding.
"""

import torch
from torch import nn

from .encode import KINDS, N_NUM


class Net(nn.Module):
    def __init__(self, vocab_size, d=128, layers=3, heads=4):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, d=d, layers=layers, heads=heads)
        self.ids = nn.Embedding(vocab_size, d, padding_idx=0)
        self.kind = nn.Embedding(len(KINDS), d)
        self.num = nn.Linear(N_NUM, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.none = nn.Parameter(torch.zeros(d))  # stands in for the second pointer of a one-token input
        self.pi = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 1))
        self.v = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 1))

    def forward(self, b):
        x = (self.ids(b["ident"]) + self.ids(b["sub"]).sum(2) + self.kind(b["kind"]) + self.num(b["nums"])
             + (self.ids(b["varn"]) * b["varv"].unsqueeze(-1)).sum(2))
        h = self.norm(self.body(x, src_key_padding_mask=~b["mask"]))
        glob = h[:, 0]  # encode() always writes the global token first
        m = b["mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1)
        value = self.v(torch.cat([glob, pooled], -1)).squeeze(-1)

        act = b["act"]
        B, A, _ = act.shape
        idx = act.clamp(min=0)
        gather = lambda j: torch.gather(h, 1, idx[..., j].unsqueeze(-1).expand(B, A, h.shape[-1]))
        first, second = gather(0), gather(1)
        second = torch.where((act[..., 1] < 0).unsqueeze(-1), self.none.expand_as(second), second)
        logits = self.pi(torch.cat([first, second, glob.unsqueeze(1).expand(B, A, -1)], -1)).squeeze(-1)
        logits = logits.masked_fill(~b["act_mask"], float("-inf"))
        return logits, value
