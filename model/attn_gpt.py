# model/gpt_oracle.py
import torch
import torch.nn as nn
from .attention import LLMAttention

class _Block(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.att  = LLMAttention(d_model, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, att):
        x = x + self.att(self.ln1(x), att)
        x = x + self.mlp(self.ln2(x))
        return x

class AttnGPT(nn.Module):
    def __init__(self, vocab_size: int,
                 d_model: int = 128,
                 n_layers: int = 4,
                 d_ff: int = 512,
                 max_len: int = 512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(max_len, d_model))
        self.blocks  = nn.ModuleList([_Block(d_model, d_ff) for _ in range(n_layers)])
        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, att_tensor):
        # idx: (B, T)  |  att_tensor: (B, T, T)
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:T]
        for blk in self.blocks:
            x = blk(x, att_tensor)
        return self.head(self.ln_f(x))