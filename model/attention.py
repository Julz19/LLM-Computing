# model/attention.py
import torch
import torch.nn as nn

class LLMAttention(nn.Module):
    """
    Consumes an externally-supplied (softmax-normalised) attention tensor
    and just does attn @ values.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, attn: torch.Tensor):
        # values: (B, T, d), attn: (B, T, T)
        if attn.shape[-1] != values.shape[1]:
            raise ValueError(f"Shape mismatch: attn {attn.shape}, values {values.shape}")
        return self.dropout(attn @ values)