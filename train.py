# train.py
import random, re, os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from attention.scorer import get_attention_matrix       # synchronous helper
from model.attn_gpt import AttnGPT

# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
corpus_path = Path("data/corpus.txt")

def tokenize(line: str) -> List[str]:
    # simple word + '.' tokenizer
    return re.findall(r"\w+|\.", line.strip().lower())

# ---------------------------------------------------------------------------
# 1) Build vocab from entire corpus (one sentence per line)
with corpus_path.open() as f:
    sentences = [tokenize(l) for l in f if l.strip()]

vocab = sorted({tok for sent in sentences for tok in sent})
stoi  = {w: i for i, w in enumerate(vocab)}
itos  = {i: w for w, i in stoi.items()}

# 2) Model + optimiser
model = AttnGPT(len(vocab)).to(device)
opt   = optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

# ---------------------------------------------------------------------------
def tensors_for_sentence(tokens: List[str]):
    idx  = torch.tensor([stoi[t] for t in tokens], dtype=torch.long, device=device)
    att  = get_attention_matrix(tuple(tokens)).to(device)  # (T, T)
    return idx, att

EPOCHS = 500
for epoch in range(1, EPOCHS + 1):
    tokens = random.choice(sentences)
    idx, att = tensors_for_sentence(tokens)

    # teacher-forcing next-token task
    idx_in  = idx[:-1].unsqueeze(0)         # (1, T-1)
    idx_out = idx[1:]                       # (T-1)
    att_in  = att[:-1, :-1].unsqueeze(0)    # (1, T-1, T-1)

    logits = model(idx_in, att_in)
    loss = loss_fn(logits.squeeze(0), idx_out)

    opt.zero_grad(); loss.backward(); opt.step()

    if epoch % 50 == 0:
        print(f"[epoch {epoch:>4}] loss = {loss.item():.4f}")