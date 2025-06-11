# sample.py
import torch, re, readline       # noqa: F401 (readline gives nicer CLI)
from attention.scorer import get_attention_matrix
from llm_attn.model.attn_gpt import AttnGPT

from train import stoi, itos, device     # reuse what we built during training
from pathlib import Path

# ------------ load the trained checkpoint (or import the model directly) ----
ckpt_path = Path("checkpoint.pt")
model = AttnGPT(len(stoi))
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.to(device).eval()

# ------------ utils ----------------------------------------------------------
def tokens_to_tensor(tokens):
    return torch.tensor([stoi[t] for t in tokens], dtype=torch.long, device=device)

@torch.inference_mode()
def generate(prompt: str, max_new: int = 20):
    tokens = re.findall(r"\w+|\.", prompt.lower())
    idx = tokens_to_tensor(tokens)

    for _ in range(max_new):
        att = get_attention_matrix(tuple(tokens)).to(device)
        att_trim = att[: idx.shape[0], : idx.shape[0]].unsqueeze(0)

        logits = model(idx.unsqueeze(0), att_trim)[0, -1]
        next_id = torch.distributions.Categorical(logits.softmax(-1)).sample()
        idx = torch.cat([idx, next_id.view(1)], 0)
        tokens.append(itos[next_id.item()])

    return " ".join(tokens)

# ------------ CLI -----------------------------------------------------------
while True:
    try:
        prompt = input("\nPrompt> ").strip()
        if not prompt:
            continue
        print(generate(prompt))
    except (EOFError, KeyboardInterrupt):
        break