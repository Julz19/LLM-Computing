# oracle/scorer.py
import asyncio, os, re
from collections import deque
from functools import lru_cache
from typing import List

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()                          # loads OPENAI_API_KEY
client = AsyncOpenAI()
HISTORY_SIZE = 10                      # short rolling context for the prompt

# -------- structured output schema -----------------------------------------
class AttentionScore(BaseModel):
    consice_reasoning: str
    score: float

# -------- single pair score -------------------------------------------------
async def _oracle_score(q: str, k: str, history: deque):
    history_str = "\n".join([f"{a}->{b}={s:.2f}" for a, b, s in history]) or "[None yet]"
    prompt = f"""
You are computing semantic attention scores between tokens in a sentence.

Previously computed attention scores:
{history_str}

Now determine how strongly:
- Query Token: "{q}"
should semantically attend to:
- Key Token: "{k}"

Only return a float between 0.0 and 1.0."""
    # responses.parse returns the Pydantic object directly
    resp = await client.responses.parse(
        model="gpt-4.1-nano",
        input=[{"role": "user", "content": prompt}],
        temperature=0.3,
        text_format=AttentionScore,
    )
    try:
        score = float(resp.output_parsed.score)
    except ValueError:
        score = 0.0
    history.append((q, k, score))
    return score

# -------- full matrix for a token list -------------------------------------
async def _compute_attention_matrix(tokens: List[str]) -> torch.Tensor:
    n = len(tokens)
    history = deque(maxlen=HISTORY_SIZE)

    tasks = [
        _oracle_score(tokens[i], tokens[j], history.copy())
        for i in range(n) for j in range(n)
    ]
    results = await asyncio.gather(*tasks)
    mat = torch.tensor(results, dtype=torch.float32).view(n, n)
    return F.softmax(mat, dim=-1)      # (n, n)

# -------- public helper with in-memory cache -------------------------------
@lru_cache(maxsize=2_048)             # key = tuple(tokens); keeps RAM small
def get_attention_matrix(tokens: tuple) -> torch.Tensor:
    """
    Synchronous wrapper so the rest of the code can stay non-async.
    Results are memoised for the lifetime of the process.
    """
    return asyncio.run(_compute_attention_matrix(list(tokens)))