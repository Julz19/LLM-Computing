# oracle/scorer.py
"""
Provider-agnostic oracle attention scorer.

Supported providers
-------------------
* OpenAI  (default)  → LLM_PROVIDER=openai
* Cerebras           → LLM_PROVIDER=cerebras  +  CEREBRAS_API_KEY
* Custom             → LLM_PROVIDER=custom    +  CUSTOM_BASE_URL  +  CUSTOM_API_KEY

All non-OpenAI options must expose an OpenAI-compatible REST interface
(e.g. Cerebras, Groq, Together, Ollama, etc.).
"""

import asyncio, os
from collections import deque
from functools import lru_cache
from typing import List

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

# ────────────────────────── 0. Provider bootstrap ──────────────────────────
load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "cerebras").lower()

if PROVIDER == "openai":
    _client = AsyncOpenAI()                                       # uses OPENAI_API_KEY
    async def _parse(model, messages, schema, temperature=0.3):
        resp = await _client.responses.parse(                     # <-- OpenAI “responses” API
            model=model,
            input=messages,
            temperature=temperature,
            text_format=schema,
        )
        return resp.output_parsed

elif PROVIDER == "cerebras":
    _client = AsyncOpenAI(
        base_url=os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"),
        api_key=os.getenv("CEREBRAS_API_KEY"),
    )
    async def _parse(model, messages, schema, temperature=0.3):
        resp = await _client.beta.chat.completions.parse(         # <-- beta chat API
            model=model,
            messages=messages,
            response_format=schema,
            temperature=temperature,
        )
        return resp.choices[0].message.parsed

else:  # generic OpenAI-compatible endpoint
    _client = AsyncOpenAI(
        base_url=os.getenv("CUSTOM_BASE_URL"),
        api_key=os.getenv("CUSTOM_API_KEY"),
    )
    async def _parse(model, messages, schema, temperature=0.3):
        resp = await _client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=schema,
            temperature=temperature,
        )
        return resp.choices[0].message.parsed

# ────────────────────────── 1. Scoring logic ───────────────────────────────
HISTORY_SIZE   = 10
DEFAULT_MODEL  = os.getenv("LLM_MODEL", "qwen-3-32b")           # override when needed

class AttentionScore(BaseModel):
    consice_reasoning: str
    score: float

async def _oracle_score(q: str, k: str, history: deque):
    history_str = "\n".join(f"{a}->{b}={s:.2f}" for a, b, s in history) or "[None yet]"
    prompt = f"""
You are computing semantic attention scores between tokens in a sentence.

Previously computed attention scores:
{history_str}

Now determine how strongly:
- Query Token: "{q}"
should semantically attend to:
- Key Token: "{k}"

Only return a float between 0.0 and 1.0."""
    parsed: AttentionScore = await _parse(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        schema=AttentionScore,
    )
    try:
        score = float(parsed.score)
    except (TypeError, ValueError):
        score = 0.0
    history.append((q, k, score))
    return score

async def _compute_attention_matrix(tokens: List[str]) -> torch.Tensor:
    n, history = len(tokens), deque(maxlen=HISTORY_SIZE)
    tasks = [_oracle_score(tokens[i], tokens[j], history.copy())
             for i in range(n) for j in range(n)]
    results = await asyncio.gather(*tasks)
    mat = torch.tensor(results, dtype=torch.float32).view(n, n)
    return F.softmax(mat, dim=-1)

@lru_cache(maxsize=2_048)
def get_attention_matrix(tokens: tuple) -> torch.Tensor:
    """Synchronously obtain the (n×n) attention matrix for *tokens*."""
    return asyncio.run(_compute_attention_matrix(list(tokens)))