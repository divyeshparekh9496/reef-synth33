"""Nebius Token Factory client with tiered Nemotron routing.

Token Factory exposes an OpenAI-compatible API, so the standard `openai`
client works with a different base URL.

Tiers (override any of them with environment variables):

    nano   NVIDIA Nemotron 3 Nano   - fast, cheap, high-volume calls (shift notes)
    super  NVIDIA Nemotron 3 Super  - structured generation (scenario specs,
                                      control proposals, data critique)
    ultra  NVIDIA Nemotron 3 Ultra  - deep reasoning (root-cause diagnosis,
                                      daily farm report)

If NEBIUS_API_KEY is not set, `chat` returns None and every agent falls
back to a deterministic rule-based answer. That keeps the repo runnable for
anyone, but the Nemotron path is the real product.

Model IDs on Token Factory change over time. List the current ones with:

    curl -s https://api.tokenfactory.nebius.com/v1/models \
         -H "Authorization: Bearer $NEBIUS_API_KEY" | jq '.data[].id'
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

TIERS = {
    "nano": os.getenv("NEMOTRON_NANO_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"),
    "super": os.getenv("NEMOTRON_SUPER_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
    "ultra": os.getenv("NEMOTRON_ULTRA_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b"),
}

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class Ledger:
    """Tracks every model call so the demo can show where credits went."""

    calls: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, **kw) -> None:
        with self.lock:
            self.calls.append(kw)

    def summary(self) -> dict:
        out: dict[str, dict] = {}
        for c in self.calls:
            t = out.setdefault(c["tier"], {"model": c["model"], "calls": 0, "prompt_tokens": 0,
                                           "completion_tokens": 0, "seconds": 0.0, "errors": 0})
            t["calls"] += 1
            t["prompt_tokens"] += c.get("prompt_tokens", 0)
            t["completion_tokens"] += c.get("completion_tokens", 0)
            t["seconds"] = round(t["seconds"] + c.get("seconds", 0.0), 2)
            t["errors"] += int(bool(c.get("error")))
        return out


LEDGER = Ledger()
_client = None


def live() -> bool:
    return bool(os.getenv("NEBIUS_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        client = OpenAI(base_url=os.getenv("NEBIUS_BASE_URL", DEFAULT_BASE_URL),
                        api_key=os.environ["NEBIUS_API_KEY"], timeout=float(os.getenv("NEBIUS_TIMEOUT", "120")))
        # Optional LangSmith tracing: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY.
        if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
            try:
                from langsmith.wrappers import wrap_openai

                client = wrap_openai(client)
            except ImportError:
                pass
        _client = client
    return _client


def chat(tier: str, system: str, user: str, max_tokens: int = 800, temperature: float = 0.2,
         agent: str = "") -> str | None:
    """Call a Nemotron tier. Returns the text, or None when offline or on error."""
    if not live():
        LEDGER.add(tier=tier, model=TIERS[tier], agent=agent, offline=True)
        return None
    model = TIERS[tier]
    t0 = time.time()
    try:
        resp = _get_client().chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        LEDGER.add(tier=tier, model=model, agent=agent, seconds=time.time() - t0,
                   prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                   completion_tokens=getattr(usage, "completion_tokens", 0) or 0)
        return _THINK.sub("", text).strip()
    except Exception as e:  # network, quota, unknown model id ...
        LEDGER.add(tier=tier, model=model, agent=agent, seconds=time.time() - t0, error=str(e)[:300])
        return None


def extract_json(text: str | None):
    """Pull the first JSON object or array out of a model reply."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None
