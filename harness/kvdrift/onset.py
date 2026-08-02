"""Context-onset probe: where does identifier-copy corruption begin?

Milestone-3 evidence: quantized-KV configs pass a ~40-token probe yet
corrupt identifiers at the ~1k-token first trajectory turn. This
instrument maps copy fidelity as a function of context length, in two
referent placements:

  near — filler first, then "look up SKU-XXXX": the identifier sits ~30
         tokens from generation; total context length is the variable.
  far  — the identifier is named first, then filler, then "look up today's
         item": referent distance ≈ context length.

Single-turn requests, no history, greedy. Filler is deterministic,
semantically inert prose built from common words (seeded), so the KV under
test carries no competing identifiers.
"""

from __future__ import annotations

import random
from typing import Any

from .scoring import extract_first_call
from .warehouse import tool_schemas

SYSTEM = "You are a warehouse operations assistant with tool access. Use exactly one tool call when asked to look something up."

_TEMPLATES = [
    "The morning shift finished the receiving dock work before the first truck arrived.",
    "Staff moved the empty pallets to the back wall and swept the main aisle.",
    "The afternoon count matched the morning count in every aisle that was checked.",
    "A new forklift operator started this week and completed the safety training.",
    "The loading schedule for next week was posted on the board near the office.",
    "Two delivery trucks were rescheduled because of roadwork on the highway.",
    "The break room got new chairs and the old ones went to the storage annex.",
    "Maintenance replaced the light fixtures above the packing stations yesterday.",
]


def make_filler(rng: random.Random, approx_tokens: int) -> str:
    """Roughly approx_tokens of inert prose (≈1.3 tokens per word)."""
    words_needed = int(approx_tokens / 1.3)
    parts: list[str] = []
    count = 0
    while count < words_needed:
        s = rng.choice(_TEMPLATES)
        parts.append(s)
        count += len(s.split())
    return " ".join(parts)


def build_messages(mode: str, sku: str, filler: str) -> list[dict]:
    if mode == "near":
        user = (f"Background notes from today: {filler}\n\n"
                f"Now look up {sku}.") if filler else f"Look up {sku}."
    elif mode == "far":
        user = (f"Today's item of interest is {sku}.\n\n"
                f"Background notes from today: {filler}\n\n"
                "Now look up today's item of interest.") if filler else (
                f"Today's item of interest is {sku}. Look up today's item of interest.")
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def run_onset(client, *, lengths: list[int], reps: int, modes: list[str],
              seed_base: int) -> list[dict]:
    tools = tool_schemas()
    records: list[dict] = []
    for mode in modes:
        for length in lengths:
            for rep in range(reps):
                # str hash() is salted per-process — build a stable seed by index
                rng = random.Random(seed_base * 1_000_003 + lengths.index(length) * 10_007
                                    + modes.index(mode) * 101 + rep)
                sku = f"SKU-{rng.randint(1000, 9998)}"
                filler = make_filler(rng, length) if length > 0 else ""
                resp = client.chat(build_messages(mode, sku, filler), tools, max_tokens=128)
                call = extract_first_call(resp["message"])
                sku_actual = str((call["args"] or {}).get("sku", "")) if call["parse_valid"] else None
                rec: dict[str, Any] = {
                    "mode": mode,
                    "approx_tokens": length,
                    "rep": rep,
                    "sku_expected": sku,
                    "call_present": call["present"],
                    "parse_valid": call["parse_valid"] if call["present"] else None,
                    "tool_correct": call["name"] == "lookup_item" if call["present"] else None,
                    "sku_actual": sku_actual,
                    "sku_exact": (sku_actual is not None
                                  and sku_actual.strip().upper() == sku),
                    "assistant_text": (resp["message"].get("content") or "")[:300],
                    "usage": resp["usage"],
                    "latency_s": resp["latency_s"],
                }
                records.append(rec)
    return records


def summarize_onset(records: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["mode"], r["approx_tokens"]) for r in records})
    for mode, length in keys:
        rs = [r for r in records if r["mode"] == mode and r["approx_tokens"] == length]
        n = len(rs)
        out.append({
            "mode": mode,
            "approx_tokens": length,
            "n": n,
            "call_present_rate": round(sum(r["call_present"] for r in rs) / n, 4),
            "sku_exact_rate": round(sum(r["sku_exact"] for r in rs) / n, 4),
        })
    return out
