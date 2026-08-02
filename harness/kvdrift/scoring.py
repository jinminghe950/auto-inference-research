"""Per-turn scoring: syntax validity, tool selection, argument fidelity,
and free-text fact recall — the separate failure channels from H3."""

from __future__ import annotations

import json
import re
from typing import Any

from .trajectory import Turn


def _norm_str(v: Any) -> str:
    return str(v).strip().strip("'\"").casefold()


def _num_eq(a: Any, b: Any, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _field_match(spec: dict, value: Any) -> bool:
    if "eq" in spec:
        return _norm_str(value) == _norm_str(spec["eq"])
    if "eq_num" in spec:
        return _num_eq(value, spec["eq_num"])
    if "contains_ci" in spec:
        return _norm_str(spec["contains_ci"]) in _norm_str(value)
    if "nonempty" in spec:
        return bool(str(value).strip())
    return False


def extract_first_call(message: dict) -> dict:
    """Pull the first tool call out of an assistant message.

    Returns {present, parse_valid, name, args, raw_arguments, n_calls}.
    parse_valid means: a call is present AND its arguments parse as a JSON
    object. (The serving stack's tool parser has already accepted the
    surrounding structure; malformed inner JSON is the syntax channel.)
    """
    calls = message.get("tool_calls") or []
    out: dict[str, Any] = {"present": bool(calls), "n_calls": len(calls),
                           "parse_valid": False, "name": None, "args": None, "raw_arguments": None}
    if not calls:
        return out
    fn = (calls[0] or {}).get("function") or {}
    out["name"] = fn.get("name")
    raw = fn.get("arguments")
    out["raw_arguments"] = raw
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(args, dict):
            out["args"] = args
            out["parse_valid"] = True
    except (json.JSONDecodeError, TypeError):
        pass
    return out


def score_turn(turn: Turn, message: dict) -> dict:
    """Score one assistant response against the oracle expectation."""
    call = extract_first_call(message)
    text = message.get("content") or ""
    rec: dict[str, Any] = {
        "kind": turn.kind,
        "scoreable": turn.scoreable,
        "expected_tool": turn.expected_tool,
        "call_present": call["present"],
        "n_calls": call["n_calls"],
        "parse_valid": call["parse_valid"] if call["present"] else None,
        "tool_actual": call["name"],
        "args_actual": call["args"] if call["parse_valid"] else call["raw_arguments"],
        "tool_correct": None,
        "args_fidelity": None,
        "args_exact": None,
        "action_correct": None,
        "overcall": None,
        "recall_hit": None,
        "recall_detail": None,
    }

    if turn.expected_tool is not None:
        if not call["present"]:
            rec.update(tool_correct=False, args_fidelity=0.0, args_exact=False, action_correct=False)
            return rec
        rec["tool_correct"] = call["name"] == turn.expected_tool
        if not turn.scoreable:
            return rec
        spec = turn.expected_args
        if rec["tool_correct"] and call["parse_valid"] and spec:
            matches = {f: _field_match(s, call["args"].get(f)) for f, s in spec.items()}
            rec["args_fidelity"] = sum(matches.values()) / len(matches)
            rec["args_exact"] = all(matches.values())
            rec["args_field_matches"] = matches
        else:
            rec["args_fidelity"] = 0.0
            rec["args_exact"] = False
        rec["action_correct"] = bool(rec["tool_correct"] and rec["args_exact"])
        return rec

    # Free-text turn: calling any tool is an overcall; score fact recall.
    rec["overcall"] = call["present"]
    if turn.scoreable and turn.recall_facts:
        qty = turn.recall_facts.get("qty")
        bin_code = turn.recall_facts.get("bin")
        qty_hit = re.search(rf"\b{int(qty)}\b", text) is not None if qty is not None else None
        bin_hit = re.search(rf"\b{re.escape(str(bin_code))}\b", text, re.IGNORECASE) is not None if bin_code else None
        hits = [h for h in (qty_hit, bin_hit) if h is not None]
        rec["recall_hit"] = all(hits) if hits else None
        rec["recall_detail"] = {"qty_hit": qty_hit, "bin_hit": bin_hit,
                                "expected_qty": qty, "expected_bin": bin_code}
    return rec


def summarize(records: list[dict]) -> dict:
    """Aggregate one trajectory's per-turn records into channel rates."""
    tool_turns = [r for r in records if r["expected_tool"] is not None and r["scoreable"]]
    text_turns = [r for r in records if r["expected_tool"] is None]
    recall_turns = [r for r in text_turns if r["recall_hit"] is not None]

    def rate(rs: list[dict], key: str) -> float | None:
        vals = [r[key] for r in rs if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "n_turns": len(records),
        "n_tool_turns": len(tool_turns),
        "n_text_turns": len(text_turns),
        "call_present_rate": rate(tool_turns, "call_present"),
        "parse_valid_rate": rate([r for r in tool_turns if r["call_present"]], "parse_valid"),
        "tool_correct_rate": rate(tool_turns, "tool_correct"),
        "args_fidelity_mean": rate(tool_turns, "args_fidelity"),
        "action_correct_rate": rate(tool_turns, "action_correct"),
        "exec_error_rate": rate(tool_turns, "exec_error"),
        "overcall_rate": rate(text_turns, "overcall"),
        "recall_hit_rate": rate(recall_turns, "recall_hit"),
        "n_recall_turns": len(recall_turns),
    }
