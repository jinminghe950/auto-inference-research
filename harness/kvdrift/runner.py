"""Drive one trajectory against a running server and record per-turn evidence."""

from __future__ import annotations

import json
import time
from typing import Any

from .client import ChatClient
from .scoring import extract_first_call, score_turn
from .trajectory import ScriptedUser
from .warehouse import Warehouse, tool_schemas


def run_trajectory(client: ChatClient, *, sim_seed: int, n_turns: int,
                   max_tokens: int = 400) -> list[dict]:
    """Live (self-conditioned) mode: the model's own outputs and its tool
    results stay in the transcript, exactly as a production agent would see
    them. Scoring judges each turn against the oracle's expectation given
    the instruction and the current true simulator state."""
    sim = Warehouse(sim_seed)
    user = ScriptedUser(sim_seed, sim, n_turns)
    tools = tool_schemas()
    messages: list[dict] = [{"role": "system", "content": user.system_prompt()}]
    records: list[dict] = []

    for idx in range(n_turns):
        turn = user.next_turn(idx)
        messages.append({"role": "user", "content": turn.instruction})
        resp = client.chat(messages, tools, max_tokens=max_tokens)
        message = resp["message"]
        rec: dict[str, Any] = {
            "turn": idx,
            "instruction": turn.instruction,
            "expected_args": turn.expected_args,
            "recall_ref_turn": turn.recall_ref_turn,
            **score_turn(turn, message),
            "assistant_text": message.get("content") or "",
            "finish_reason": resp["finish_reason"],
            "usage": resp["usage"],
            "latency_s": resp["latency_s"],
            "ts": time.time(),
        }

        # Append the assistant message verbatim, then execute every tool call
        # in order and append its result — errors included. The transcript is
        # the model's reality; the simulator is ours.
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        exec_results = []
        for call in tool_calls:
            fn = (call or {}).get("function") or {}
            name = fn.get("name") or ""
            raw = fn.get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            result = sim.execute(name, args)
            exec_results.append({"name": name, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id") or f"call_{idx}_{len(exec_results)}",
                "content": json.dumps(result, separators=(",", ":")),
            })
            user.observe_tool_result(turn, name, args, result)

        first = extract_first_call(message)
        rec["exec_error"] = (
            bool(exec_results and isinstance(exec_results[0]["result"], dict)
                 and "error" in exec_results[0]["result"])
            if (turn.expected_tool is not None and first["present"]) else None
        )
        rec["exec_results"] = exec_results
        records.append(rec)

    return records
