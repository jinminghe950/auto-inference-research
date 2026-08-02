#!/usr/bin/env python3
"""Offline self-check for the harness — no GPU, no network, no vLLM.

Drives the real runner/scoring path with two fake models:
  1. a mirror model that always performs the oracle action  -> every channel
     must score perfect;
  2. a faulty model with planted defects                    -> each defect
     must land in exactly the channel built to catch it.
Then round-trips the manifest (build, verify, tamper, verify-fails).

Run: python3 selftest.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kvdrift.manifest import verify_manifest, write_manifest
from kvdrift.runner import run_trajectory
from kvdrift.scoring import summarize
from kvdrift.trajectory import ScriptedUser
from kvdrift.warehouse import Warehouse

SEED = 4242
TURNS = 16


def _spec_value(field: str, spec: dict):
    if "eq" in spec:
        return spec["eq"]
    if "eq_num" in spec:
        return spec["eq_num"]
    if "contains_ci" in spec:
        return f"cycle count for bin {spec['contains_ci']} is complete"
    return "delivery"


class MirrorModel:
    """Duck-types ChatClient.chat; keeps a lockstep copy of sim + script and
    plays the oracle action every turn (optionally corrupted)."""

    def __init__(self, seed: int, n_turns: int, corrupt: dict[int, str] | None = None):
        self.sim = Warehouse(seed)
        self.user = ScriptedUser(seed, self.sim, n_turns)
        self.idx = 0
        self.corrupt = corrupt or {}

    def chat(self, messages, tools, **kw):
        turn = self.user.next_turn(self.idx)
        mode = self.corrupt.get(self.idx)
        message: dict = {"content": None}

        if mode == "no_call" or (turn.expected_tool is None and mode != "overcall"):
            if turn.expected_tool is None and turn.recall_facts:
                q, b = turn.recall_facts["qty"], turn.recall_facts["bin"]
                if mode == "bad_recall":
                    q = q + 1
                message["content"] = f"It had {q} units on hand and was stored in bin {b}."
            else:
                message["content"] = "Understood."
        else:
            tool = turn.expected_tool or "lookup_item"
            args = {f: _spec_value(f, s) for f, s in turn.expected_args.items()}
            if turn.expected_tool is None:  # forced overcall on a text turn
                args = {"sku": self.user.sku_schedule[0]}
            if mode == "wrong_tool":
                tool = "audit_bin"
                args = {"bin": "A1"}
            elif mode == "bad_args" and "qty" in args:
                args["qty"] = int(args["qty"]) + 1
            message["tool_calls"] = [{
                "id": f"mirror_{self.idx}",
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(args)},
            }]

        # Mirror the actual behaviour on our own sim to stay in lockstep.
        for call in message.get("tool_calls") or []:
            fn = call["function"]
            args = json.loads(fn["arguments"])
            result = self.sim.execute(fn["name"], args)
            self.user.observe_tool_result(turn, fn["name"], args, result)

        self.idx += 1
        return {"message": message, "finish_reason": "stop",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 0},
                "latency_s": 0.0}


def check(cond: bool, label: str, failures: list[str]):
    print(f"  {'ok' if cond else 'FAIL'}  {label}")
    if not cond:
        failures.append(label)


def main() -> int:
    failures: list[str] = []

    print("[selftest] perfect mirror model")
    records = run_trajectory(MirrorModel(SEED, TURNS), sim_seed=SEED, n_turns=TURNS)
    s = summarize(records)
    check(s["n_turns"] == TURNS, "all turns ran", failures)
    check(s["action_correct_rate"] == 1.0, f"action_correct == 1.0 (got {s['action_correct_rate']})", failures)
    check(s["parse_valid_rate"] == 1.0, "parse_valid == 1.0", failures)
    check(s["exec_error_rate"] == 0.0, f"exec_error == 0.0 (got {s['exec_error_rate']})", failures)
    check(s["overcall_rate"] == 0.0, "overcall == 0.0", failures)
    check(s["recall_hit_rate"] == 1.0 and s["n_recall_turns"] >= 2,
          f"recall perfect over >=2 probes (got {s['recall_hit_rate']} / {s['n_recall_turns']})", failures)

    print("[selftest] faulty model: each defect lands in its channel")
    corrupt = {1: "bad_args", 5: "wrong_tool", 7: "bad_recall", 8: "no_call"}
    records = run_trajectory(MirrorModel(SEED, TURNS, corrupt), sim_seed=SEED, n_turns=TURNS)
    by = {r["turn"]: r for r in records}
    check(by[1]["args_exact"] is False and by[1]["exec_error"] is True,
          "bad qty -> args_exact False + sim rejects", failures)
    check(by[1]["tool_correct"] is True, "bad qty leaves tool_correct True", failures)
    check(by[5]["tool_correct"] is False, "wrong tool -> tool_correct False", failures)
    check(by[7]["recall_hit"] is False, "wrong recalled qty -> recall_hit False", failures)
    check(by[8]["call_present"] is False and by[8]["action_correct"] is False,
          "no call on tool turn -> call_present False", failures)
    unaffected = [r for r in records if r["turn"] not in corrupt and r["expected_tool"] and r["scoreable"]]
    check(all(r["action_correct"] for r in unaffected), "untouched turns stay perfect", failures)

    print("[selftest] prometheus counter parsing")
    from kvdrift.main import _prom_counter_total
    prom = ('vllm:prefix_cache_hits_total{engine="0"} 100.0\n'
            'vllm:prefix_cache_hits_created{engine="0"} 1.78e+09\n'
            'vllm:prefix_cache_hits_total{engine="1"} 20.0\n')
    check(_prom_counter_total(prom, "vllm:prefix_cache_hits_total") == 120.0,
          "sums exact metric across label sets, ignores _created gauge", failures)
    check(_prom_counter_total(prom, "vllm:gpu_prefix_cache_hits_total") == -1.0,
          "absent metric reports -1", failures)

    print("[selftest] manifest round-trip")
    code_dir = Path(__file__).resolve().parent / "kvdrift"
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "results"
        out.mkdir()
        (out / "a.jsonl").write_text('{"x":1}\n')
        write_manifest(out, code_dir, {"selftest": True})
        check(verify_manifest(out, code_dir) == [], "clean verify passes", failures)
        (out / "a.jsonl").write_text('{"x":2}\n')
        check(any("hash mismatch" in p for p in verify_manifest(out, code_dir)),
              "tamper detected", failures)
        (out / "stray.txt").write_text("not manifested")
        check(any("not in manifest" in p for p in verify_manifest(out, code_dir)),
              "unmanifested file detected", failures)

    if failures:
        print(f"[selftest] FAIL ({len(failures)}): {failures}")
        return 1
    print("[selftest] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
