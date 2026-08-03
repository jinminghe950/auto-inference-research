#!/usr/bin/env python3
"""Pre-registered analysis for fp8-kv-hopper milestone 1 (frozen before data).

Implements exactly the plan in ../milestone-1.md §Analysis plan. Layout:
the runs root contains up to six group directories, each an independent
harness run with its own manifest and gate verdict:

  m1-sweep-fp16/   trajectories: vllm-fp16-{reuse,recompute}      (control, G4 floor)
  m1-sweep-e4m3/   trajectories: vllm-fp8e4m3-{reuse,recompute}   (marketed path audit)
  m1-sweep-e5m2/   trajectories: vllm-fp8e5m2-{reuse,recompute}   (regime under test)
  m1-onset-fp16/   onset grid:   vllm-fp16                        (instrument sanity)
  m1-onset-e4m3/   onset grid:   vllm-fp8e4m3
  m1-onset-e5m2/   onset grid:   vllm-fp8e5m2

Groups are split so an engine-level crash in one precision (a registered
possible outcome on sm_90, cf. vllm#31843) cannot destroy the manifested
evidence of the others.

Computed strictly from raw artifacts (trajectory_*.jsonl, onset.jsonl,
probe.json); summary.json is never trusted. The repo-frozen
analysis/analyze.py is additionally run per sweep group for CIs/bins.

Registered decision rules:
  Instrument sanity — the fp16 onset control must show sku_exact_rate
  >= 0.90 and call_present_rate >= 0.95 in every cell AND the fp16 sweep
  group must be complete; otherwise verdict INSTRUMENT_FAIL and no regime
  claim may be made.

  Regime classification for vllm-fp8e5m2 on this host:
    ERROR    — the e5m2 sweep or onset group is absent/incomplete because
               the engine failed to serve (evidence: server logs). A loud
               deployment failure on sm_90, reported as such.
    SALAD    — sweep G0 coherence probe failed (text_ok false) for any
               e5m2 config: worse than the Ada/Ampere regime.
    PERSISTS — onset: call_present_rate == 0 in ALL e5m2 cells; sweep:
               pooled no_call >= 0.99 over e5m2 tool turns. The
               Ada/Ampere signature replicated on sm_90.
    CLEAN    — onset: every e5m2 cell has call_present_rate >= 0.95 and
               sku_exact_rate >= (fp16 same-cell - 0.05); sweep: pooled
               e5m2 action_error <= 0.05. Hopper escapes the failure.
    MIXED    — anything else: a measurable intermediate regime exists;
               triggers the pre-registered milestone-2 condition.
  The e4m3 legs are reported with identical statistics (they audit the
  configuration vLLM's 2026-04-22 blog markets as default-ready on
  Hopper) and additionally receive the same classification labels for
  reporting, but e4m3 never gates and never enters the e5m2 regime call.

Stdlib only, deterministic. Usage:
  python3 analyze_m1.py <runs_root> [--out analysis-m1.json]
No numbers outside this script's output (and analyze.py's) are reported
as milestone results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GROUPS = ["fp16", "e4m3", "e5m2"]
SWEEP_CONFIGS = {
    "fp16": ["vllm-fp16-reuse", "vllm-fp16-recompute"],
    "e4m3": ["vllm-fp8e4m3-reuse", "vllm-fp8e4m3-recompute"],
    "e5m2": ["vllm-fp8e5m2-reuse", "vllm-fp8e5m2-recompute"],
}
ONSET_CONFIGS = {"fp16": "vllm-fp16", "e4m3": "vllm-fp8e4m3", "e5m2": "vllm-fp8e5m2"}


def onset_cells(rows: list[dict]) -> list[dict]:
    cells = []
    for key in sorted({(r["mode"], r["approx_tokens"]) for r in rows}):
        rs = [r for r in rows if (r["mode"], r["approx_tokens"]) == key]
        n = len(rs)
        cells.append({
            "mode": key[0], "approx_tokens": key[1], "n": n,
            "call_present_rate": sum(bool(r["call_present"]) for r in rs) / n,
            "parse_valid_rate": sum(bool(r.get("parse_valid")) for r in rs) / n,
            "tool_correct_rate": sum(bool(r.get("tool_correct")) for r in rs) / n,
            "sku_exact_rate": sum(bool(r["sku_exact"]) for r in rs) / n,
        })
    return cells


def load_onset_group(runs_root: Path, group: str) -> dict:
    d = runs_root / f"m1-onset-{group}"
    cfg_dir = d / f"config_{ONSET_CONFIGS[group]}"
    f = cfg_dir / "onset.jsonl"
    if not f.is_file():
        return {"present": False, "cells": [], "n_rows": 0}
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return {"present": True, "cells": onset_cells(rows), "n_rows": len(rows)}


def load_sweep_group(runs_root: Path, group: str) -> dict:
    d = runs_root / f"m1-sweep-{group}"
    names = SWEEP_CONFIGS[group]
    out: dict = {"present": False, "configs": names, "flat": [],
                 "probe_text_ok": {}, "n_expected": None}
    meta = d / "run_meta.json"
    if not meta.is_file():
        return out
    matrix = json.loads(meta.read_text())["matrix"]
    out["n_expected"] = int(matrix["trajectories"]) * len(names)
    n_found = 0
    for name in names:
        cfg_dir = d / f"config_{name}"
        for p in sorted(cfg_dir.glob("trajectory_*.jsonl")):
            out["flat"] += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            n_found += 1
        if (cfg_dir / "probe.json").is_file():
            out["probe_text_ok"][name] = json.loads(
                (cfg_dir / "probe.json").read_text()).get("text_ok")
    out["present"] = True
    out["n_trajectories_found"] = n_found
    out["complete"] = n_found == out["n_expected"]
    return out


def pooled(flat: list[dict], err_key: str) -> float | None:
    num = den = 0
    for r in flat:
        if r.get("expected_tool") is not None and r.get("scoreable"):
            den += 1
            if err_key == "action_error":
                num += not r.get("action_correct")
            elif err_key == "no_call":
                num += not r.get("call_present")
    return num / den if den else None


def classify(group: str, sweep: dict, onset: dict, fp16_cells: list[dict]) -> tuple[str, dict]:
    """Registered rule, applied to a quantized group."""
    fp16_by_cell = {(c["mode"], c["approx_tokens"]): c for c in fp16_cells}
    cells = onset["cells"]
    err = (not onset["present"] or not sweep["present"]
           or not sweep.get("complete") or not cells)
    salad = any(v is False for v in sweep["probe_text_ok"].values())
    no_call = pooled(sweep["flat"], "no_call")
    action_error = pooled(sweep["flat"], "action_error")
    onset_dead = bool(cells) and all(c["call_present_rate"] == 0 for c in cells)
    traj_dead = no_call is not None and no_call >= 0.99
    onset_clean = bool(cells) and all(
        c["call_present_rate"] >= 0.95
        and c["sku_exact_rate"] >= fp16_by_cell.get(
            (c["mode"], c["approx_tokens"]), {"sku_exact_rate": 1.0})["sku_exact_rate"] - 0.05
        for c in cells)
    traj_clean = action_error is not None and action_error <= 0.05
    inputs = {"salad": salad, "onset_dead": onset_dead, "traj_dead": traj_dead,
              "onset_clean": onset_clean, "traj_clean": traj_clean,
              "sweep_complete": sweep.get("complete"), "onset_present": onset["present"],
              "action_error": action_error, "no_call": no_call}
    if salad:
        return "SALAD", inputs
    if err:
        return "ERROR", inputs
    if onset_dead and traj_dead:
        return "PERSISTS", inputs
    if onset_clean and traj_clean:
        return "CLEAN", inputs
    return "MIXED", inputs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    onset = {g: load_onset_group(args.runs_root, g) for g in GROUPS}
    sweep = {g: load_sweep_group(args.runs_root, g) for g in GROUPS}

    report: dict = {"onset": {}, "sweep": {}, "instrument_sanity": None,
                    "regime": None, "e4m3_label": None}
    for g in GROUPS:
        report["onset"][g] = {k: v for k, v in onset[g].items() if k != "cells"} | {
            "cells": onset[g]["cells"]}
        s = sweep[g]
        report["sweep"][g] = {
            "present": s["present"], "complete": s.get("complete"),
            "configs": s["configs"], "probe_text_ok": s["probe_text_ok"],
            "action_error": pooled(s["flat"], "action_error"),
            "no_call": pooled(s["flat"], "no_call"),
            "n_tool_turns": sum(1 for r in s["flat"]
                                if r.get("expected_tool") is not None and r.get("scoreable")),
        }

    fp16_cells = onset["fp16"]["cells"]
    sane = (bool(fp16_cells)
            and all(c["sku_exact_rate"] >= 0.90 and c["call_present_rate"] >= 0.95
                    for c in fp16_cells)
            and sweep["fp16"].get("complete") is True)
    report["instrument_sanity"] = "PASS" if sane else "INSTRUMENT_FAIL"

    e5m2_label, e5m2_inputs = classify("e5m2", sweep["e5m2"], onset["e5m2"], fp16_cells)
    e4m3_label, e4m3_inputs = classify("e4m3", sweep["e4m3"], onset["e4m3"], fp16_cells)
    report["regime"] = e5m2_label if sane else "INSTRUMENT_FAIL"
    report["e4m3_label"] = e4m3_label
    report["classification_inputs"] = {"e5m2": e5m2_inputs, "e4m3": e4m3_inputs}

    out_path = args.out or (args.runs_root / "analysis-m1.json")
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    print("# fp8-kv-hopper m1 analysis\n")
    print(f"instrument_sanity: {report['instrument_sanity']}")
    print(f"regime (vllm-fp8e5m2 on this host): **{report['regime']}**")
    print(f"e4m3 label (reporting only, never gates): {report['e4m3_label']}\n")
    print("| sweep group | complete | action_error | no_call | n tool-turns |")
    print("|---|---|---|---|---|")
    for g in GROUPS:
        s = report["sweep"][g]
        def f(x):
            return "n/a" if x is None else f"{x:.3f}"
        print(f"| {g} | {s['complete']} | {f(s['action_error'])} | {f(s['no_call'])} "
              f"| {s['n_tool_turns']} |")
    print("\n| onset group | mode | tokens | call_present | sku_exact |")
    print("|---|---|---|---|---|")
    for g in GROUPS:
        for c in report["onset"][g]["cells"]:
            print(f"| {g} | {c['mode']} | {c['approx_tokens']} "
                  f"| {c['call_present_rate']:.2f} | {c['sku_exact_rate']:.2f} |")
    print(f"\nFull numbers: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
