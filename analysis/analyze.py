#!/usr/bin/env python3
"""Pre-registered analysis for the milestone-3 sweep (frozen before data).

Implements exactly the plan in ../milestone-3.md §Analysis plan:
  1. per-config channel rates by turn bin, bootstrap 95% CIs over seeds;
  2. H1: linear slope of error rate over turn bins, per config;
  3. H2: reuse − recompute contrast per precision (aggregate + deepest bin),
     paired over seeds;
  4. H3: channel decomposition (semantic vs syntax vs recall) per config/bin.

Stdlib only, deterministic (fixed bootstrap seed). Reads manifested JSONL
under a results dir; writes analysis.json next to it and prints a markdown
summary. No numbers outside this script's output are reported as results.

Usage: python3 analyze.py <results_dir> [--out analysis.json]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

BOOT_N = 10_000
BOOT_SEED = 20260802
BINS = [(0, 9), (10, 19), (20, 29), (30, 39)]

# channel -> (record predicate for denominator, error predicate)
def _tool_turns(r):
    return r.get("expected_tool") is not None and r.get("scoreable")


CHANNELS = {
    "action_error": (_tool_turns, lambda r: not r.get("action_correct")),
    "no_call": (_tool_turns, lambda r: not r.get("call_present")),
    "parse_invalid": (lambda r: _tool_turns(r) and r.get("call_present"),
                      lambda r: not r.get("parse_valid")),
    "wrong_tool": (lambda r: _tool_turns(r) and r.get("call_present") and r.get("parse_valid"),
                   lambda r: not r.get("tool_correct")),
    "args_wrong": (lambda r: _tool_turns(r) and r.get("tool_correct"),
                   lambda r: not r.get("args_exact")),
    "exec_error": (lambda r: _tool_turns(r) and r.get("exec_error") is not None,
                   lambda r: bool(r.get("exec_error"))),
    "recall_miss": (lambda r: r.get("recall_hit") is not None,
                    lambda r: not r.get("recall_hit")),
    "overcall": (lambda r: r.get("overcall") is not None,
                 lambda r: bool(r.get("overcall"))),
}


def load(results_dir: Path) -> dict[str, list[list[dict]]]:
    matrix = json.loads((results_dir / "run_meta.json").read_text())["matrix"]
    out: dict[str, list[list[dict]]] = {}
    for cfg in matrix["configs"]:
        trajs = []
        for p in sorted((results_dir / f"config_{cfg['name']}").glob("trajectory_*.jsonl")):
            trajs.append([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
        out[cfg["name"]] = trajs
    return out


def pooled_rate(trajs: list[list[dict]], channel: str, lo: int = 0, hi: int = 10**9,
                seed_idx: list[int] | None = None) -> float | None:
    denom_p, err_p = CHANNELS[channel]
    idx = range(len(trajs)) if seed_idx is None else seed_idx
    num = den = 0
    for i in idx:
        for r in trajs[i]:
            if lo <= r.get("turn", 0) <= hi and denom_p(r):
                den += 1
                num += bool(err_p(r))
    return num / den if den else None


def bootstrap_ci(fn, n_seeds: int, rng: random.Random) -> tuple[float, float] | None:
    vals = []
    for _ in range(BOOT_N):
        sample = [rng.randrange(n_seeds) for _ in range(n_seeds)]
        v = fn(sample)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[min(int(0.975 * len(vals)), len(vals) - 1)])


def ols_slope(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(pairs) < 2:
        return None
    xs2 = [p[0] for p in pairs]
    ys2 = [p[1] for p in pairs]
    mx, my = statistics.mean(xs2), statistics.mean(ys2)
    denom = sum((x - mx) ** 2 for x in xs2)
    return sum((x - mx) * (y - my) for x, y in zip(xs2, ys2)) / denom if denom else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    data = load(args.results_dir)
    rng = random.Random(BOOT_SEED)
    report: dict = {"bins": BINS, "boot_n": BOOT_N, "boot_seed": BOOT_SEED,
                    "configs": {}, "h2_contrasts": {}}

    mids = [(lo + hi) / 2 for lo, hi in BINS]
    for name, trajs in data.items():
        n = len(trajs)
        cfg_rep: dict = {"n_seeds": n, "channels": {}}
        for ch in CHANNELS:
            per_bin = []
            for lo, hi in BINS:
                rate = pooled_rate(trajs, ch, lo, hi)
                ci = bootstrap_ci(lambda s, lo=lo, hi=hi, ch=ch: pooled_rate(trajs, ch, lo, hi, s), n, rng) if n > 1 else None
                per_bin.append({"bin": [lo, hi], "rate": rate, "ci95": ci})
            agg = pooled_rate(trajs, ch)
            agg_ci = bootstrap_ci(lambda s, ch=ch: pooled_rate(trajs, ch, seed_idx=s), n, rng) if n > 1 else None
            cfg_rep["channels"][ch] = {"aggregate": agg, "aggregate_ci95": agg_ci, "by_bin": per_bin}
        # H1 slope on action_error
        ys = [b["rate"] for b in cfg_rep["channels"]["action_error"]["by_bin"]]
        slope = ols_slope(mids, ys)
        slope_ci = bootstrap_ci(
            lambda s: ols_slope(mids, [pooled_rate(trajs, "action_error", lo, hi, s) for lo, hi in BINS]),
            n, rng) if n > 1 else None
        cfg_rep["h1_slope_per_turn"] = {"slope": slope, "ci95": slope_ci}
        report["configs"][name] = cfg_rep

    # H2: reuse - recompute per stem, paired over seeds
    stems = sorted({n.rsplit("-", 1)[0] for n in data if n.endswith("-reuse")})
    for stem in stems:
        a, b = f"{stem}-reuse", f"{stem}-recompute"
        if a not in data or b not in data or not data[a] or not data[b]:
            continue
        n = min(len(data[a]), len(data[b]))

        def diff(seed_idx, lo=0, hi=10**9):
            ra = pooled_rate(data[a], "action_error", lo, hi, seed_idx)
            rb = pooled_rate(data[b], "action_error", lo, hi, seed_idx)
            return None if ra is None or rb is None else ra - rb

        entry = {}
        for label, (lo, hi) in [("aggregate", (0, 10**9)), ("deepest_bin", BINS[-1])]:
            d = diff(list(range(n)), lo, hi)
            ci = bootstrap_ci(lambda s, lo=lo, hi=hi: diff(s, lo, hi), n, rng) if n > 1 else None
            entry[label] = {"reuse_minus_recompute": d, "ci95": ci}
        report["h2_contrasts"][stem] = entry

    out_path = args.out or (args.results_dir / "analysis.json")
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    # Markdown summary
    print(f"# Sweep analysis ({args.results_dir.name})\n")
    print("| config | action_error (95% CI) | bin 0-9 | bin 30-39 | H1 slope/turn (CI) |")
    print("|---|---|---|---|---|")
    for name, c in report["configs"].items():
        ae = c["channels"]["action_error"]
        def fmt(x):
            return "n/a" if x is None else f"{x:.3f}"
        def fmtci(ci):
            return "n/a" if not ci else f"[{ci[0]:.3f},{ci[1]:.3f}]"
        s = c["h1_slope_per_turn"]
        slope_txt = "n/a" if s["slope"] is None else f"{s['slope']:.4f}"
        print(f"| {name} | {fmt(ae['aggregate'])} {fmtci(ae['aggregate_ci95'])} "
              f"| {fmt(ae['by_bin'][0]['rate'])} | {fmt(ae['by_bin'][-1]['rate'])} "
              f"| {slope_txt} {fmtci(s['ci95'])} |")
    print("\n| precision | H2 reuse−recompute agg (CI) | deepest bin (CI) |")
    print("|---|---|---|")
    for stem, e in report["h2_contrasts"].items():
        def f(v):
            return "n/a" if v is None else f"{v:+.3f}"
        def fc(ci):
            return "n/a" if not ci else f"[{ci[0]:+.3f},{ci[1]:+.3f}]"
        print(f"| {stem} | {f(e['aggregate']['reuse_minus_recompute'])} {fc(e['aggregate']['ci95'])} "
              f"| {f(e['deepest_bin']['reuse_minus_recompute'])} {fc(e['deepest_bin']['ci95'])} |")
    print(f"\nFull numbers: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
