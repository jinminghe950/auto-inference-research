#!/usr/bin/env python3
"""Machine gate for an m1 onset group, run locally on collected copies.

Checks (pre-registered in ../milestone-1.md):
  1. row count: onset.jsonl has exactly 2 modes x 8 lengths x 20 reps = 320
     rows for the group's single config;
  2. manifest: every manifested file re-hashes cleanly and no result file
     on disk is unmanifested (kvdrift.manifest.verify_manifest, code side
     checked against this repo's harness/kvdrift — the tarball uploaded to
     the pod is built from the same tree).

Usage: python3 verify_onset.py <onset_group_dir> [--harness <harness_dir>]
Exit 0 iff both checks pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED_ROWS = 2 * 8 * 20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onset_dir", type=Path)
    ap.add_argument("--harness", type=Path,
                    default=Path(__file__).resolve().parents[3] / "harness")
    args = ap.parse_args()
    sys.path.insert(0, str(args.harness))
    from kvdrift.manifest import verify_manifest

    problems: list[str] = []
    jsonls = sorted(args.onset_dir.glob("config_*/onset.jsonl"))
    if len(jsonls) != 1:
        problems.append(f"expected exactly 1 config onset.jsonl, found {len(jsonls)}")
    for p in jsonls:
        n = sum(1 for l in p.read_text().splitlines() if l.strip())
        if n != EXPECTED_ROWS:
            problems.append(f"{p.parent.name}: {n}/{EXPECTED_ROWS} rows")
    problems += verify_manifest(args.onset_dir, args.harness / "kvdrift")

    if problems:
        for p in problems:
            print(f"[verify_onset] FAIL: {p}")
        print(f"[verify_onset] GATE FAIL ({len(problems)} problem(s))")
        return 1
    print(f"[verify_onset] {args.onset_dir.name}: rows {EXPECTED_ROWS}/{EXPECTED_ROWS}, "
          "manifest clean: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
