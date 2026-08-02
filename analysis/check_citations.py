#!/usr/bin/env python3
"""G-cite: machine gate for the milestone-5 write-up.

Every artifact citation in the paper has the form [[art:PATH@HASH8]] where
PATH is relative to research/kv-drift/ and HASH8 is the first 8 hex chars
of the file's sha256. The gate fails if any citation's file is missing or
its hash prefix does not match the file on disk, or if the paper contains
zero citations (a hash-linked paper cannot have none).

Usage: python3 check_citations.py <paper.md> [--min N]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

# Citation paths resolve relative to the problem directory: for a paper at
# research/<slug>/paper/paper.md that is research/<slug>/ (override: --base).
CITE_RE = re.compile(r"\[\[art:([^@\]]+)@([0-9a-f]{8})\]\]")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paper", type=Path)
    ap.add_argument("--min", type=int, default=10, help="minimum distinct citations")
    ap.add_argument("--base", type=Path, default=None,
                    help="directory artifact paths are relative to (default: paper's grandparent)")
    args = ap.parse_args()
    base = (args.base or args.paper.resolve().parent.parent)
    text = args.paper.read_text()
    cites = CITE_RE.findall(text)
    problems: list[str] = []
    seen: dict[str, str] = {}
    for rel, h8 in cites:
        if rel in seen and seen[rel] == h8:
            continue
        seen[rel] = h8
        p = base / rel
        if not p.is_file():
            problems.append(f"missing artifact: {rel}")
            continue
        actual = sha256_file(p)
        if not actual.startswith(h8):
            problems.append(f"hash mismatch: {rel} cited @{h8}, actual {actual[:8]}")
    if len(seen) < args.min:
        problems.append(f"only {len(seen)} distinct citations (< {args.min})")
    for pr in problems:
        print(f"[check-citations] FAIL: {pr}")
    if problems:
        print(f"[check-citations] G-cite FAIL ({len(problems)} problem(s))")
        return 1
    print(f"[check-citations] G-cite PASS ({len(seen)} distinct artifacts, all hashes match)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
