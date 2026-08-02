# research/ — one directory per executed problem

`research/<slug>/` (slug from the problem file, e.g. `001-int8-kv-cross-family`):

```
collision-scan.md      milestone 1: verdict, nearest neighbors, sharpened
                       claim, milestone/budget plan
milestone-N.md         one per milestone: pre-registration on top (frozen
                       matrix hashes, gates, analysis plan, budget),
                       Deviations log in the middle, Results + Audit
                       sections appended after the gates
matrices/              problem-local frozen config matrices, versioned
                       (matrix-v2.json …), never edited
runs/<name>/           collected evidence: per-turn JSONL, summaries,
                       metrics snapshots, server logs, run_meta.json,
                       manifest.json — landed for failed gates too
analysis.json          frozen-analyzer output per run (inside runs/<name>/)
paper/paper.md         hash-cited canonical draft (G-cite gated)
paper/paper.tex        arXiv mirror, generated from the md when wanted
reports/               ready-to-file upstream issue drafts
```

Citation base for `[[art:path@hash8]]` is `research/<slug>/`
(`analysis/check_citations.py` resolves relative to the paper's
grandparent directory).
