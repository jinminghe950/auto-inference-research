# auto-inference-research

An autonomous lab for **LLM inference research**. There is no app and no
orchestrator here: **the Claude Code session is the agent**; this repo is
the lab — the method, the instruments, the problem queue, and the evidence.
One session executes one research problem end to end: problem in →
gate-verified, adversarially audited paper + content-addressed artifacts
out.

Scope: LLM inference only (serving stacks, KV caches, quantization,
batching/caching, decoding, kernels — as shipped). Chosen because inference
problems are economically loaded and, crucially, **executably verifiable**:
nearly every gate here is machine-checked, which is what makes full
autonomy credible. Prefer study designs whose quality metrics are exact
(structured outputs, identifier fidelity, latency/memory) over judged ones.

## Session protocol

When the user names a problem (from `problems/`, or a new one — file it
first), execute this loop autonomously until done, reporting at milestone
boundaries. Read `METHOD.md` before starting; it is the contract.

1. **Collision scan** — independent search agents over recent literature,
   engine changelogs, and GitHub issues; verdict + nearest-neighbor table +
   sharpened claim + milestone/budget plan. Commit as
   `research/<slug>/collision-scan.md` before anything else.
2. **Per milestone:**
   - *Pre-register*: frozen config matrix (sha256-pinned), gate criteria,
     and frozen analysis code, committed **before any data exists**. The
     commit ordering is the proof.
   - *Execute mechanically* via the `gated-experiment` skill (Runpod
     executor, token-auth control plane, no credentials on the pod). Never
     improvise infra; never edit frozen files mid-run.
   - *Machine gates* (G0–G4, `harness/`): verified on the pod AND
     re-verified locally on collected copies. No agent may override a
     machine gate.
   - *Adversarial audit* per `AUDIT.md` (evidence / methods / provenance
     critics; blocking objections need two independent confirmations).
   - Pass → next milestone. Objections → bounded revision (≤3 attempts)
     with every amendment logged in the milestone doc's Deviations section,
     versioned never edited, and screened per METHOD.md's amendment policy.
3. **Paper** — written from manifested artifacts only, every empirical
   claim cited as `[[art:path@hash8]]`, gated by
   `analysis/check_citations.py` (G-cite) plus a final AUDIT.md referee
   pass. LaTeX mirror for arXiv if the user wants it.
4. **Attention states — stop and report to the user instead of
   proceeding**: budget stop-loss reached, an amendment screened as
   result-shopping, revision attempts exhausted, or anything requiring a
   frozen file to change semantics after data exists.

## Layout

- `METHOD.md` — the gated method (invariants, amendment policy, budgets)
- `AUDIT.md` — the frozen adversarial-panel protocol
- `.claude/skills/gated-experiment/` — the executor runbook
- `harness/` — the lab equipment (gates, manifests, engines, pod control
  plane, tool-calling instrument); `harness/README.md` maps what is
  domain-agnostic vs instrument-specific; `python3 harness/selftest.py`
  must pass before any instrument change lands
- `analysis/` — G-cite checker; frozen-analyzer pattern example
- `problems/` — the queue (format + scoring in its README)
- `research/<slug>/` — one directory per executed problem

## Standing rules

- Trust evidence, never the executor's narrative; critics recompute from
  raw artifacts.
- Negative and unresolved results are first-class deliverables.
- Budgets: default $15 stop-loss per milestone, $150 per problem; override
  only in a pre-registration, never mid-run.
- Secrets: pods receive only the per-pod random token; sweep artifacts for
  credentials before every evidence commit.
- Executed case study and provenance conventions: the KV-drift program in
  `jinminghe950/agentic-research` (`research/kv-drift/`), whose deviation
  logs are the calibration set for the amendment screen.
