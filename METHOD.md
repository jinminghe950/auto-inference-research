# The gated method

The credibility contract for everything produced in this repo. The session
executes; these rules decide what counts. Validated end-to-end by the
KV-drift program (`jinminghe950/agentic-research`, `research/kv-drift/`,
five milestones, ~$10 GPU) — cited below as "the case study".

## Why gates

Autonomous research fails in a characteristic way: fluent papers whose
numbers the artifacts do not support, and search processes that quietly
optimize toward publishable outcomes. Both are structural, not incidental
(see the independent evaluation of AI-Scientist-v1: hallucinated numerics
in 4/7 manuscripts). The method makes them impossible rather than
unlikely: claims are mechanically bound to hashed artifacts, verification
reads artifacts not prose, and confirmatory designs are frozen before data
exists. Exploration may be outcome-guided; **confirmation may not**.

## The milestone state machine

`pre-registered (commit) → running → evidence-collected → machine-gated →
audited → {approved → next | revision (≤3, logged) | attention (human)}`

### Pre-registration

Committed before any data it governs: the config matrix (sha256-pinned in
the milestone doc), gate criteria including any thresholds, the analysis
code, seeds, and the milestone budget. The commit ordering *is* the
ordering proof — never rewrite history to fake it.

### Machine gates (implemented in `harness/`, non-negotiable)

- **G0** pre-flight coherence probes per serving config — a broken config
  fails as INFRA before consuming trajectories; corrupted-but-measurable
  output is DATA, not infra (the case study's A1/A2 lesson: gating too
  strictly excludes the measurand).
- **G1** completeness: every pre-registered trajectory × turn present.
- **G2** toggle/treatment evidence: the manipulated variable must be
  *demonstrated* active/inactive from server metrics, usage counters, or a
  pre-registered physical signature (e.g. paired latency) — never assumed.
- **G3** sha256 manifest over results **and** the exact harness code;
  re-verified locally on collected copies. No hash, no verdict. A stored
  run is bound to the code that produced it — re-verify historical runs at
  their landing commit.
- **G4** matrix-frozen thresholds (e.g. a baseline sanity floor: if the
  control config can't pass, the experiment cannot attribute effects).
- **G-cite** (papers): every empirical claim carries `[[art:path@hash8]]`,
  machine-verified by `analysis/check_citations.py`.

Failed gates land in the repo exactly like passed ones — fail closed with
evidence preserved. No agent, panel, or session may override a machine
gate; that escape hatch belongs to the human, in writing.

### Adversarial audit

Per `AUDIT.md`, after machine gates. Critics read only manifested
artifacts and gate outputs, recompute rather than trust, and may only ADD
failures on top of machine gates. Blocking objections count only after two
independent skeptics confirm them.

### Amendment policy (the risk surface of autonomy)

Pre-registration disciplines only if amendments are deliberate. Rules:

1. Auto-amendment is permitted **only** for instrument/infrastructure
   failure classes: a serving config that cannot produce measurable output,
   capacity/overflow errors, a broken evidence channel, host/toolkit
   incompatibilities.
2. Every amendment is a new logged entry (A1, A2, …) in the milestone
   doc's Deviations section, committed **before** the data it governs,
   quoting the evidence that forced it. Frozen files are versioned
   (`matrix-v2.json`), never edited.
3. The audit's provenance critic screens each amendment with one question:
   *instrument fix, or result-shopping?* (Result-shopping = changing what
   counts as success after seeing outcomes.) A result-shopping
   classification is an attention state — stop, report.
4. Calibration set: the case study's m3 A1–A3 and m4 A1 (all legitimate),
   with the reasoning that made them so.

### Budgets

Defaults: $15 stop-loss per milestone, $150 per problem; GPU spend
computed from pod lifetime × list rate and reported in every milestone
doc. Breach → attention state. Never trim a frozen matrix to fit a budget
— stop and re-plan instead.

## Reporting standards

- Negative and unresolved results get the same prominence as positive
  ones; pre-register the null as live.
- Attribution is tuple-scoped: (engine, version, GPU, driver, model).
  "Bug vs numerics" claims require cross-condition evidence (the case
  study needed cross-GPU × cross-model runs to earn its attribution).
- Every run directory carries `run_meta.json` (versions, GPU, matrix
  hash), the manifest, raw per-turn JSONL, server logs, and the gate
  verdict.

## Instrument development

New instruments (benchmark + oracle + channels) are code in `harness/`,
land with selftests that plant defects in every failure channel and prove
each is caught (see `harness/selftest.py`), and are versioned like
everything else. Prefer exact-verifiable channels; a quality metric that
needs a human judge is a design smell in this domain.
