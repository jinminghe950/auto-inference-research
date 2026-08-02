# Adversarial milestone audit (frozen protocol)

Run after machine gates, before a milestone is approved; also as the
referee pass on papers. The session executes this with **parallel
subagents** (one per critic, then verifiers). The prompts, schemas, and
decision rules below are the frozen procedure — do not paraphrase them
per-session; drift in the audit is drift in the method.

Inputs: the milestone doc `research/<slug>/milestone-N.md`, its runs
directories, and (for the referee pass) the paper.

## Step 1 — Claim inventory (one agent)

> Read <milestone doc, and paper if auditing one> and produce an inventory
> of its EMPIRICAL claims (numbers, verdicts, comparisons — not motivation
> or plans) with the artifact paths cited for each (citations look like
> `[[art:path@hash8]]`; paths are relative to `research/<slug>/`). Return
> raw structured data only.

Schema: `{claims: [{id, text, artifacts: [path]}]}`, max 25 claims.

## Step 2 — Three critics (parallel agents, each trying to REFUTE)

Every critic returns `{objections: [{title, detail, evidence: [paths],
blocking: bool}], checked: [what was actually recomputed or re-run]}`,
max 8 objections. `blocking: true` only where the milestone conclusion
cannot stand while the objection holds.

**Evidence critic**
> You are the EVIDENCE critic auditing a research milestone. Claims under
> audit: <inventory>. Your job is to REFUTE: for each claim, open the
> cited artifacts, recompute the numbers from the raw JSONL/summaries with
> small python scripts — do not trust prose, including summary files if
> the raw per-turn data can check them. Verify cited sha256 prefixes match
> the files. Report objections; mark blocking=true only where a claim is
> contradicted or unsupported by its artifacts.

**Methods critic**
> You are the METHODS critic auditing the milestone described in <doc>.
> Claims: <inventory>. Your job is to REFUTE the inferences: confounds,
> seed counts vs claimed effect sizes, whether CIs support the stated
> verdicts, whether negative/unresolved results are reported at the same
> prominence as positive ones, whether the analysis matches the
> pre-registered plan in the same document. Read the frozen analysis code
> for silent divergences. Mark blocking=true only for flaws that
> invalidate a conclusion.

**Provenance critic**
> You are the PROVENANCE critic auditing the milestone described in <doc>
> with runs in <dirs>. Your job is to REFUTE the process claims: use git
> log to check that the pre-registration commit precedes the evidence
> commit, that amendments in the Deviations log were committed before the
> data they govern, and that each amendment is an instrument/infrastructure
> fix rather than result-shopping (changing what counts as success after
> seeing outcomes — see METHOD.md). Re-run the machine gate locally
> (`python3 -m kvdrift.main verify --out <runs dir>` from `harness/`) at
> the commit that landed the run if code has since moved — a manifest
> mismatch at HEAD with a clean verify at the landing commit is NOT an
> objection. Mark blocking=true only for ordering violations, gate
> discrepancies, or result-shopping amendments.

## Step 3 — Verify blocking objections (two skeptics each, parallel)

For each blocking objection (cap 6; log any overflow):

> An audit critic (<lens>) raised this BLOCKING objection against a
> research milestone: "<title>" — <detail>. Evidence cited: <paths>. Try
> hard to REFUTE the objection by re-examining the artifacts yourself.
> Uphold it only if you independently confirm the problem. Default to
> upheld=false if uncertain.

Schema: `{upheld: bool, reasoning}`. An objection is **confirmed only if
both skeptics uphold it**.

## Verdict

```
pass                = (confirmed blocking objections == 0)
confirmed_blocking  = [...]   -> revision loop with objections attached
unconfirmed_blocking= [...]   -> recorded, non-gating
advisory            = [...]   -> recorded, non-gating
checked             = per-critic list of what was recomputed
```

Record the full verdict in the milestone doc (an `## Audit` section) with
the commit hash audited. The panel may only ADD failures — it cannot pass
a milestone whose machine gates failed, and it cannot be re-rolled until
"pass" without logging each round.
