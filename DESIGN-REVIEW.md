# Design review — auto-inference-research (2026-08-03)

Scope: the whole lab as a system for its stated goal — *a Claude Code
session autonomously finds LLM-inference research problems, executes them
on rented GPUs, and produces gate-verified artifacts plus a
publication-ready paper*. Reviewed: METHOD.md, AUDIT.md, CLAUDE.md, the
gated-experiment skill, all of `harness/` (~2,100 LOC), `analysis/`,
`problems/`, and the repo's operational surface (git, CI, Runpod path).

Method of this review: six parallel specialist reviewers (method
integrity, gate/harness code, infra/security, statistics, autonomy
pipeline, ops/budget), 47 raw findings, deduplicated; the nine highest-
severity findings were each re-examined by an independent adversarial
verifier instructed to refute them against the actual files. All nine were
independently confirmed; none were refuted. Lower-severity findings were
spot-checked by hand where cited code had been read directly.

**Verdict.** The instrument layer is genuinely good: the gates fail
closed, the manifest verifier catches tampering and unmanifested files,
the selftest plants defects per scoring channel, and the control-plane
design (ephemeral pod, per-pod token, no credentials on the pod) is the
right shape. But the lab's central promise — that the method makes
fabrication and result-shopping "impossible rather than unlikely"
(METHOD.md:15-16) — is not delivered: **every link in the integrity chain
is ultimately attested by the same agent it is supposed to constrain.**
The gates verify internal consistency of a run, not conformance of the
run to the committed pre-registration; the ordering proof is mutable local
git history; the "adversarial audit" is spawned, prompted, aggregated, and
recorded by the session under audit. Separately, the statistics layer has
real defects (a pairing bug, no decision rules, an unverifiable power
claim), and the "automatically find problems" half of the goal has no
mechanism at all yet.

Severity legend: **CRITICAL** defeats the system's purpose or integrity
guarantees; **MAJOR** likely bites in real use; **minor** worth fixing.

---

## 1. The integrity chain is self-attested (CRITICAL)

### 1.1 Commit-before-data ordering is unenforceable — CRITICAL (verified)

METHOD.md:26-29 makes local git commit ordering the ordering proof, and
the only checker is the provenance critic running `git log` (AUDIT.md:
50-53) — a subagent of the audited session, reading history that session
constructed. `push` appears exactly once in the entire protocol, at
evidence-landing time (SKILL.md step 9), so during the whole run the
pre-registration exists only in mutable local history. The session also
holds direct Runpod tools (`create-pod`, `run-endpoint`, …) outside the
gated-experiment skill, so nothing prevents run-first-preregister-after:
explore on a GPU, observe the result, write a "pre-registration" that
predicts it, commit, then commit the evidence. The parent chain shows
prereg-before-evidence and every gate and critic passes. This does not
even require deliberate deception — METHOD.md:17 explicitly licenses
outcome-guided exploration, and nothing requires exploration runs to be
logged or disclosed (see 1.6). `GIT_AUTHOR_DATE`/`GIT_COMMITTER_DATE`
defeat the timestamp cross-check that AUDIT.md doesn't ask for anyway.

**Fix (cheap, closes most of the hole):** require the pre-registration
commit to be **pushed to the remote before any pod is provisioned**, and
record the pushed SHA + remote URL in `run_meta.json` on the pod (upload
it with the matrix). GitHub's server-side push timestamp is the external
anchor local history can't fake. Add an audit step reconciling Runpod
billing/pod history for the period against the registered runs.

### 1.2 Nothing binds the executed matrix to the pre-registration — CRITICAL (verified, two independent reviewers)

`cmd_verify` loads its ground truth from `run_meta.json` **inside the run
directory** — the run's own output (`harness/kvdrift/main.py:273`), and
the `verify` CLI accepts only `--out` (main.py:442-444), so the committed
matrix cannot even be supplied to the gate. G1 "every pre-registered
trajectory present" therefore means "every trajectory the run says it
intended"; G4's "matrix-frozen thresholds" come from the same
self-referential file. The session picks the `MATRIX` path freely at
launch (sweep.sh:14,35; SKILL.md step 4), and `matrix_sha256` in
run_meta (main.py:137-139) is self-recorded with no comparator: a grep
shows the sha256 pin exists only in prose (METHOD.md:26, CLAUDE.md:28).
Run a different matrix than the doc pinned — fewer trajectories, softer
`gate_thresholds` — and every machine gate passes; the provenance critic
re-runs the identical self-referential verify (AUDIT.md:56-58) and is
equally blind. The manifest is also rebuildable: tamper with results,
re-run `write_manifest`, and G3 is clean — G3 is tamper-*evidence* only
against an adversary who can't run the manifest writer, which is nobody
in this threat model.

**Fix:** `verify --matrix <path>` (required argument): recompute the
sha256 of the given matrix file, compare to `run_meta.matrix_sha256`
*and* to a `matrix` copy embedded in the manifest; the audit invokes it
with the **committed** matrix path at the pre-registration commit. That
one flag turns G1/G4 from self-consistency checks into
conformance-to-preregistration checks.

### 1.3 The adversarial audit is run by the agent under audit — CRITICAL (verified)

AUDIT.md:4-5: the session executes the audit "with parallel subagents".
The audited session assembles the claim inventory input, fills every
`<inventory>/<doc>/<dirs>` slot in the critic prompts, aggregates the
verdicts, authors the `## Audit` section that is the sole record
(AUDIT.md:85-88), and decides when to re-roll ("cannot be re-rolled until
'pass' **without logging each round**" — the log is written by the same
session). Critic `checked` lists are self-reported. Subagents share the
parent model, its context-shaping, and its incentives; they are
independent samples, not independent auditors. The frozen prompts limit
drift but nothing detects a session that paraphrases them anyway (see
also §4.9 — the audit is prose, not a skill, so per-session drift is
structurally invited despite AUDIT.md:6-7 warning against exactly that).

**Fix:** make the audit a pinned skill (frozen prompt bytes, hashed), have
it emit a machine-readable verdict file that is committed *by the audit
step itself*, and — for milestones that gate a paper — run the referee
pass in a **fresh session** (or second repo checkout) whose only input is
the pushed commit, not the executing session's context. True independence
from a same-account session is impossible; a fresh-context session with
pinned prompts is still a large improvement over same-context subagents.

### 1.4 The blocking-objection rule launders true objections — MAJOR (verified)

Step 3 (AUDIT.md:63-73) requires **both** skeptics to uphold, each is
instructed to "try hard to REFUTE" and to "default to upheld=false if
uncertain". So a true objection needs two independent confirmations
against a refutation-biased prior, while the milestone's own claims need
only survive critics the same session spawned. Unconfirmed blocking
objections are recorded but non-gating (AUDIT.md:78-80). The asymmetry is
backwards for a lab whose stated failure mode is fluent overclaiming:
uncertainty about an objection should *block* (fail closed), not pass.
The cap of 6 verified objections with overflow "logged" (by the audited
session) compounds this.

**Fix:** flip the default for evidence-class objections (uncertain =
upheld), or require 2-of-3 with one skeptic given the *uphold* framing;
remove the cap for objections marked blocking by the evidence critic.

### 1.5 G-cite verifies hashes, not numbers — MAJOR

`analysis/check_citations.py` checks that cited files exist, hash-match
on an 8-hex prefix, and number ≥ 10 distinct. It never checks that any
*stated number* matches artifact contents, nor that all empirical claims
carry citations — both are delegated to the LLM evidence critic, i.e. to
§1.3. METHOD.md:15 ("claims are mechanically bound to hashed artifacts")
oversells this: the binding is "this file existed", not "this file
supports this sentence". Also minor but real: the regex (`:23`) is strict
lowercase-8-hex — a malformed citation (`@ABCD1234`, 7 chars, `[[art:x]]`
with no hash) silently doesn't match and is *not checked at all*, while
the gate prints "all hashes match". An 8-hex (32-bit) prefix is also
grindable in ~2³² attempts if anyone ever cares to fake one, and is never
cross-checked against the full hashes in `manifest.json`.

**Fix:** flag any `[[art:` occurrence that doesn't fully match the strict
pattern; resolve cited files against the run manifest (full sha256), not
just the prefix; add a structured claims file (claim → artifact → JSON
pointer → expected value) that a script can check exactly for at least
the headline numbers.

### 1.6 Exploration/confirmation boundary and selection effects are unpoliced — MAJOR

METHOD.md:17 permits outcome-guided exploration but nothing requires
exploration runs to be logged, disclosed in the milestone doc, or
reconciled against GPU billing — combined with 1.1, unregistered pilot
runs are invisible. At the problem level, nothing prevents parking
problems whose confirmed results look boring and publishing only winners
(`problems/README.md` statuses are hand-edited fields with no checker) —
classic file-drawer, one level up. On the pod, `control_plane.py:108`
overwrites `/workspace/run.log` on every `/run`, so executing a sweep
five times and collecting the last one leaves no trace in the collected
"operational evidence".

**Fix:** an append-only run ledger on the pod (control plane appends
`{ts, cmd}` to `/workspace/run_ledger.jsonl`, fetched with every
collection); a standing "exploration log" section in milestone docs;
billing reconciliation in the audit (one Runpod API call); parked
problems require a committed one-paragraph parking note.

---

## 2. Statistics (CRITICAL in aggregate)

### 2.1 H1/H2/H3 are computations, not tests — CRITICAL (verified)

`analysis/analyze.py` emits ~250 uncorrected nominal-95% intervals for a
6-config matrix (8 channels × (4 bins + aggregate) × configs, plus slope
and contrast CIs) and **no pre-registered decision rule anywhere** maps
them to verdicts: no primary endpoint, no α, no multiplicity control, no
stated success criterion for H1/H2/H3. "Frozen analysis" without a frozen
*decision rule* leaves verdict-assignment to post-hoc narrative — the
exact flexibility pre-registration exists to remove. With ~250 intervals,
a dozen spurious "significant" cells are expected by chance, free for a
motivated writer to select (the audit can't object: the analysis matched
the frozen plan).

**Fix:** each milestone pre-registers (a) one primary contrast + channel
+ bin, (b) its decision rule ("CI excludes 0" or better, a paired
permutation test), (c) everything else labeled exploratory in the
analyzer's own output, mechanically.

### 2.2 H2 pairing is positional, not by seed — MAJOR (hand-verified)

`load()` globs `sorted(...glob("trajectory_*.jsonl"))` and the H2
contrast pairs by index with `n = min(len(a), len(b))`
(analyze.py:58, 139-149). One missing/aborted trajectory in one arm
misaligns **every subsequent pair** (and lexicographic sort breaks pairing
outright if seeds ever cross a digit boundary, e.g. 999→1000). The
"paired over seeds" claim in the docstring is false as implemented.
**Fix:** key trajectories by parsed seed; pair on the intersection;
report dropped seeds.

### 2.3 The problem-003 power claim is unverifiable and likely optimistic — MAJOR

"~30 seeds for a 0.1 effect at 80% power" cites measured variance that
lives in the absent `agentic-research` repo; nothing in this repo can
recompute it, and the reviewers' back-of-envelope from binomial variance
at the observed rates suggests it is optimistic. A pre-registration would
freeze this number as the design's justification without any auditor
being able to check it. **Fix:** vendor the variance estimates (or the
m3/m4 summary JSONLs) into this repo and commit the power calculation as
a script.

### 2.4 Bootstrap details bias exactly the cells verdicts read — MAJOR

Percentile bootstrap over n=10 clusters is already undercovered; in
sparse cells (deepest bin, recall channel) resamples with an empty
denominator return `None` and are silently dropped (analyze.py:77-87),
conditioning the CI on "the resample had data" and narrowing it precisely
where data are thinnest. `ols_slope` likewise silently drops empty bins
(:90-98), so a "slope over turn bins" can quietly become a slope over two
points. **Fix:** report the None-drop count per CI; require a minimum
denominator per cell (ties into G4, §3.6); prefer BCa or a permutation
test for the primary contrast at n≤10 clusters.

### 2.5 "Seeds" are scenarios, not sampling seeds — minor (but must be stated)

Decoding is greedy with a constant API seed (client.py:19-20, server
`--seed 0`, serve.py:42); `sim_seed` varies only warehouse contents and
instruction schedule. CIs "over seeds" are therefore over *task
heterogeneity* of one simulator's scenario distribution. That is a
defensible design — but no document says it, and "10 seeds" invites the
sampling-noise misreading, including in the power claim. Self-conditioned
trajectories additionally make turn-depth effects (H1) a mix of context
length and accumulated transcript contamination; worth one honest
paragraph in every paper.

### 2.6 G2's latency fallback is an unjustified constant on the wrong quantity — MAJOR

The 0.6 deep-turn median ratio (main.py:242-243) is compared on
**total request latency**, not TTFT as the comment reasons, and is the
*only* treatment evidence for engines exposing neither usage counters nor
prom metrics — which includes the LMDeploy int8 configs that problem 003
depends on. Long generations dilute the prefill gap toward 1.0
(false G2 failure of a working cache); shared-host jitter through the
proxy adds noise in both directions. Worse, the recompute arm is
**fail-open**: it only fails if positive cache evidence appears
(main.py:324-328), so on an engine that exposes no counters, a
contaminated recompute leg (cache silently on) passes G2 — the
treatment's zero-control is unverified exactly where it matters.
**Fix:** measure and record TTFT explicitly (streaming first-token
timestamp); require *positive* evidence of cache-off for recompute legs
on such engines (e.g. the same paired-latency test inverted), or declare
the configuration unmeasurable and design around it.

---

## 3. Gate & harness code defects

### 3.1 `run_meta.json`/manifest written only after ALL configs — MAJOR

`cmd_run` writes run_meta and the manifest after the config loop
(main.py:205-214); `server.start()` raising (host lottery, OOM, wedged
CUDA) propagates and — under `set -euo pipefail` — kills the run with
**zero manifest**, leaving all completed configs' evidence unmanifested
and unverifiable ("no hash, no verdict" turns into "no verdict for good
data"). With per-config cost up to 40 min of billed startup timeout
(serve.py:70), this is paid-for evidence lost to a crash in config N.
**Fix:** write run_meta + incremental manifest after every config;
wrap the per-config block in try/except that records the failure and
continues (a failed config already fails G1, correctly).

### 3.2 Onset runs have no gate — MAJOR

`cmd_verify` KeyErrors on onset run dirs (`matrix["trajectories"]`,
main.py:274) — so milestone-4-style evidence has no machine gate at all,
not even G3 re-verification, despite problem 002's design being
probe/onset-first. **Fix:** teach `verify` the onset layout (G1 =
lengths×reps×modes complete; G3 same manifest).

### 3.3 `serve.py` can bless a stale server — MAJOR

`start()` polls `/health` on the shared fixed port; it never verifies the
responding process is *its* child (serve.py:104-117), and `stop()` skips
the group-kill when the leader already exited (`:123` guard), which can
leave worker processes serving. A wedged previous config's server can
absorb the next config's entire traffic — wrong KV dtype under the right
label — and pass G0/G1. G2 might catch a caching mismatch; nothing
catches an fp16-vs-int8 swap. **Fix:** after health, assert identity
(`/v1/models` + a per-config nonce in `--served-model-name`, or verify
listening PID ∈ child pgid via `/proc`); make `stop()` always
`killpg` when the pgid still has members; verify port is free before
start.

### 3.4 `selftest.py` never exercises the gates — MAJOR

Selftest covers scoring channels, prom parsing, manifest round-trip —
but never calls `cmd_verify`: G0/G1/G2/G4 logic and the G2 latency
arithmetic have zero planted-defect coverage, contradicting
harness/README.md:5-7's own standard. Every gate bug in this section
would have been caught by applying the repo's stated discipline to its
own gate code. **Fix:** build a synthetic run dir in selftest (fake
run_meta, trajectories, metrics) and plant one defect per gate channel;
same for `analyze.py` (planted misalignment must move H2).

### 3.5 G4 ignores unknown threshold keys but prints PASS — minor (nasty)

Only `fp16_action_correct_min` is dispatched (main.py:335-343); any other
key — or a typo like `fp16_action_corect_min` — in a **frozen**
pre-registration is silently ignored and "G4 … PASS" still prints.
A frozen file typo disables the sanity floor with no signal.
**Fix:** unknown `gate_thresholds` keys are a gate failure.

### 3.6 Missing sanity floors — MAJOR

No minimum-denominator floor anywhere: a config where only 3 turns were
scoreable can still produce headline rates (and §2.4's narrow CIs). No
"minimum recall probes", no "minimum tool turns". **Fix:** G4 floors for
denominators, frozen in the matrix.

### 3.7 `analysis.json` lands inside the manifested tree — minor (workflow bug)

analyze.py defaults to `<results_dir>/analysis.json` (:153); manifest
completeness then fails any later `verify` with "result file not in
manifest" (manifest.py:61-68) — and SKILL.md orders verify (step 7)
before analysis (step 9), so every collected run directory becomes
un-reverifiable the moment the frozen analyzer runs, which is exactly
when the audit re-runs verify. The tempting "fix" — regenerating the
manifest — would destroy its evidentiary value. **Fix:** write analysis
outputs next to, not inside, the manifested dir (or add a manifest
`derived/` allowance).

### 3.8 Recall oracle admits false positives — minor

Bare word-boundary regex over the whole reply (scoring.py:109-110): a
coincidental number or a hedged multi-guess answer ("either 34 or 43")
scores as recall. Fine at current effect sizes; worth tightening
(answer-span extraction) before recall becomes a headline channel.

---

## 4. The autonomy gap: "automatically find problems … paper ready to publish"

### 4.1 There is no automatic problem discovery — MAJOR (verified)

The pipeline's sole entry point is "the user names a problem"
(CLAUDE.md:19-20; problems/README.md:3-4), and sourcing is explicitly
manual ("Until watchers are automated, refill the queue by scanning…",
README.md:28-31). Nothing schedules sessions, no watcher skill, no
routine, no scoring script (the scoring formula is one prose sentence).
Half of the stated goal currently does not exist. **Fix:** a `problem-scout`
skill (scan vLLM/SGLang/LMDeploy/TensorRT-LLM releases + issues + arXiv,
emit candidate files with scores) run on a schedule (cron/Routine) that
opens a PR against `problems/` — keeping the *human* as the selector of
what runs, which you want anyway for §4.4.

### 4.2 The collision scan — the only novelty defense — is unfrozen prose — MAJOR

It's a CLAUDE.md paragraph, self-graded by the motivated session, with
no minimum-coverage requirement, no query log, no gate. A shallow scan
that misses prior art wastes the entire problem budget (or worse,
publishes a known result). At paper time — months later — **nothing
re-checks novelty at all**, and the audit's claim inventory explicitly
excludes motivation/related work (AUDIT.md:14-16), so the bibliography
is entirely ungated: hallucinated citations would sail through G-cite
and the referee pass as specified. **Fix:** collision-scan skill with a
committed query/coverage log; a paper-time re-scan step; a bibliography
gate (resolve every DOI/arXiv ID, verify title match — this is
mechanically checkable).

### 4.3 No figures, ungated LaTeX, no packaging — MAJOR

There is no figure-generation path at all (papers about latency/error
curves without plots), the LaTeX mirror is generated from the md but
never G-cite-checked (numbers can drift in translation), and no
arXiv-packaging step exists. **Fix:** frozen plotting script per
instrument (reads analysis.json, writes hashed PNGs/PDF that are
themselves citable artifacts); run check_citations on a text-extraction
of the .tex too.

### 4.4 No human boundary before public release — MAJOR

The repo defines attention states for budget and amendments, but not for
*public actions*: nothing says arXiv submission or filing the drafted
upstream vLLM/LMDeploy issues (`reports/`) requires explicit human
sign-off. For a lab whose sessions run autonomously, "ready to publish"
must be a hard hand-off, in writing. **Fix:** one paragraph in METHOD.md:
publication, issue-filing, and any outward communication are
human-approval-gated, always.

### 4.5 No crash/context-exhaustion recovery — MAJOR

One session executes a problem end-to-end across multi-hour GPU runs;
the protocol has no checkpoint/resume design: what a successor session
must read to resume a half-done milestone is undefined (milestone doc +
runs dir is probably sufficient — but nothing says so), and a session
that dies mid-run leaves a pod billing (§5.2) and a run in limbo.
**Fix:** a RESUME.md convention (or a `state` field in the milestone doc)
plus a session-start step: sweep for running pods and half-landed runs
before doing anything else.

### 4.6 The calibration set lives in an unreachable repo — MAJOR

The amendment screen — the method's self-identified "risk surface of
autonomy" — calibrates against deviation logs in
`jinminghe950/agentic-research`, which is not vendored and (in this
session) not attached. Fresh sessions inherit the rule but not the
calibration data; the provenance critic screens amendments against
examples it cannot read. Problem 001/002/003 also cite its artifacts as
their empirical basis. **Fix:** vendor `research/kv-drift/`'s milestone
docs (or at least the m3/m4 Deviations sections and summary JSONLs) into
this repo.

### 4.7 Zero tooling enforcement of the method's invariants — MAJOR

No CI (no `.github/`), so `selftest.py` before landing is honor-system;
no hook prevents editing frozen `matrix-v*.json` files; no hook runs the
secrets sweep before evidence commits; queue statuses are hand-edited.
Every invariant is prose. **Fix:** a GitHub Actions workflow (selftest +
check_citations on papers + a frozen-file-diff guard that fails any PR
modifying an existing `matrix-v*.json` or a committed pre-registration
section) plus a pre-commit secrets scan. CI is the cheapest *external*
enforcement this design can get, and it composes with fix 1.1
(push-before-run) nicely: the push triggers the checks.

### 4.8 Only the executor is a skill — minor

Collision scan, audit, and paper pass are re-improvised from prose each
session; AUDIT.md:6-7 warns that paraphrase-drift *is* method-drift while
being stored as exactly the kind of document that gets paraphrased.
**Fix:** skills for all four procedures, with the audit prompts loaded
from hashed files.

---

## 5. Ops, budget, and security

### 5.1 Budget stop-losses have zero machine enforcement — MAJOR (verified)

Spend is "pod lifetime × list rate", computed and reported by the session
(METHOD.md:82-85); SKILL.md's enforcement is the sentence "track the
budget". The lab's own axiom — "trust evidence, never the executor's
narrative" — is applied to every number except the money. The Runpod API
exposes billing queries; nothing uses them. **Fix:** a `harness/pod/budget.py`
that queries Runpod billing/pod uptime, writes a `spend.json` artifact
into the run dir (manifested), and hard-fails the milestone landing if
spend > stop-loss; audit recomputes it.

### 5.2 Session death orphans a billing pod indefinitely — CRITICAL (verified)

`control_plane.py` is `ThreadingHTTPServer(...).serve_forever()` — no
TTL, no idle timeout, no self-destruct; termination exists only as
SKILL.md step 8, which requires a live, well-behaved session. A crash,
context exhaustion, or plain abandonment leaves a GPU pod billing until a
human notices — the single most likely way this lab loses ten times a
milestone budget. **Fix (belt and braces):** (a) control plane
self-terminates the pod via the Runpod API… it has no credentials, so
instead: shut down the pod from *inside* (e.g. watchdog thread: if no
authenticated request for N minutes AND no run in progress → `runpodctl
stop`/poweroff, which on Runpod stops billing for GPU); (b) SKILL.md
step 1 gains "set a `send_later`/Routine check-in that lists pods and
terminates leftovers"; (c) session-start orphan sweep (§4.5).

### 5.3 Engine versions live outside the pre-registration — MAJOR

The sha256-pinned matrix does **not** contain engine versions; they are
env-overridable shell defaults that *disagree between entrypoints*
(sweep.sh: vLLM 0.26.0; repro.sh: 0.19.1), and `sweep.sh`/`repro.sh`/
`configs/` are outside the G3 code manifest (build_manifest walks
`kvdrift/` only — manifest.py:20-24). Attribution is defined as
tuple-scoped *(engine, version, GPU, driver, model)* (METHOD.md:91), yet
the version half of the tuple is unpinned by every gate. run_meta records
versions after the fact — but per §1.2 nothing compares that to a
commitment. **Fix:** engine versions go in the matrix JSON (the harness
asserts them against the venv at startup); manifest `code` scope extends
to `harness/**` minus results.

### 5.4 Evidence-in-git with no size policy — MAJOR

30 seeds × 40 turns × configs of per-turn JSONL (with full transcripts)
plus complete engine server logs, committed per milestone, no LFS, no
`.gitattributes`, no cap. vLLM server logs alone routinely run tens of
MB; a single >100 MB log hard-blocks the push that the fail-closed
landing *requires*. **Fix:** compress logs, commit summaries + manifests
always, raw JSONL under LFS or a size gate in the landing step.

### 5.5 The pod template is unpinned external state — MAJOR

Template `9yc1o57ktn` (SKILL.md step 1) lives in one Runpod account and
is referenced only by prose ID: container image, exposed port, disk
sizing are pinned nowhere machine-readable; mutating the template
silently changes every future executor. `make_start_cmd.py` regenerates
the start command but not the rest. **Fix:** a committed
`harness/pod/template.json` (image tag, ports, disk, env) and a skill
step that verifies the live template matches it (or creates it).

### 5.6 Control-plane / restart semantics — MAJOR

`/status` "running: false" cannot distinguish "run finished" from
"control plane restarted mid-run" (in-memory `proc` is lost;
control_plane.py:31), and container restart wipes `/tmp` results (the
run writes to local disk by design). The monitor's exit condition can
therefore fire on a dead-and-restarted pod and the session would proceed
to "collect" a partial run — G1 catches the gaps, but the diagnosis will
be misleading and the spend wasted. **Fix:** control plane writes a
`run_id` + pid file to `/workspace` on `/run` and reports
`{run_id, booted_at}` in `/status`, so a restart is detectable.

### 5.7 Security posture of the executor path (hand review; the fanned-out reviewer for this dimension was blocked, so this is from direct reading)

The shape is right: per-pod 48-hex token (2¹⁹² — entropy is fine), no
credentials on the pod, ephemeral executor, path traversal on `GET`
guarded by a sound realpath check (control_plane.py:55-57). Real items,
in descending order:

- **Gated-model credentials are unhandled — MAJOR, blocks problem 001.**
  Llama-3.1-8B (and Mistral weights) are license-gated on HF; downloading
  them on the pod requires an `HF_TOKEN`, which contradicts "no
  credentials on the pod" (CLAUDE.md) — and no mechanism exists either
  way. Problem 001 as pre-registered cannot run. Fix: pre-stage weights
  onto the network volume in a separate, credentialed, *non-experiment*
  step (documented in the skill), keeping experiment pods credential-free;
  or use a scoped fine-grained HF token accepted as an explicit exception.
- **Token comparison isn't constant-time** (control_plane.py:50):
  `!=` on the header vs `hmac.compare_digest`. Over the Runpod proxy the
  timing signal is likely unusable, but the fix is one line — minor.
- **`tarfile.extractall` falls back to no filter on Python < 3.12**
  (control_plane.py:94-96): only the token-holder can upload, so this is
  defense-in-depth only — minor; pin the template image's Python ≥ 3.12.
- **The pod token is deliberately printed into the session transcript**
  (SKILL.md step 1). Acceptable given pod ephemerality — but the secrets
  sweep greps the *artifacts*, not the transcript, and transcripts may
  land in session logs. Keep tokens per-pod and terminate promptly (both
  already required) and this stays minor.
- **Injection surface:** collected artifacts (model outputs, server logs)
  are later *read by audit subagents*. A degraded model's output — or a
  poisoned upstream model — could contain text addressed to the critics.
  The critics' recompute-with-python discipline mitigates; still, critic
  prompts should state that artifact *content* is data, never
  instructions. One sentence in AUDIT.md — minor today, cheap to close.

---

## 6. What to fix first

The theme of §1 is one sentence: **give the method an external anchor.**
Almost everything else is incremental hardening. In order of leverage:

1. **Push-before-run** (1.1): pre-registration must be on the remote
   before a pod exists; record the SHA in run_meta; audit reconciles
   against Runpod history. Cheapest possible external anchor.
2. **`verify --matrix`** (1.2): make the gate compare the run to the
   *committed* matrix. One flag; converts G1/G4 into real conformance
   gates.
3. **Pod watchdog + budget artifact** (5.1, 5.2): idle self-stop on the
   pod; manifested `spend.json` from the Runpod API; hard stop-loss at
   landing.
4. **Pre-registered decision rules** (2.1) and the **H2 pairing fix**
   (2.2) — before any new milestone runs.
5. **CI** (4.7): selftest + G-cite + frozen-file guard on every push;
   composes with (1).
6. **Gate selftests** (3.4) and the crash-robust manifest (3.1), the
   onset gate (3.2), server identity check (3.3).
7. **Fresh-session audit with pinned prompts** (1.3, 1.4) + flipped
   skeptic default for evidence objections.
8. **Vendor the case study** (4.6) — unblocks the amendment screen and
   the power claim (2.3).
9. **Problem-scout skill + schedule** (4.1) — the missing half of the
   goal; keep the human as selector and as the publication gate (4.4).
10. **Weights pre-staging for gated models** (5.7) — unblocks problem 001.

None of this discards the design. The gates-fail-closed instinct, the
content-addressed evidence, the fail-closed landing of negative results,
and the executor/scientist separation are the right bones; the work is
moving the trust anchors *outside* the agent being constrained.
