# Milestone 1 — H100 probe + sweep: is the fp8-KV tool-calling death hardware-conditional?

**Status: EXECUTED — machine gates verified locally; results in
§Results.** Everything above §Results was frozen at the pre-registration
commit (71bba72), before any data existed; the commit ordering is the
proof. Amendments only via the Deviations log.

## Questions under test

The case study (`jinminghe950/agentic-research`, `research/kv-drift/`)
established: vLLM 0.26.0 `--kv-cache-dtype fp8_e5m2` (calibration-free)
silently kills the tool-call channel for Qwen2.5-7B-Instruct — zero
parseable calls at every context length (960 probes) and every trajectory
depth (800 turns), fluent free text with character-level copy corruption
— identically on RTX 4090 (sm_89) and A100 (sm_80), where the flag runs
identical FlashInfer FA2-template dequant-on-load kernels. The collision
scan (`collision-scan.md`, committed 42f5677) confirmed the Hopper cell
unmeasured and found that sm_90 exercises genuinely different code.

1. **Q1 (persistence)** — does the e5m2 regime replicate on H100 SXM
   (sm_90), where the same flag uniquely selects FlashInfer's native FA3
   fp8 prefill (queries themselves quantized to e5m2) and TRT-LLM's XQA
   decode kernel (documented KV support: e4m3 only, gated in by vLLM for
   any fp8)?
2. **Q2 (marketed-path audit)** — calibration-free fp8_e4m3 on Hopper is
   the configuration vLLM's 2026-04-22 blog calls "ready to be the
   default starting point," evaluated there only on reasoning/long-context
   benchmarks at ≥8B. What do the agentic channels (tool-call presence,
   parse validity, tool selection, argument fidelity, identifier copy)
   show at 7B?
3. **Q3 (attribution)** — whatever the outcomes, bind them to the
   logged attention backend/kernel selection per leg (server logs are
   hash-manifested), completing the case study's (engine, GPU, model)
   tuple with the one architecture whose fp8 path is materially distinct.

## Registered predictions

- **P-persist** (favored by the case study's m4 attribution: the failure
  is model-conditional numerics — Qwen2.5-7B KV outlier structure vs
  e5m2's 2-bit mantissa — not a kernel defect): the dead regime
  replicates on sm_90. Zero parseable calls at all onset lengths, sweep
  no_call ≈ 1.0, fluent-but-corrupted free text.
- **P-hw** (alternative, live because sm_90 runs different kernels in
  both attention phases): a materially different regime on H100 — clean,
  intermediate, or a new failure shape (including a loud engine error;
  cf. vllm#31843 and the undocumented e5m2-through-XQA path).
- **P-e4m3** (genuinely open): the repo's own m2 found e4m3 = token salad
  on Ada at vLLM 0.19.1; the vLLM blog found e4m3 near-lossless on
  Hopper at ≥8B on non-agentic metrics. Both a clean and a degraded
  outcome at 7B/sm_90/0.26.0 are live; either is reportable.
- The null is live in every direction: persistence, escape, and loud
  failure are all deployment-relevant, unclaimed results. No outcome of
  this milestone is a failed milestone.

## Design (frozen)

- **Host**: one NVIDIA H100 SXM 80GB (Runpod id "NVIDIA H100 80GB HBM3"),
  SECURE cloud, $2.99/hr; host driver must support CUDA ≥ 13.0 (vLLM
  0.26.0 wheels) — validated via nvidia-smi before proceeding, pod
  re-rolled on a non-conforming host. Template `kv-drift-harness`
  (id `9yc1o57ktn`), env HF_HOME=/workspace/hf, KV_DRIFT_ROOT=/workspace,
  KV_DRIFT_PORT=8000, KV_DRIFT_TOKEN=<per-pod random>. Engine pins:
  vLLM==0.26.0, LMDeploy==0.15.0 (sweep.sh defaults; lmdeploy venv is
  built but unused — no lmdeploy configs in this milestone).
- **Model**: Qwen/Qwen2.5-7B-Instruct (not gated; downloaded on-pod).
- **Six run groups**, each an independent harness run with its own
  manifest and gate verdict, split per precision so an engine-level crash
  in one precision cannot destroy the others' manifested evidence
  (registered outcome class, see Gates). Controls run first.

  Sweep groups — 2 configs × 5 seeds (3000–3004) × 40 turns each, exact
  case-study knobs (max_tokens 192, max_model_len 24576, preflight on):

  | matrix (harness/configs/) | sha256 | configs |
  |---|---|---|
  | `h100-sweep-fp16.json` | `eb73823a44ce19ec6b9b7bc274f5bac2d8efbd4761c690bd366b58a9b0f85472` | vllm-fp16-{reuse,recompute}; G4 floor fp16_action_correct_min 0.85 |
  | `h100-sweep-e4m3.json` | `63ef5081350337d578c930b3526cf6d845a4d2a50f7c8ff316589d2f1a065a80` | vllm-fp8e4m3-{reuse,recompute} |
  | `h100-sweep-e5m2.json` | `0dcf6026fbfeaaeb6f7c48fa7bafa42874cbb1b3a1ac96fcdb7a78c76a38e02c` | vllm-fp8e5m2-{reuse,recompute} |

  Onset groups — 1 config × {near,far} × lengths {0,128,256,512,1024,
  2048,4096,8192} × 20 reps = 320 rows each, seed_base 5000,
  max_model_len 16384:

  | matrix (harness/configs/) | sha256 | config |
  |---|---|---|
  | `h100-onset-fp16.json` | `d84bfc696a85deb4449622004d7469482f35c764b3bbed96781b596c835785a9` | vllm-fp16 |
  | `h100-onset-e4m3.json` | `822969cdcc0a003731789d1348939d29b8bcf8d0de4f5ee09dfbcc8bf552a9bf` | vllm-fp8e4m3 |
  | `h100-onset-e5m2.json` | `cbf13cf1d9f000a8092dc0d9bb4d452b8d1ee8877558e296322debd9736011d2` | vllm-fp8e5m2 |

- **Execution** (gated-experiment skill; single /run command, frozen):

  ```
  cd harness && \
  for g in fp16 e4m3 e5m2; do \
    VENVS=/venvs MATRIX=configs/h100-sweep-$g.json OUT_DIR=/tmp/m1-sweep-$g ./sweep.sh; \
    echo SWEEP_${g}_EXIT=$?; \
  done; \
  for g in fp16 e4m3 e5m2; do \
    VENVS=/venvs VLLM_BIN=/venvs/vllm/bin/vllm LMDEPLOY_BIN=/venvs/lmdeploy/bin/lmdeploy \
    python3 -m kvdrift.main onset --matrix configs/h100-onset-$g.json --out /tmp/m1-onset-$g; \
    echo ONSET_${g}_EXIT=$?; \
  done; \
  for d in /tmp/m1-sweep-* /tmp/m1-onset-*; do \
    cp -r $d /workspace/harness/results/$(basename $d) && echo COPY_OK_$(basename $d); \
  done
  ```

  Venvs on pod-local disk, results written to /tmp then copied (the case
  study's MooseFS Errno-5 lesson). Groups run controls-first; a nonzero
  exit in one group does not stop later groups.
- **Collection**: every group fetched into
  `research/fp8-kv-hopper/runs/<group>/` via the control plane, plus
  `/run.log` as operational evidence. The pod is terminated immediately
  after local re-verification.
- **Kernel-path evidence (Q3)**: the per-config server logs
  (`logs/<config>.server.log`, hash-manifested with each group) are the
  attribution record for attention-backend/kernel selection. The
  milestone report quotes backend-selection lines only from these
  manifested logs.

## Machine gates (frozen)

- **Sweep groups**: G0–G4 as implemented in `harness/kvdrift/main.py`
  (`verify`), run on-pod by sweep.sh AND re-run locally on the collected
  copies: `python3 -m kvdrift.main verify --out
  ../research/fp8-kv-hopper/runs/m1-sweep-<g>` from `harness/`. The G4
  floor (fp16 action_correct ≥ 0.85) binds the fp16 group.
- **Onset groups**: row count 320/320 (enforced on-pod by `kvdrift
  onset`'s exit code) + manifest re-hash, re-run locally:
  `python3 research/fp8-kv-hopper/analysis/verify_onset.py
  research/fp8-kv-hopper/runs/m1-onset-<g>` (sha256
  `25c2584e13797866f62703dee0a006217ef3a502cc83ff00b12a8a61d018500d`).
- **Registered readings of anticipated gate failures** (fixed now so no
  post-hoc interpretation is needed; failed gates land fail-closed with
  evidence preserved either way):
  - A quantized sweep group failing G0 coherence (probe text_ok false →
    trajectories skipped → G0+G1 FAIL for that group) is the **SALAD**
    outcome for that precision — a regime finding, evidenced by
    probe.json + server log. It does not impugn the other groups.
  - A group aborted by an engine-level serving failure (server never
    healthy / crash mid-run; run.log + server log as evidence) is the
    **ERROR** outcome for that precision — a loud deployment failure on
    sm_90, reported as such. Other groups' verdicts stand.
  - The fp16 groups have no such registered escape: any gate failure
    there is instrument/infra failure — fix via the amendment policy or
    stop.

## Analysis plan (frozen)

1. Per sweep group: the repo-frozen `analysis/analyze.py` (unchanged,
   case-study provenance) for channel rates, turn bins, bootstrap CIs,
   H2 reuse−recompute contrast.
2. `research/fp8-kv-hopper/analysis/analyze_m1.py` (sha256
   `5e64c186aa91aef9ebdada46d60ff85dcf9c56da35980ad517fc93d22b3ac4fb`)
   over the runs root: recomputes everything from raw JSONL (summaries
   never trusted), applies the registered instrument-sanity rule (fp16
   onset ≥ 0.90 sku_exact and ≥ 0.95 call_present in every cell, fp16
   sweep complete) and the registered e5m2 regime classification
   {ERROR, SALAD, PERSISTS, CLEAN, MIXED} with thresholds exactly as
   documented in its docstring. e4m3 receives the same statistics and a
   reporting-only label; it never gates and never enters the e5m2 call.
   **Validation on known data (pre-registration check)**: run against the
   case study's A100 artifacts (m4-replication + m4-onset mapped into the
   group layout), the analyzer returns instrument_sanity PASS, regime
   PERSISTS, fp16 action_error 0.003, e5m2 action_error 1.000 — matching
   the published m4 numbers exactly.
3. Cross-generation comparison table (4090/A100 numbers cited from the
   case study's published artifacts; H100 numbers only from this
   milestone's manifested runs).

## Milestone-2 condition (registered)

A powered follow-up milestone runs **iff** the registered classification
returns **MIXED** for e5m2 or e4m3 (a measurable intermediate regime
worth characterizing); it will be pre-registered separately before any
m2 data. PERSISTS / CLEAN / SALAD / ERROR outcomes proceed directly to
the write-up milestone.

## Budget

H100 SXM SECURE $2.99/hr. Estimate: venv build + model download ~20 min;
6 sweep-config server cycles + 30 trajectory-seeds ~75 min; 3 onset
groups ~40 min; total ≈ 2.2–2.8 h ≈ **$7–9**. **Stop-loss $15** (repo
default): if projected pod time exceeds it, terminate and report
attention. Program spend to date: $0.

## Deviations log

- **A1 (2026-08-03, collection phase, after all data production —
  infra).** The frozen /run command's final copy step assumed
  `/workspace/harness/results/` existed; on this pod's fresh volume it
  did not, so every `cp` failed (`cannot create directory`) while all six
  group dirs remained intact in pod-local `/tmp`. Recovery: a second
  control-plane /run with `mkdir -p /workspace/harness/results` plus the
  identical copy loop (all six `COPY_OK` confirmed). No frozen file was
  touched and no data was re-produced — the deviation is confined to
  moving already-written artifacts into fetchable space. Consequence:
  the control plane tees every /run to `/workspace/run.log`, so the
  recovery run overwrote the original run.log before its self-copy line
  executed. The original operational log survives as (a) the
  monitor-captured marker stream, landed as `runs/m1-monitor-events.log`
  (sha256 4c71b225…), and (b) the e5m2 traceback excerpt fetched from
  the live run.log before the overwrite, quoted in §Results. All
  scientific artifacts (trajectories, probes, onset rows, server logs,
  manifests, run_meta) never lived in run.log and are unaffected.
- **A2 (2026-08-03, audit revision round 1 — provenance fix).** The
  frozen analysis plan's step 1 (repo-frozen `analyze.py`) writes its
  derived `analysis.json` into the results directory by default; running
  it after local gate verification left an unmanifested derived file
  inside each sweep group's content-addressed directory, so the frozen
  `verify` command failed G3 on the committed tree (audit objection B1,
  confirmed by two skeptics). Fix: the two derived files were **moved**
  (bytes unchanged; both critics reproduced them byte-identically from
  the frozen analyzer) to `runs/analysis/m1-sweep-{fp16,e4m3}.analysis.json`,
  outside the manifested dirs — matching the step-2 analyzer's correct
  root-level placement. After the move the frozen command reproduces
  G0–G4 PASS (fp16) and G0–G3 PASS (e4m3) from the committed tree. No
  raw evidence file was touched; no frozen file changed; the plan named
  the tool, not the output path — placement was an execution error.

## Results (frozen analyzers; artifacts in `runs/`)

Executor: Runpod pod `6263ewex81sh10`, NVIDIA H100 80GB HBM3 (sm_90),
driver 580.126.09 / CUDA 13.0, datacenter AP-IN-1, SECURE, 00:22–00:57
UTC 2026-08-03. vLLM 0.26.0 (pip wheel), torch 2.11.0, Python 3.12,
Qwen/Qwen2.5-7B-Instruct. Machine gates were evaluated on-pod and
re-verified locally on the collected copies; the local verdict is the
one reported.

| group | gate verdict (local) | outcome |
|---|---|---|
| m1-sweep-fp16 | G0–G4 **PASS** | control clean: action_error 0.006 [0.000,0.017] |
| m1-sweep-e4m3 | G0–G3 **PASS** | **silent total tool-call death**: action_error 1.000 [1.000,1.000] |
| m1-sweep-e5m2 | no data — engine never started | **ERROR** (registered class): startup crash |
| m1-onset-fp16 | rows 320/320 + manifest **PASS** | sku_exact 1.00, call_present 1.00 at every cell |
| m1-onset-e4m3 | rows 320/320 + manifest **PASS** | call_present 0.00 at **every** cell (16/16) |
| m1-onset-e5m2 | no data — engine never started | **ERROR** (registered class): startup crash |

Registered classification (frozen `analysis/analyze_m1.py`, output
`runs/analysis-m1.json` sha256 83bec634…): instrument_sanity **PASS**;
**e5m2 regime on sm_90 = ERROR**; **e4m3 label = PERSISTS**. The
milestone-2 condition (MIXED) is **not triggered**.

### e5m2: vLLM 0.26.0 cannot serve `fp8_e5m2` on H100 under its default backend selection

Both attempted e5m2 engine starts (the sweep group's first config,
vllm-fp8e5m2-reuse, and the single onset config; the registered
recompute config was never reached because the sweep aborted at its
first config) died identically during FlashInfer autotune warmup,
before serving a single request. The e5m2 groups have **no group
manifest** — the harness writes manifests at run end and died first —
so their server logs are hash-pinned here instead of by a manifest.
Root cause from the preserved server log
(`runs/m1-sweep-e5m2/logs/vllm-fp8e5m2-reuse.server.log`, sha256
424aeacf…; onset twin 221a258e…):

```
INFO [cuda.py:482] Using FLASHINFER attention backend out of potential
  backends: ['FLASHINFER', 'TRITON_ATTN'].
INFO [flashinfer.py:822] FlashInfer resolved query dtypes:
  prefill=torch.float8_e5m2, decode=torch.bfloat16, decode_backend=xqa,
  kv_cache_dtype=torch.float8_e5m2, arch=sm90
...
ValueError: The dtype of q torch.float8_e4m3fn does not match the
  q_data_type torch.float8_e5m2 specified in plan function.
RuntimeError: Engine core initialization failed.
```

On sm_90 the FP8-Q prefill path plans FlashInfer with
`q_data_type=e5m2` (following the KV dtype) but vLLM's query
quantization emits `float8_e4m3fn` unconditionally; FlashInfer's dtype
check rejects the mismatch and the engine core dies. Operational
markers: `SWEEP_e5m2_EXIT=1` and the final `RUN_FINISHED exit_code 1`
are machine-captured in `runs/m1-monitor-events.log`; the
`ONSET_e5m2_EXIT=1` marker and the harness traceback line
(`serve.py line 107: RuntimeError: vllm exited during startup (code
1)`) were observed in the live run.log before the A1 overwrite and
survive **only in this prose capture** — the onset crash itself is
independently proven by the preserved onset server log (221a258e…),
which records a separate engine start dying with the identical
ValueError, and by the frozen `serve.py` raise site the quote matches.

Scope of the claim: the crash occurred under vLLM's **default backend
selection** — both logs show `auto` resolving to FLASHINFER "out of
potential backends: ['FLASHINFER', 'TRITON_ATTN']" — and the
TRITON_ATTN override path was **not exercised** (it is outside the
frozen matrix). So what m1 establishes is: as-shipped, out-of-the-box
`--kv-cache-dtype fp8_e5m2` on H100 fails loudly at startup at v0.26.0;
whether a non-default backend override could still serve (and what
quality it would produce) is unmeasured. Upstream status (dated search:
`notes/e5m2-crash-upstream-search.md`): this specific FlashInfer
q-dtype planning regression **appears unreported**; the broader "e5m2
accepted by CLI, crash at engine init on H100" class has prior lineage
(vllm#42587, fixed by #42685 before 0.26.0 — the fix is why FLASH_ATTN
correctly rejects e5m2 here — and the stale sm_90 fp8 JIT failure
#31843).

### e4m3: the marketed path silently kills tool calling completely

The configuration vLLM's 2026-04-22 blog calls "ready to be the default
starting point" (calibration-free e4m3, FA3 on Hopper — backend line in
the manifested server log: `Using FLASH_ATTN attention backend` +
`Using FlashAttention version 3`) reproduces the case study's silent
death signature exactly, at 7B, on the agentic channels:

- **Trajectories** (5 seeds × 40 turns × both cache modes, G0–G3 PASS):
  action_error **1.000 [1.000, 1.000]**, no_call 1.000, in every turn
  bin — flat from turn 0, no depth compounding. fp16 twin: 0.006.
  Denominator note: fp16 has 350 scoreable tool turns vs e4m3's 400 —
  the scripted turn-7 recall probe only replaces a tool turn after an
  earlier successful lookup, which never happens under e4m3; the
  contrast is unaffected (e4m3 is dead at turn 0, where prompts are
  byte-identical across arms) and the larger denominator is
  conservative for the 1.000 estimate.
- **Onset** (320 probes): call_present **0.00 in all 16 cells**,
  including the minimum tested context (length-0 cells measure ~1.0k
  prompt tokens once tool schemas + system prompt are counted) — no
  context cliff; the channel is dead at the shortest tested prompt.
- **Silence**: G0 coherence passes (word overlap 8 of a maximum 8
  distinct words — a byte-perfect echo of the probe sentence).
  The probe records the phenomenology: exact-echo of
  `KVDRIFT-VLLM-FP8E4M3-REUSE-7429` returns
  `KVDRIFT-VLLM-FPPEE4MPPREUSEP4P` (character-level copy corruption),
  and tool turns emit mangled call markup (e.g.
  `<functionlookup_item', {"sku": "5555>>` — trajectory_3000 turn 0)
  that no parser can accept: structural corruption of the call format,
  not call suppression by choice.

Attribution is tuple-scoped per METHOD.md: (vLLM 0.26.0, H100 sm_90,
FA3 native-fp8 path, Qwen2.5-7B-Instruct, calibration-free e4m3) →
total silent agentic death. Combined with the case study's
(0.26.0, sm_89/sm_80, FlashInfer FA2 dequant path, e5m2) → same death,
the silent failure now spans both fp8 formats and **two** attention
kernel families for this model (the third candidate path — FlashInfer
FP8-Q/XQA with e5m2 on sm_90 — failed loudly at startup and so
contributes nothing to the *silent* span). Two independent
format×kernel-family combinations are consistent with the
model-conditional numerics interpretation (Qwen2.5-7B KV outlier
structure vs calibration-free scale-1.0 fp8) rather than a single
kernel defect — though this milestone alone cannot exclude an
FA3-specific numeric bug; the mechanism claim inherits the case
study's cross-condition evidence. The frozen classifier's e4m3 label
"PERSISTS" is defined by the dead-channel signature (zero calls at all
onset cells + trajectory no_call ≥ 0.99), not by continuity with the
case study's e4m3-on-Ada result (which was token salad at vLLM 0.19.1,
a different regime at a different version): what "persists" on Hopper
is the silent-death *shape*, carried by the other fp8 format.

### Answer to the milestone questions

- **Q1 (persistence)**: not persistence but **escalation-to-loud**
  under the default path: e5m2 fails at startup on sm_90 (ERROR) under
  vLLM's default backend selection, so the silent e5m2 regime was not
  observable on Hopper in this design — by accident of a dtype-plumbing
  bug, not by numerics. (Whether a non-default backend override could
  reach a silent e5m2 regime on sm_90 is unmeasured.)
- **Q2 (marketed-path audit)**: calibration-free e4m3 on Hopper —
  vLLM's recommended default-ready configuration — **silently zeroes
  the tool-call channel for Qwen2.5-7B-Instruct** while the blog's
  reasoning benchmarks (≥8B, different models, no structured output)
  show near-lossless. The "near-lossless" claim does not transfer to
  the agentic channel for this model (n=1 model; the favored mechanism
  predicts the failure is model-conditional, so other 7B-class models
  may well match the blog — untested here).
- **Q3 (attribution)**: backend selection captured per leg from
  manifested logs: fp16→FLASH_ATTN, e4m3→FLASH_ATTN/FA3 (native fp8),
  e5m2→FLASHINFER (FP8-Q, crash). Note the deployed priority list
  (`FLASH_ATTN` first on sm_90 for supported dtypes) differs from the
  collision scan's source-reading prediction (FLASHINFER first) —
  exactly why the pre-registration required logging over documentation.

## Cost (actual)

Pod 00:22:26–00:56:48 UTC ≈ 0.57 h × $2.99 ≈ **$1.71** (vs $15
stop-loss; $7–9 estimate — the e5m2 crashes refunded ~40 min of
projected serving time). Program GPU spend to date ≈ **$1.71** of $150.

## Audit (per AUDIT.md; commit audited: afd988f)

Panel: claim inventory (23 claims) → three parallel critics (evidence,
methods, provenance) → two independent skeptics per blocking objection.
Every critic recomputed from raw artifacts: all pinned sha256s
re-verified; frozen analyzers re-executed byte-identically
(analysis-m1.json, both sweep analysis.json files); action_error /
no_call / all 32 onset cells recomputed from raw JSONL with independent
scripts; the C21 calibration run re-executed against the case-study A100
artifacts (PASS/PERSISTS/0.003/1.000 reproduced); server-log quotes
verified verbatim; git ordering verified (42f5677 00:13 → 71bba72 00:16
→ afd988f 01:00, linear, no rewrites, frozen files untouched after
pre-registration).

**Verdict: 3 confirmed blocking objections → revision round 1 (this
commit). Post-revision, all three are addressed; no machine gate was
overridden.**

Confirmed blocking (each upheld by 2/2 skeptics):

- **B1 — committed tree failed its own G3 gate**: the step-1 analyzer's
  derived analysis.json was written into the manifested sweep dirs
  after local verification, so `kvdrift verify` failed G3 at the landing
  commit. Fixed by amendment A2 (files moved out; frozen command now
  reproduces PASS from the tree). All raw artifacts hash-verified clean
  throughout — the defect was placement of a derived file, not data
  integrity.
- **B2 — "three attention kernel families" over-counted**: the silent
  failure is evidenced in exactly two families (FlashInfer FA2
  dequant-on-load: case-study e5m2 on sm_89/sm_80; FA3 native fp8: e4m3
  here); the FP8-Q/XQA path crashed before serving and cannot carry a
  silent span. Sentence corrected to "two", with the loud-failure
  caveat, and the PERSISTS-label semantics clarified.
- **B4 — e5m2 universality + "unreported" overclaim**: "cannot serve on
  H100 at all" claimed the fate of the untested TRITON_ATTN override,
  and "unreported upstream" lacked a post-discovery search artifact
  (skeptic search surfaced vllm#42587/#42685 as the phenomenon-class
  lineage, while confirming the specific q-dtype regression appears
  unreported). Rescoped to default backend selection; dated search note
  landed as notes/e5m2-crash-upstream-search.md; Q1 answer rescoped.

Unconfirmed blocking (1/2 skeptics; recorded, non-gating):

- **B3 — ONSET_e5m2_EXIT / traceback provenance**: the doc's provenance
  labeling ("captured from the live run.log before the A1 overwrite")
  was ruled honest and the ERROR conclusion independent of the contested
  strings; the advisory residue (mark the ONSET marker as surviving only
  in prose) is incorporated in the revised §Results anyway.

Advisory (recorded, non-gating; incorporated where marked):

- A1 screened **legitimate** by the provenance critic (infra fix, not
  result-shopping). e5m2 logs are doc-hash-pinned, not group-manifested
  (now stated explicitly). fp16-vs-e4m3 tool-turn denominators differ
  350 vs 400 via the state-dependent turn-7 recall probe (now disclosed;
  also: the harness docstring's "pure function of seed" claim is
  imprecise — carried to the write-up milestone). Onset length-0 cells
  measure ~1.0k prompt tokens with schemas (now corrected). Q2 scoped to
  the model, not the scale (now corrected). C1/C23 pod metadata and cost
  are operational narrative with no artifact (accepted as such;
  METHOD.md requires cost reporting). The pre-registration's analyzer
  calibration claim was re-executed successfully by the evidence critic
  against the case-study repo but is not committed here (accepted:
  cross-repo artifacts remain in jinminghe950/agentic-research).

Audit rounds used: 1 of ≤3. GPU cost of audit: $0.

## Referee pass (paper; commit audited: 48bdc21)

Panel per AUDIT.md on `paper/paper.md` + the two upstream report
drafts: claim inventory (22 claims) → three parallel critics. All three
recomputed from raw artifacts (G-cite re-run PASS 20/20; gates re-run
PASS at HEAD; frozen classifier reproduced byte-identically; every
rate, CI, cell, and verbatim quote re-derived from JSONL/logs).

**Verdict: 0 blocking objections against the paper's conclusions; 2
blocking objections against the report drafts, conceded without
contest after direct mechanical re-verification, and fixed together
with 10 advisories in this revision.**

Conceded blocking (report drafts):

- The e4m3 draft claimed "800 trajectory turns" (this run produced
  400; 800 was the case study's figure) and "~1k to ~9k prompt tokens"
  (measured max 8175). Corrected to 400 / ~8.2k.
- The e5m2 draft asserted "(also crashes without the tool flags" —
  untested (both preserved starts had the flags). Reworded as an
  explicit inference, not an observation.

Advisories incorporated: abstract/§3.1 "unreported" hedged to
"apparently unreported (dated tracker search)"; "worse than either
registered prediction" reframed (both outcomes were inside the
registered space — what happened is the two worst registered branches
co-occurred); §2 manifest/gate claim scoped to the four data-producing
groups with on-pod verification marked operational; greedy-decoding
citation re-anchored to the manifested client code, CUDA version
marked operational; degenerate e4m3 CI labeled (exact binomial 95%
lower bound ≈0.991 on 400/400) and the pooled/per-config analyzer
hybrid stated; coherence probe corrected from "8/9" to 8-of-8 (the
pangram has 8 distinct words; the echo was byte-perfect — stronger
than previously stated); recompute-probe citation scoped to its own
corrupted nonce; recall-probe schedule corrected (turns 7, 15, 23, 31,
39 — 50 replaced turns across the fp16 arms) with the harness
docstring imprecision surfaced; confounded RUN_FINISHED exit-code
corroboration dropped in favor of the SWEEP marker + onset server log;
cost restated with its operational provenance; the cross-repo
analyzer-calibration claim marked not machine-checkable from this repo.

Referee rounds used: 1. GPU cost: $0.
