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

### e5m2: vLLM 0.26.0 cannot serve `fp8_e5m2` on H100 at all

Both e5m2 groups died identically during FlashInfer autotune warmup,
before serving a single request. Root cause from the manifested server
log (`runs/m1-sweep-e5m2/logs/vllm-fp8e5m2-reuse.server.log`, sha256
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
check rejects the mismatch and the engine core dies. The operational
traceback (captured from the live run.log before the A1 overwrite):
`serve.py line 107: RuntimeError: vllm exited during startup (code 1)`,
for both the sweep (`SWEEP_e5m2_EXIT=1`) and onset
(`ONSET_e5m2_EXIT=1`) groups. So the Ada/Ampere silent killer is not
reachable on Hopper at v0.26.0 — the flag fails **loudly** (an
unreported upstream bug: the latest release cannot serve e5m2 on the
hardware where fp8-KV is marketed).

### e4m3: the marketed path silently kills tool calling completely

The configuration vLLM's 2026-04-22 blog calls "ready to be the default
starting point" (calibration-free e4m3, FA3 on Hopper — backend line in
the manifested server log: `Using FLASH_ATTN attention backend` +
`Using FlashAttention version 3`) reproduces the case study's silent
death signature exactly, at 7B, on the agentic channels:

- **Trajectories** (5 seeds × 40 turns × both cache modes, G0–G3 PASS):
  action_error **1.000 [1.000, 1.000]**, no_call 1.000, in every turn
  bin — flat from turn 0, no depth compounding. fp16 twin: 0.006.
- **Onset** (320 probes): call_present **0.00 in all 16 cells**,
  including the ~40-token minimum context — no context cliff; the
  channel is dead on arrival.
- **Silence**: G0 coherence passes (word overlap 8/9 — fluent prose).
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
the silent failure now spans both fp8 formats and three attention
kernel families for this model, consistent with the model-conditional
numerics interpretation (Qwen2.5-7B KV outlier structure vs
calibration-free scale-1.0 fp8) rather than any single kernel defect —
though this milestone alone cannot exclude an FA3-specific numeric bug;
the mechanism claim inherits the case study's cross-condition evidence.

### Answer to the milestone questions

- **Q1 (persistence)**: not persistence but **escalation-to-loud**:
  e5m2 is unserveable on sm_90 (ERROR), so the silent regime is
  Ada/Ampere-specific for e5m2 — by accident of a dtype-plumbing bug,
  not by numerics.
- **Q2 (marketed-path audit)**: calibration-free e4m3 on Hopper —
  vLLM's recommended default-ready configuration — **silently zeroes
  the tool-call channel at 7B** while the blog's reasoning benchmarks
  (≥8B, no structured output) show near-lossless. The "near-lossless"
  claim does not transfer to the agentic channel at this scale.
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
