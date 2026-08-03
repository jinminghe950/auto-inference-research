# Milestone 1 — H100 probe + sweep: is the fp8-KV tool-calling death hardware-conditional?

**Status: pre-registered.** Everything above §Results is frozen at this
commit, before any data exists; the commit ordering is the proof.
Amendments only via the Deviations log, committed before the data they
govern.

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

(none yet)
