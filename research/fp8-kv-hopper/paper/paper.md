# Loud on Hopper, Silent on the Marketed Path: vLLM's Calibration-Free FP8 KV Cache vs. Tool Calling on H100

**auto-inference-research, problem 002 (fp8-kv-hopper) · 2026-08-03**
Pre-registered, machine-gated, adversarially audited. Every empirical
claim below cites a sha256-pinned artifact as `[[art:path@hash8]]`
(paths relative to `research/fp8-kv-hopper/`), verified by
`analysis/check_citations.py`.

## Abstract

The KV-drift case study established that vLLM's calibration-free
`--kv-cache-dtype fp8_e5m2` silently destroys tool calling for
Qwen2.5-7B-Instruct on RTX 4090 (sm_89) and A100 (sm_80). Hopper — the
architecture where fp8 KV cache is actually marketed, and where vLLM's
April 2026 blog declares calibration-free fp8 KV "ready to be the
default starting point" — was the untested cell. We ran a
pre-registered probe-and-sweep milestone on one H100 SXM (vLLM 0.26.0,
the latest release at execution time) and found the failure surface is
worse than either registered prediction, in two different ways. First,
**e5m2 cannot start at all under vLLM's default backend selection on
sm_90**: FlashInfer's FP8-Q prefill is planned with `q_data_type=e5m2`
while vLLM quantizes queries to `float8_e4m3fn`, and the engine core
dies during warmup with a dtype-mismatch ValueError — a loud,
previously unreported regression in the current release. Second, and
more consequentially for operators: **calibration-free e4m3 — the
exact configuration the blog markets, on the native FlashAttention-3
fp8 path — silently zeroes the tool-call channel for this model**:
action error 1.000 across 5 seeds × 40 turns × both prefix-cache
modes, zero parseable tool calls in all 16 onset cells from the
shortest tested prompt, while coherence probes pass and prose stays
fluent. The fp16 control is clean (action error 0.006) on the same
host, instrument, and parser. The silent fp8 agentic death now spans
both fp8 formats and two attention kernel families for Qwen2.5-7B —
and nothing in the serving stack warns you.

## 1. Question and prior state

The case study (`jinminghe950/agentic-research`,
`research/kv-drift/`) measured, with an exact-channel tool-calling
instrument: vLLM 0.26.0 `fp8_e5m2` KV → zero parseable tool calls at
every context length and trajectory depth for Qwen2.5-7B-Instruct,
with fluent free text, identically on sm_89 and sm_80 — which share
one kernel family for this flag (FlashInfer FA2-template,
dequant-on-load). Our collision scan (five independent search angles,
committed before any of this milestone's data;
`collision-scan.md`) found: (i) no published measurement of fp8-KV
structured-output quality on Hopper for any small model, in either fp8
dtype; (ii) vLLM's fp8-KV blog evaluates **e4m3 only**, at ≥8B, with
no structured-output metric, and recommends it as a default; (iii) the
adjacent silent-corruption mechanism was already filed as vllm#41343
(e5m2 + default scale, Qwen-VL models, Ada) — so the open cells were
precisely e5m2-on-Hopper and agentic-e4m3-on-Hopper; and (iv) on
sm_90, uniquely, the e5m2 flag routes to FlashInfer's native FA3 fp8
prefill plus TRT-LLM's XQA decode — genuinely different code whose
outcome no source predicted.

Design, gates, seeds, matrices (sha256-pinned), analyzer, and the
regime-classification rule were frozen and committed before any data
existed (commit 71bba72; evidence landed in afd988f; the commit
ordering is the pre-registration proof). The registered classification
rule was validated, before registration, against the case study's A100
artifacts, reproducing its published numbers exactly.

## 2. Method (summary)

One H100 SXM 80GB (sm_90; driver 580.126.09, CUDA 13.0), vLLM 0.26.0,
Qwen/Qwen2.5-7B-Instruct, greedy decoding
[[art:runs/m1-sweep-fp16/run_meta.json@2c74ea75]]. Six independent run
groups, split per precision so an engine-level crash in one cannot
destroy another's manifested evidence:

- **Sweep groups** ({fp16, e4m3, e5m2} × {reuse, recompute} × 5 seeds
  (3000–3004, identical to the case study) × 40 turns; max_tokens 192,
  max_model_len 24576): the deterministic warehouse tool-calling
  instrument with separable channels (call presence, parse validity,
  tool selection, argument fidelity, recall), machine gates G0–G4.
- **Onset groups** ({fp16, e4m3, e5m2} × {near, far} × 8 lengths
  (0–8192 filler tokens) × 20 reps = 320 single-turn copy-fidelity
  probes each; gates: full row count + content-addressed manifest).

Every group carries a sha256 manifest over results and harness code;
gates were verified on-pod and re-verified locally on the collected
copies. Analysis is the repo-frozen `analyze.py` (bootstrap CIs over
seeds) plus the pre-registered `analyze_m1.py` classifier
[[art:runs/analysis-m1.json@83bec634]]. Total GPU cost: **$1.71**
(34 min × $2.99/hr). An adversarial audit (three critics, two-skeptic
verification, AUDIT.md protocol) ran on the landed evidence; three
confirmed objections were fixed in a logged revision (milestone doc
§Audit) — the claims below are the post-audit, scoped versions.

## 3. Results

### 3.1 e5m2 on H100: a loud, unreported startup regression

Both attempted e5m2 engine starts (sweep-reuse and the onset config)
died identically during FlashInfer autotune warmup, before serving a
single request. The preserved server logs
[[art:runs/m1-sweep-e5m2/logs/vllm-fp8e5m2-reuse.server.log@424aeacf]]
[[art:runs/m1-onset-e5m2/logs/vllm-fp8e5m2.server.log@221a258e]]
record: backend auto-selection to FLASHINFER "out of potential
backends: ['FLASHINFER', 'TRITON_ATTN']" (FLASH_ATTN categorically
rejects e5m2); the FP8-Q resolution line
`prefill=torch.float8_e5m2, decode=torch.bfloat16, decode_backend=xqa,
arch=sm90`; then

```
ValueError: The dtype of q torch.float8_e4m3fn does not match the
  q_data_type torch.float8_e5m2 specified in plan function.
RuntimeError: Engine core initialization failed.
```

vLLM plans the FlashInfer prefill kernel to expect e5m2 queries but
its query-quantization op emits `float8_e4m3fn` unconditionally. The
machine-captured marker stream confirms the sweep group exited
nonzero and the run ended with exit code 1
[[art:runs/m1-monitor-events.log@4c71b225]].

**Scope** (post-audit): this establishes that as-shipped, out-of-the-box
`--kv-cache-dtype fp8_e5m2` fails loudly at startup on H100 at v0.26.0
under vLLM's **default backend selection**; the TRITON_ATTN override
path was outside the frozen matrix and remains unmeasured, so we make
no claim about non-default routes. A dated tracker search
(`notes/e5m2-crash-upstream-search.md`) found no report of this
specific q-dtype planning regression; the broader "e5m2 accepted by
CLI, crash at engine init on H100" class has lineage (vllm#42587,
fixed by #42685 before 0.26.0 — that fix is why FLASH_ATTN correctly
rejects e5m2 here — and the stale sm_90 fp8 JIT failure #31843).

For operators the polarity matters: on Ada/Ampere this flag fails
*silently* (fluent text, dead tools); on Hopper, today, it refuses to
start. A loud failure is the better failure — but it also means the
marketed fp8 hardware cannot run the calibration-free e5m2 configuration
at all in the current release.

### 3.2 e4m3 on H100: the marketed path silently kills tool calling

The configuration vLLM's 2026-04-22 blog calls "ready to be the
default starting point" — calibration-free `fp8_e4m3`, which on sm_90
runs the native FlashAttention-3 fp8 path (backend line in the
manifested log
[[art:runs/m1-sweep-e4m3/logs/vllm-fp8e4m3-reuse.server.log@4ee62c60]])
— reproduces the case study's silent-death signature at 7B, in full:

| metric (pooled over both cache modes) | fp16 | e4m3 |
|---|---|---|
| trajectory action_error (95% CI per config) | 0.006 [0.000, 0.017] | **1.000 [1.000, 1.000]** |
| trajectory no_call | 0.006 | **1.000** |
| onset call_present, all 16 cells | 1.00 | **0.00** |
| onset sku_exact, all 16 cells | 1.00 | **0.00** |
| G0 coherence probe (fluent prose) | pass | pass |

Trajectory evidence: 5 seeds × 40 turns × two cache modes, gates
G0–G3 PASS, manifested
[[art:runs/m1-sweep-e4m3/manifest.json@ba502802]]
[[art:runs/analysis/m1-sweep-e4m3.analysis.json@910c0e07]]; raw
per-turn records e.g.
[[art:runs/m1-sweep-e4m3/config_vllm-fp8e4m3-recompute/trajectory_3000.jsonl@d3ff0540]].
Onset evidence: 320/320 probes
[[art:runs/m1-onset-e4m3/config_vllm-fp8e4m3/onset.jsonl@b854bde7]]
[[art:runs/m1-onset-e4m3/manifest.json@1414eecb]]. The error is flat
from turn 0 and from the shortest tested prompt (length-0 onset cells
still measure ~1.0k prompt tokens once tool schemas and system prompt
are counted): there is no depth compounding and no context cliff — the
channel is dead on arrival.

The failure is **silent and structural**, not call suppression: the
G0 probe passes word-overlap coherence (8/9) while its recorded copy
channel shows character-level corruption — the exact-echo request for
`KVDRIFT-VLLM-FP8E4M3-REUSE-7429` returns
`KVDRIFT-VLLM-FPPEE4MPPREUSEP4P`
[[art:runs/m1-sweep-e4m3/config_vllm-fp8e4m3-reuse/probe.json@bd822795]]
[[art:runs/m1-sweep-e4m3/config_vllm-fp8e4m3-recompute/probe.json@dce5f804]]
— and tool turns emit mangled call markup (e.g.
`<functionlookup_item', {"sku": "5555>>`, trajectory_3000 turn 0
[[art:runs/m1-sweep-e4m3/config_vllm-fp8e4m3-recompute/trajectory_3000.jsonl@d3ff0540]])
that no parser can accept. A short free-text smoke test would show a
healthy server; every agentic request dies.

### 3.3 Instrument sanity

The fp16 control on the same host, weights, parser, and instrument:
G0–G4 PASS including the pre-registered action-correct floor
[[art:runs/m1-sweep-fp16/manifest.json@88dd5a5f]]
[[art:runs/analysis/m1-sweep-fp16.analysis.json@eb7ca141]], probe
channels exact
[[art:runs/m1-sweep-fp16/config_vllm-fp16-reuse/probe.json@1483ecfc]],
onset perfect in all 16 cells
[[art:runs/m1-onset-fp16/config_vllm-fp16/onset.jsonl@8c26473f]]
[[art:runs/m1-onset-fp16/manifest.json@48acc755]], raw trajectories
e.g. [[art:runs/m1-sweep-fp16/config_vllm-fp16-reuse/trajectory_3000.jsonl@d1772c95]].
This kills the parser-plumbing alternative (cf. vllm#46863): the
identical hermes parser on the identical stack parses fp16 calls at
99.4% action-correct, and the copy-corruption channel is not a parser
artifact. Reuse vs recompute is behaviorally transparent at fp16
(H2 contrast +0.000) and uninformative under e4m3 (both modes floored).

### 3.4 Attribution

Backend selection was captured per leg from hash-manifested server
logs (fp16 → FLASH_ATTN
[[art:runs/m1-sweep-fp16/logs/vllm-fp16-reuse.server.log@3c297705]];
e4m3 → FLASH_ATTN with "FlashAttention version 3"
[[art:runs/m1-sweep-e4m3/logs/vllm-fp8e4m3-reuse.server.log@4ee62c60]];
e5m2 → FLASHINFER, crash
[[art:runs/m1-sweep-e5m2/logs/vllm-fp8e5m2-reuse.server.log@424aeacf]]);
the deployed priority order differs from what source-reading predicted,
which is why the pre-registration required logging over documentation.
Combined with the case study: for Qwen2.5-7B-Instruct, total silent
agentic death is now evidenced in **two independent format ×
kernel-family combinations** — (e5m2, FlashInfer FA2 dequant-on-load,
sm_89/sm_80) and (e4m3, FA3 native fp8, sm_90) — consistent with
model-conditional numerics (this model's known extreme KV outlier
structure vs calibration-free scale-1.0 fp8) rather than a single
kernel defect. This milestone alone cannot exclude an FA3-specific
numeric bug; the mechanism claim inherits the case study's
cross-condition evidence (32B sibling clean on identical kernels).

## 4. Limitations

Single model (n=1): the favored mechanism predicts model-conditional
failure, so other 7B models may match the blog's near-lossless numbers
— unmeasured here. Single host/datacenter; single engine version
(0.26.0, latest at execution). The e5m2 result is scoped to default
backend selection; TRITON_ATTN was not exercised, so a silent e5m2
regime on Hopper via override remains possible. The e5m2-recompute
config was never launched (its group aborted at the first config's
startup crash). fp16 and e4m3 trajectory denominators differ (350 vs
400 tool turns) because the scripted turn-7 recall probe only replaces
a tool turn after a successful lookup, which never occurs under e4m3;
e4m3 is dead at turn 0 where prompts are byte-identical across arms,
and the larger denominator is conservative. Quality claims are about
the agentic/structured channel of this instrument; we did not measure
reasoning benchmarks.

## 5. What operators and upstream should take away

1. **Do not enable calibration-free fp8 KV for a 7B-class agent
   deployment without an agentic smoke test.** Coherent chat output is
   not evidence the tool channel survived: under e4m3 on H100 the
   server looks healthy and every tool call dies. A ten-request
   tool-call probe (fp16 vs quantized, same prompts) would have caught
   this.
2. **vLLM should not present calibration-free e4m3 as default-ready
   without structured-output evaluation at small scale.** The blog's
   evaluation (reasoning + long-context, ≥8B, different models) missed
   a total failure mode for a widely deployed 7B model.
3. **The e5m2/H100 startup crash should be fixed or the flag gated.**
   Upstream drafts: `reports/vllm-e5m2-h100-startup-crash.md`
   (q-dtype planning regression, with #42587/#42685/#31843 lineage) and
   `reports/vllm-e4m3-agentic-death-7b.md` (silent agentic death on the
   marketed path, extending #41343 from VL free-text to text-model
   agentic channels).

## 6. Negative and unresolved results

The original question — does the *silent* e5m2 regime persist on
Hopper's different kernels? — is **unresolved and currently
unaskable**: the dtype-plumbing regression blocks the flag before
numerics can act (registered outcome class ERROR). If upstream fixes
the crash, the pre-registered onset+sweep design answers it for ~$3.
The reuse-inheritance axis produced a clean null at fp16 and an
uninformative floor at e4m3. The pre-registered MIXED condition for a
powered follow-up milestone did not trigger.

## Provenance

Commit chain: collision scan `42f5677` → pre-registration `71bba72`
(matrices, gates, classifier frozen; no data) → evidence `afd988f`
(gates verified on-pod and locally) → audit revision `f500302` (three
confirmed objections fixed; verdict recorded in
`milestone-1.md` §Audit). Executor: ephemeral Runpod H100 SXM pod,
token-auth control plane, no credentials on the pod; terminated after
local re-verification. Program GPU spend: **$1.71** of the $150
ceiling. Classifier verdict [[art:runs/analysis-m1.json@83bec634]]:
instrument_sanity PASS, e5m2 = ERROR, e4m3 = PERSISTS (dead-channel
signature; label semantics per milestone doc), milestone-2 condition
not triggered.
