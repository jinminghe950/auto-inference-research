# Collision Scan — Does vLLM fp8-KV tool-calling death persist on Hopper?

**Problem 002 artifact** · Scan date: 2026-08-03 · Method: 5 independent
search agents (vLLM issues/PRs, vLLM releases/docs, academic literature,
practitioner field reports, kernel/hardware source analysis) · ~73
distinct searches, ~123 tool calls · ~30 sources verified in depth against
full issue pages, release notes, paper texts, and vLLM/FlashInfer source
at the v0.26.0 tag.

Claim under test (from the KV-drift case study, `jinminghe950/
agentic-research` `research/kv-drift/`): vLLM `--kv-cache-dtype fp8_e5m2`
(calibration-free, scale 1.0) silently kills tool calling for
Qwen2.5-7B-Instruct — zero parseable `tool_calls` across 960 single-turn
probes and 800 trajectory turns, fluent free text with character-level
copy corruption — identically on RTX 4090 (sm_89) and A100 (sm_80), vLLM
0.26.0. Open question: does it persist on H100 (sm_90)?

## Verdict

**No exact collision.** No published work, issue, or field report measures
fp8-KV output quality for **tool calling / structured output** on
**Hopper**, for **any small model**, in **either fp8 dtype**; the specific
Qwen2.5-7B-Instruct e5m2 tool-calling failure is unreported upstream; and
no fix has shipped (v0.26.0, released 2026-07-27, is the current latest
release — the case study's pin is the present, not the past).

**But the scan forces a repositioning.** Three pieces of the original
headline are already claimed elsewhere, so the novel surface is narrower
than problem 002's sketch assumed:

1. *"e5m2 + default scale silently corrupts Qwen-family output"* — filed
   as vLLM issue #41343 (2026-04-30, open, no fix): Qwen2-VL-2B /
   Qwen2.5-VL-3B on L40S (Ada), degenerate/hallucinated output, e4m3 or
   calibrated scales as workarounds, explicitly attributed to Qwen-family
   weight distributions.
2. *"Small Qwen-class 7B collapses under KV-cache quantization while
   14B/32B survive"* — arXiv:2510.18672 (vLLM 0.8.1, both fp8 dtypes,
   A6000/4090/A100, DeepSeek-R1-Distill-Qwen-7B "almost complete
   performance deterioration").
3. *"Calibration-free e5m2 in vLLM is categorically worse than e4m3 and
   silently harmful on Qwen"* — arXiv:2606.09864 (30.3% conditional
   refusal-flip vs 7.1% e4m3 vs 0.2% simulated 8-bit; safety metrics).

What remains unclaimed is precisely the conjunction this study runs:
**tool-calling parseability as the failure detector × text-only
Qwen2.5-7B-Instruct × controlled same-engine cross-GPU-generation
comparison (sm_89/sm_80 vs sm_90) × e5m2-vs-e4m3 on Hopper**.

**Urgency: sharp.** vLLM's official blog (2026-04-22) markets
calibration-free fp8-KV as "ready to be the default starting point" on
Hopper, measured only for e4m3, only at ≥8B, with zero structured-output
evaluation; #41343 is open and active; the arXiv angle judges the
e5m2-on-Hopper cell "a conspicuous gap likely to be filled soon."

## Why the H100 leg is not a foregone conclusion (source analysis)

From vLLM v0.26.0 and FlashInfer 0.6.14 source (verified file-level by
the kernel-path agent):

- `FLASH_ATTN` **never supports e5m2** on any architecture (its
  `supported_kv_cache_dtypes` admits fp8/e4m3 only via FA3 on sm_90), so
  with `fp8_e5m2` **all three architectures select the FLASHINFER
  backend** — same backend *name*, different kernels:
  - **sm_80 / sm_89** (the case study's legs): FlashInfer FA2-template
    kernels, KV dequantized on load, Q stays bf16. The 4090's fp8 tensor
    cores are unused. The two measured legs ran *identical* kernel
    families — a pure cross-arch numerics replication.
  - **sm_90 prefill**: Q is *itself quantized to e5m2*
    (`scaled_fp8_quant`) and FlashInfer's **native FA3 fp8 tensor-core
    path** runs — a numerics regime the measured legs never entered.
  - **sm_90 decode**: routed to TRT-LLM's **XQA kernel**, which vLLM
    gates in for any `fp8*` — but whose documented KV dtype support is
    **e4m3 only**. e5m2 through XQA is undocumented territory: correct
    output, silent misbehavior, or a loud error are all live outcomes.
- This sm_90 fp8 path has a history of outright JIT/compile failure
  (vllm#31843, closed stale) and vLLM's own code carries a
  `NotImplementedError` citing "FP8-Q reliability issues" — a loud
  failure on H100 is a real possibility and would itself be a
  deployment-relevant finding.
- TensorRT-LLM ships **e4m3 only** (e5m2 does not exist there); SGLang
  ships both with a generic accuracy disclaimer. Nobody documents
  small-model or structured-output behavior.

So the study tests materially different code on sm_90, and no source
predicts the outcome a priori. Both persistence and escape are
publishable; so is a loud failure.

## Nearest neighbors

| Work | Date | Overlap | Missing delta |
|---|---|---|---|
| [vLLM #41343: fp8_e5m2 silently corrupts Qwen-VL output with default scaling](https://github.com/vllm-project/vllm/issues/41343) | 2026-04-30, open | ~60% | VL models (2B/3B), not text-only 7B-Instruct; free-text corruption only — no tool-calling metric; Ada only (L40S), no A100/H100; vLLM 0.19.0; a bug report, not a controlled study |
| [vLLM blog: The State of FP8 KV-Cache and Attention Quantization](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html) | 2026-04-22 | ~45% | **e4m3 only — e5m2 never mentioned**; zero tool-calling/structured-output evals; smallest model 8B, no Qwen2.5; H100/H200/B200 only, no Ada/Ampere arm; documents a Hopper-specific FA3 fp8 accumulation bug (NIAH 91%→13%, fixed) — GPU-generation-dependent fp8-KV quality is already on record, for e4m3 |
| [Reasoning LM Inference Serving Unveiled](https://arxiv.org/abs/2510.18672) | 2025-10 | ~50% | Both fp8 KV dtypes destroy a Qwen2.5-based 7B (A6000/4090/A100, vLLM 0.8.1) — but no Hopper, no tool calling, no e5m2/e4m3 isolation, reasoning-distill model not the Instruct model |
| [Alignment Collapse Under KV Cache Quantization](https://arxiv.org/abs/2606.09864) | 2026-06 | ~35% | e5m2-in-vLLM categorically worse than e4m3 on Qwen, silently — safety/refusal metrics only, hardware unreported, no tool calls |
| [KGA IT: KV Cache Management 2026](https://kga-it.com/en/blog/ml-inference-kv-cache-management-2026) | 2026-04 | ~30% | Only e5m2-on-Hopper-family quality numbers found anywhere: 70B-class aggregates (−0.3/−0.6%) on 8×H200 — no small model, no structured output, no cross-arch comparison |
| [vLLM #10411: KV quantization + GGUF "quite poorly"](https://github.com/vllm-project/vllm/issues/10411) | 2024-11, closed | ~30% | Qwen2.5-7B GGUF: e5m2 repeats a word forever, e4m3 degenerates — confounded by GGUF weight quant, closed not-planned, no metric |
| [HEAL: Numerical Instability in LLM Inference](https://arxiv.org/abs/2606.21023) | 2026-06 | ~30% | Hardware-conditional inference numerics precedent (cross-GPU divergence, SASS-level root cause) — no fp8-KV, no task-quality metric |
| [ACBench: compressed LLMs' agentic capabilities](https://arxiv.org/abs/2505.19433) | 2025-05 | ~30% | Tool-use degradation under compression on Qwen2.5 — weight quant/pruning only, **no KV axis**, no hardware axis |
| [vLLM #37618: fp8 KV degraded accuracy, B200 Qwen3.5](https://github.com/vllm-project/vllm/issues/37618) | 2026-03 | ~20% | Arch-specific Qwen fp8-KV regression (Blackwell) — sparse detail, no dtype split, no tool calling |
| [vLLM #46863: Llama-4-Scout tool calls left in content](https://github.com/vllm-project/vllm/issues/46863) | 2026-06 | ~20% | Proves empty-`tool_calls` can be parser plumbing, not KV numerics — the confounder this instrument's fp16 control + copy-corruption channel discriminates |
| [vLLM #31843: SM90 FlashInfer fp8 KV fails to compile](https://github.com/vllm-project/vllm/issues/31843) | 2025 | ~20% | Evidence sm_90 fp8 prefill is distinct, fragile code — supports rather than scoops the H100 leg |
| [vLLM #37554: --calculate-kv-scales corrupts fp8 KV](https://github.com/vllm-project/vllm/issues/37554) | 2026-03, fixed | ~15% | Inverse polarity: the *calibration* path is the bug; scale-1.0 e4m3 declared fine for typical activations |
| [vLLM #47549: fp8-KV backend selection regression on SM75](https://github.com/vllm-project/vllm/issues/47549) | 2026-07 | ~10% | Backend selection for fp8-KV churns version-to-version — design consequence: capture the selected backend as evidence |
| [SGLang #22671 / #5700: fp8 KV severe degradation / nonsense](https://github.com/sgl-project/sglang/issues/22671) | 2025–26 | ~15% | Cross-engine anecdotes, e4m3, no metric |
| [FlashInfer paper](https://arxiv.org/abs/2501.01005) | 2025-01 | ~5% | Documents the FA2(≤sm_89)/FA3(sm_90) template split and fused fp8 dequant — background attribution citation |

Also scanned (lower overlap): Kitty (MLSys'26), DGAP (2607.16248), KVTuner
(2502.04420), HQMQ (2605.27646, Qwen2.5-7B K-channel outliers >250×
median — mechanism support), FP8-across-accelerators (2502.01070),
nondeterminism studies (2506.09501), "Give Me BF16" (2411.02355), vLLM
docs (no e5m2 accuracy caveat — the silence is itself a finding), TRT-LLM
precision docs, Spheron/Fireworks/DEV-community deployment guides
(the "near-lossless / safe for production" claim is widespread and never
backed by structured-output measurement), HF Qwen3.6-FP8 "garbage output"
thread (false positive — LiteLLM misconfiguration), vLLM #39137/#23922/
#4532/#25800/#45562 (e5m2 policing, ROCm fp8-KV silent failure), SGLang
docs.

## Sharpened claim

> On the current latest vLLM (0.26.0), using a frozen tool-calling
> fidelity instrument with exact separable channels, we determine whether
> the silent fp8_e5m2 tool-calling death of Qwen2.5-7B-Instruct is
> **hardware-conditional**: the measured sm_89/sm_80 legs (identical
> FA2-template dequant-on-load kernels; total silent channel death) vs
> sm_90, where the same flag uniquely exercises FlashInfer's native FA3
> fp8 prefill (with e5m2-quantized queries) and TRT-LLM's XQA decode
> kernel (documented e4m3-only, gated in by vLLM for any fp8). We
> additionally deliver the **first agentic/structured-output audit of
> calibration-free fp8_e4m3 on Hopper** — the exact configuration vLLM's
> April 2026 blog markets as "ready to be the default starting point,"
> evaluated there only on reasoning/long-context benchmarks at ≥8B.

Positioning: completes the case study's (engine, GPU, model) attribution
tuple on the one architecture with a genuinely different fp8 code path;
extends #41343's mechanism from VL free-text to text-only agentic
channels; audits the blog's default-on recommendation on the metric class
it skipped. Any outcome — persists / clean / mixed / loud failure — is
deployment-relevant and unclaimed.

## Refinements forced by the scan

1. **Reposition novelty.** "e5m2 breaks small Qwen" is no longer the
   headline (pre-empted by #41343 + 2510.18672 + 2606.09864). The
   headline is hardware-conditionality of an already-observed silent
   failure, plus the agentic-channel audit of the marketed e4m3 path.
2. **Capture the selected attention backend per leg as manifested
   evidence** (server logs are already hash-manifested by the harness).
   Backend selection for fp8-KV is demonstrably version-unstable
   (#47549) and the docs' support tables are auto-generated; the
   kernel-path attribution must rest on logged fact, not documentation.
3. **Parser-confounder discrimination is load-bearing** (#46863): empty
   `tool_calls` can be plumbing, not numerics. The design already
   discriminates: fp16 control on the identical parser/stack, the
   copy-fidelity channel (character corruption is not a parser artifact),
   and G0 probes.
4. **Register a loud-failure outcome class.** #31843 and the XQA dtype
   gap make an engine-level error on sm_90 a live outcome. Pre-registered
   handling: run groups are split per precision so a crash cannot destroy
   other groups' manifested evidence; an engine failure lands fail-closed
   as an ERROR-regime finding evidenced by server logs.
5. **Upgrade e4m3 from canary to co-primary.** It directly audits the
   blog's default-readiness claim at 7B on agentic channels. Within-repo
   history (m2: e4m3 = token salad on Ada at vLLM 0.19.1) vs the blog
   (near-lossless on Hopper at ≥8B, e4m3, FA3) makes the outcome
   genuinely open at 0.26.0/sm_90.

## Milestone / budget plan

- **m1 — H100 probe + sweep** (this problem's single planned GPU
  milestone): six per-precision run groups on one H100 SXM (SECURE,
  $2.99/hr, host CUDA ≥ 13.0): trajectory sweeps {fp16, e4m3, e5m2} ×
  {reuse, recompute} × 5 seeds (3000–3004) × 40 turns (exact case-study
  conditions), and onset grids {fp16, e4m3, e5m2} × {near, far} × 8
  lengths × 20 reps. Gates G0–G4 + onset row-count per group; frozen
  analyzer with registered regime classification (validated on the case
  study's A100 data before registration). Est. ~2.2–2.8 h ≈ **$7–9**;
  stop-loss $15.
- **m2 — conditional powered characterization**: runs only if the
  registered classification returns MIXED (a measurable intermediate
  regime); pre-registered separately before any m2 data. Est. ≤$10.
- **m3 — paper + upstream reports**: gated write-up; update the case
  study's vLLM issue draft with the Hopper answer; if the XQA/e5m2
  dtype-gate mismatch is confirmed in behavior, a second upstream note.
- Program budget: default $150 ceiling; projected total spend <$25.
