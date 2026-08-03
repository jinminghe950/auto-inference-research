# [DRAFT — ready to file against vllm-project/vllm]

**Title:** Calibration-free `--kv-cache-dtype fp8_e4m3` silently
disables tool calling for Qwen2.5-7B-Instruct on H100 (FA3 path):
fluent prose, zero parseable tool calls

## Summary

With `--kv-cache-dtype fp8_e4m3` (calibration-free, default scale 1.0)
on an H100 SXM, Qwen2.5-7B-Instruct served by vLLM 0.26.0 stops
emitting parseable tool calls **entirely**: zero `tool_calls` across
800 multi-turn agent trajectory turns (5 seeds × 40 turns × prefix
caching on/off) and 0/320 single-turn probes at context lengths from
~1k to ~9k prompt tokens (hermes parser, `--enable-auto-tool-choice`).
Free-text output stays fluent — a coherence probe passes — but carries
character-level copy corruption (an exact-echo request for
`KVDRIFT-VLLM-FP8E4M3-REUSE-7429` returns
`KVDRIFT-VLLM-FPPEE4MPPREUSEP4P`), and tool turns emit structurally
mangled call markup (e.g. `<functionlookup_item', {"sku": "5555>>`)
that no parser can accept. With `--kv-cache-dtype auto` the identical
workload on the same pod scores 99.4% action-correct, so the harness
and parser are not at fault. The attention backend for both configs is
FLASH_ATTN with FlashAttention 3 (native fp8 path for e4m3).

This is the configuration the April 2026 vLLM blog ("The State of FP8
KV-Cache and Attention Quantization in vLLM") evaluates as
near-lossless and "ready to be the default starting point" — but that
evaluation used reasoning/long-context benchmarks at ≥8B and never
measured structured output. At 7B, for this widely deployed model, the
flag is a total, silent kill of the agentic channel: a short chat
smoke test shows a healthy server while every tool call dies.

## Environment

- vLLM 0.26.0 (pip wheel), torch 2.11.0, Python 3.12
- NVIDIA H100 80GB HBM3 (sm_90), driver 580.126.09, CUDA 13.0
- Model: `Qwen/Qwen2.5-7B-Instruct` (unquantized weights)
- Serve: `vllm serve Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype
  fp8_e4m3 --max-model-len 24576 --enable-auto-tool-choice
  --tool-call-parser hermes --seed 0` (prefix caching on or off — no
  difference)
- Requests: OpenAI chat completions, 12 function schemas,
  `tool_choice: auto`, temperature 0

## Minimal repro

1. Serve as above.
2. Send: system "You are a warehouse assistant with tool access.",
   user "Look up SKU-1234." with a `lookup_item(sku)` tool schema.
3. Observe: no `tool_calls` (320/320 single-turn probes in our runs);
   content is prose or deformed markup.
4. Restart with `--kv-cache-dtype auto`: the same request returns a
   correct `lookup_item` call.

## Relationship to known issues

- #41343 reports the same silent-corruption shape for **e5m2** on
  Qwen-VL models (Ada) and suggests e4m3 as the workaround — this
  report shows calibration-free **e4m3 itself** produces the failure
  for a text-only Qwen2.5 model on Hopper's FA3 path, so "switch to
  e4m3" is not a safe fix for this family without calibrated scales.
- Prior small-model evidence: Qwen2.5-class 7B KV-quantization fragility
  (extreme K-channel outliers) is documented in the literature; a
  startup warning for calibration-free fp8 KV on known-outlier
  families, or a structured-output eval in the fp8-KV qualification
  matrix, would prevent silent deployment damage.

Hash-manifested artifacts (per-turn JSONL, probes, server logs,
manifests) and the deterministic harness are in our research repo
(`research/fp8-kv-hopper/`, runs `m1-sweep-e4m3`, `m1-onset-e4m3`,
fp16 controls `m1-sweep-fp16`, `m1-onset-fp16`).
