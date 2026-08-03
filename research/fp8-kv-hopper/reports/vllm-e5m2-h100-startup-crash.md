# [DRAFT — ready to file against vllm-project/vllm]

**Title:** `--kv-cache-dtype fp8_e5m2` crashes at engine init on H100
(sm_90): FlashInfer prefill planned with `q_data_type=e5m2` but query
quantization emits `float8_e4m3fn`

## Summary

On an H100 SXM (sm_90), vLLM 0.26.0 with `--kv-cache-dtype fp8_e5m2`
cannot start under default backend selection. FLASH_ATTN correctly
rejects e5m2 (post-#42685 behavior), so the selector picks FLASHINFER;
the FP8-Q prefill path then resolves
`prefill=torch.float8_e5m2, decode=torch.bfloat16, decode_backend=xqa,
arch=sm90`, but the query-quantization op produces `float8_e4m3fn`
regardless of the KV dtype, and FlashInfer's plan/run dtype check
raises during kernel warmup:

```
ValueError: The dtype of q torch.float8_e4m3fn does not match the
q_data_type torch.float8_e5m2 specified in plan function.
...
RuntimeError: Engine core initialization failed.
```

Reproduced twice (independent engine starts, different max_model_len,
prefix caching on/off) with identical tracebacks. The same flag on the
same vLLM version serves (with severe quality problems, separately
reported) on RTX 4090 and A100, so this is sm_90-specific plumbing:
either the FP8-Q planner should follow the actual Q quantization dtype
(e4m3fn), or e5m2 should quantize Q to e5m2, or the combination should
be rejected at argument parsing with a clear message rather than after
model load.

## Environment

- vLLM 0.26.0 (pip wheel; torch as pulled by its dependencies), Python 3.12
- NVIDIA H100 80GB HBM3, driver 580.126.09 (CUDA 13.0 per nvidia-smi)
- Model: `Qwen/Qwen2.5-7B-Instruct` (unquantized)
- Serve: `vllm serve Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype
  fp8_e5m2 --max-model-len 24576 --enable-auto-tool-choice
  --tool-call-parser hermes --seed 0` (both reproduced starts included
  the tool flags; the crash occurs during FlashInfer autotune warmup,
  before any request is served, so the tool flags are very likely
  irrelevant — not separately tested)

## Related issues

- #42587 ("vllm serve crashes at engine init for several
  --kv-cache-dtype values accepted by CLI", H100, 0.14.1) — same
  user-facing class; its fix #42685 made FLASH_ATTN reject e5m2, which
  is why the selector now lands on FLASHINFER and hits this newer
  mismatch.
- #31843 (SM90 FlashInfer fp8-KV JIT compile failure, closed stale) —
  prior loud failure on this path.
- #41343 (e5m2 + default scale silently corrupts Qwen-VL output on
  Ada) — the quality problem that awaits behind this crash if it is
  fixed without addressing e5m2 numerics.

Full hash-manifested server logs and a reproducible harness are in our
research repo (`research/fp8-kv-hopper/`, evidence files
`runs/m1-sweep-e5m2/logs/vllm-fp8e5m2-reuse.server.log` and
`runs/m1-onset-e5m2/logs/vllm-fp8e5m2.server.log`).
