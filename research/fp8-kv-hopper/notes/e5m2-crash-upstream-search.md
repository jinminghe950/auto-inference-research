# Upstream search: is the sm_90 e5m2 startup crash already reported?

Date: 2026-08-03 (post-discovery; the collision scan predates the crash
and searched the silent tool-calling failure, not this regression).
Searched: vLLM issue tracker via web search (exact-string and topical).

Queries and findings:

1. Exact-string: `vllm "does not match the q_data_type" fp8_e5m2` —
   **no issue reports this ValueError**. Hits are the known neighbors
   only: [#41343](https://github.com/vllm-project/vllm/issues/41343)
   (e5m2 silent corruption, Qwen-VL, Ada),
   [#23922](https://github.com/vllm-project/vllm/issues/23922)
   ("Unrecognized FP8 dtype: fp8_e5m2", older loud failure, different
   error), [#39137](https://github.com/vllm-project/vllm/issues/39137)
   (e5m2 gate over-fires on quantized checkpoints),
   [#4532](https://github.com/vllm-project/vllm/issues/4532) (fp8-KV
   RFC), [#3990](https://github.com/vllm-project/vllm/issues/3990).
2. Nearest phenomenon-class prior:
   [#42587](https://github.com/vllm-project/vllm/issues/42587) (filed
   2026-05-14, H100, vLLM 0.14.1): "vllm serve crashes at engine init
   for several --kv-cache-dtype values accepted by CLI (fp8_ds_mla,
   fp8_e5m2, fp8_inc, bfloat16)" — same user-facing class (e5m2 accepted
   by CLI, crash at engine init on H100), different mechanism/version;
   addressed by [PR #42685](https://github.com/vllm-project/vllm/pull/42685)
   (FLASH_ATTN made to reject e5m2 correctly — which is why 0.26.0's
   candidate list on sm_90 is ['FLASHINFER', 'TRITON_ATTN']).
3. Also adjacent: [#31843](https://github.com/vllm-project/vllm/issues/31843)
   (SM90 FlashInfer fp8-KV JIT compile failure, closed stale).

Conclusion (scoped): the **specific** 0.26.0 regression observed in m1 —
FlashInfer FP8-Q prefill planned with `q_data_type=e5m2` while vLLM's
query quantization emits `float8_e4m3fn`, ValueError at warmup —
**appears unreported as of this search**. The broader "e5m2 crashes at
engine init on H100" phenomenon class has prior reports (#42587, fixed
pre-0.26.0) and the sm_90 fp8 path has a history of loud failures
(#31843). Any upstream filing should cite #42587/#42685 as lineage.
