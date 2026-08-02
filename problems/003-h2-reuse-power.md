# 003 — Does prefix-cache reuse amplify quantized-KV damage? (H2, powered)

status: candidate
value: The production default is prefix caching ON over quantized KV. The
case study's H2 (reuse inherits quantization error) ended unresolved-null
at n=10 seeds: int8 point estimates went the predicted direction (+0.119
in deep turns) with CIs straddling zero. A powered answer either indicts
the default combination or certifies reuse as behaviorally free under
quantization — both deployment-relevant.
verifiability: Existing instrument; the contrast is pre-registered and
computed by the frozen analyzer; power analysis from the case study's
measured variance says ~30 seeds for a 0.1 effect at 80% power.
budget: ~$10 (int8 reuse/recompute × 30 seeds × 40 turns on one 4090
≈ 4–5 h).
window: Wide — nobody else measures this axis; arXiv:2601.08343 remains
the only reuse-vs-recompute quality comparison and has no quantization.
sketch: Single-question design: lmdeploy-int8 × {reuse, recompute} × 30
seeds × 40 turns on Qwen2.5-7B (the known-degraded regime, so the
contrast has room to appear), pre-registered paired bootstrap contrast,
plus fp16 legs as the zero-control. Optionally add the cross-family
winner from problem 001 if it shows a mid-range regime.
