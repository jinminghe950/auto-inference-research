# 002 — Does vLLM fp8-KV tool-calling death persist on Hopper?

status: done (m1 executed, audited, refereed; paper + upstream drafts in research/fp8-kv-hopper/)
value: The case study found vLLM fp8_e5m2 KV kills the tool-call channel
outright for Qwen2.5-7B on Ada (4090) and Ampere (A100) — but Hopper
(H100) has dedicated fp8 attention paths and is where fp8-KV is actually
marketed and deployed. If it is clean on H100, the finding becomes a
"non-native-fp8 hardware" warning; if it persists, it indicts the flag's
default everywhere and the upstream issue draft gets much stronger.
verifiability: Existing instrument + G0 coherence/copy probes; a single
probe-and-smoke matrix answers it.
budget: ~$5–10 (H100 ~$2.7/hr, short run; probes + 2-seed smoke before
any full sweep).
window: Sharp — fp8-KV adoption is accelerating with Hopper/Blackwell
fleet growth; the vLLM issue draft (agentic-research
`research/kv-drift/reports/`) should ship with this answer attached.
sketch: Probe-first design: G0 coherence + copy-fidelity + onset grid for
fp8_e5m2 (and e4m3 canary) on H100 SXM, then a 5-seed × 40-turn
confirmation leg only if probes show a measurable regime. Attribution
completes the case study's (engine, GPU, model) tuple coverage.
