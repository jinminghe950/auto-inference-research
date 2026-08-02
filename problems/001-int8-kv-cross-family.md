# 001 — Is int8-KV fragility Qwen-specific? Cross-family replication

status: candidate
value: The KV-drift case study showed LMDeploy int8 KV destroys
Qwen2.5-7B tool calling (~65% action error) while Qwen2.5-32B is
untouched. Every operator serving a small model with quantized KV needs
to know whether their family is affected; vendors publish no such matrix.
verifiability: Existing tool-calling instrument, unchanged — exact
channels (parse/tool/args/recall), machine gates G0–G4.
budget: ~$10–15 on one 4090 + one A100 spot check; reuses everything.
window: Moderate — model families churn quarterly; the result compounds
into a family × precision matrix over time.
sketch: Replicate the case study's sweep-v3 design on Llama-3.1-8B and a
Mistral-family 7B (tool-calling capable), fp16 + int8 legs, 10 seeds ×
40 turns. Pre-registered predictions: if fragility tracks the known
small-Qwen KV-outlier structure, both non-Qwen families are clean —
turning the case-study inference into a tested mechanism claim. Either
outcome is a strong result: "Qwen-specific numerics" or "small-model
quantized-KV fragility is general".
