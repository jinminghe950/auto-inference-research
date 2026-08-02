# Problem queue

One md file per candidate, numbered (`NNN-slug.md`). At session start the
user names one; the session executes it per `CLAUDE.md` and moves its
status forward. Statuses: `candidate → scanned (collision scan done) →
active → done | parked`.

## Candidate format

```
# NNN — Title
status: candidate
value: <who pays the cost of not knowing this; the deployment surface>
verifiability: <which exact-checkable channels/instruments measure it>
budget: <estimate + which GPU class>
window: <novelty decay — what release/paper pressure exists>
sketch: <2–5 sentences: question, design axes, expected milestones>
```

## Scoring (pick what to run next)

`economic exposure × verifiability × novelty window`, with a strong prior
for problems that reuse an existing instrument (marginal cost of a
question on an amortized instrument is ~$5–20 and a day).

## Sourcing

Until watchers are automated, refill the queue by scanning: engine
release notes (vLLM, SGLang, LMDeploy, TensorRT-LLM), their GitHub issues
(field reports of quality regressions are collision-free gold), new model
family releases, and arXiv (cs.CL/cs.LG inference). The KV-drift
collision scan (`jinminghe950/agentic-research`,
`research/kv-drift/collision-scan.md`) is the format reference for
scans.
