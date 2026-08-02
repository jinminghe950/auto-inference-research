# harness — the lab equipment

Battle-tested in the KV-drift case study
(`jinminghe950/agentic-research`, `research/kv-drift/`); ported intact —
the package keeps its `kvdrift` name to avoid churning tested code. Run
`python3 selftest.py` (offline, no GPU) before landing any change here:
it plants defects in every failure channel and proves each is caught.

## Domain-agnostic core (reuse for every instrument)

- `kvdrift/manifest.py` — sha256 content-addressed manifests + verifier
  ("no hash, no verdict")
- `kvdrift/main.py` — run driver + machine gates G0–G4 (`run` / `verify` /
  `onset` CLIs); pre-flight coherence probes; matrix-frozen thresholds
- `kvdrift/serve.py` — engine abstraction (vLLM + LMDeploy TurboMind;
  per-config `server_env`/`extra_args`; venv-aware PATH handling)
- `kvdrift/client.py` — stdlib OpenAI-compatible client (4xx terminal with
  body capture; tools optional)
- `pod/control_plane.py` — token-authenticated pod control plane (upload /
  run / status / fetch over the Runpod HTTPS proxy; no credentials on the
  pod); `pod/make_start_cmd.py` emits the template start command
- `sweep.sh` / `repro.sh` — entrypoints (per-engine venvs on local disk;
  see the gated-experiment skill for why)

## Instrument #1: tool-calling fidelity (task-specific)

- `kvdrift/warehouse.py` — deterministic warehouse simulator, 12 tools,
  validated execution, oracle-readable state
- `kvdrift/trajectory.py` — scripted user: seed-deterministic instruction
  text, state-resolved expected arguments, long-range recall probes
- `kvdrift/scoring.py` — separable failure channels: call_present /
  parse_valid / tool_correct / args_fidelity / exec_error / recall_hit /
  overcall
- `kvdrift/runner.py` — self-conditioned trajectories (model's own errors
  stay in the transcript)
- `kvdrift/onset.py` — copy-fidelity vs context length probe (near/far
  referent modes)
- `configs/smoke.json`, `configs/onset.json` — working example matrices
- The frozen analyzer pattern for this instrument: `analysis/analyze.py`
  (turn-binned channel rates, bootstrap CIs over seeds, H-contrasts)

New instruments: add alongside (new sim/oracle/scoring modules + a config
schema + selftest coverage), reusing the core unchanged. Prefer channels
that are exactly checkable; a metric needing a human judge is a design
smell in this domain.
