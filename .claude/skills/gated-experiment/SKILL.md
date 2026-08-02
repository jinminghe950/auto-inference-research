---
name: gated-experiment
description: Execute one pre-registered GPU experiment milestone mechanically — provision a Runpod executor with the token-auth control plane, run a frozen matrix via the repo harness, monitor, collect and locally re-verify hash-manifested evidence, land it in a commit, terminate the pod. Use when a milestone's pre-registration is already committed and the task is to run it; never use this to design or amend a matrix.
---

# Gated experiment executor (mechanical milestone run)

You are the *executor*, not the scientist. The matrix, gates, and analysis
are frozen in an already-committed pre-registration. Do not edit them. If
anything requires changing a frozen file, STOP and report — that is an
amendment decision, outside this skill.

Inputs you must have (from the driver/pre-registration): the frozen matrix
path (e.g. `harness/configs/<matrix>.json (or a problem-local frozen matrix under research/<slug>/)`), target GPU
type + host CUDA requirement, output run name (e.g. `m6-sweep`), budget
stop-loss for the run.

## Procedure

1. **Token + pod.** Generate `secrets.token_hex(24)`; print it fully and
   write it to the scratchpad token file in the same step (a mismatch here
   cost a debugging cycle once). Create the pod from the standing Runpod template
   `kv-drift-harness` (id `9yc1o57ktn`, provisioned during the case study; the control plane is domain-agnostic despite the name) (create a new template via
   `harness/pod/make_start_cmd.py` if missing) with env
   `HF_HOME=/workspace/hf, KV_DRIFT_ROOT=/workspace, KV_DRIFT_PORT=8000,
   KV_DRIFT_TOKEN=<token>`. GPU per the pre-registration; SECURE cloud.
2. **Host validation.** Poll `https://<podId>-8000.proxy.runpod.net/health`
   (header `X-Auth-Token`) until 200, then run `nvidia-smi | head -4` via
   `POST /run` and parse the CUDA version. If it does not meet the
   pre-registration's requirement: delete the pod and recreate (host
   lottery); do not proceed on a non-conforming host.
3. **Upload.** `tar -czf harness.tgz --exclude='__pycache__' harness` from the repo root, `POST /upload`.
4. **Launch.** `POST /run` with
   `cd harness && VENVS=/venvs MATRIX=<matrix> OUT_DIR=/tmp/<run> ./sweep.sh; echo EXIT=$?; cp -r /tmp/<run> /workspace/harness/results/<run> && echo COPY_OK`.
   Venvs on **local disk** (`/venvs`) and results written to local disk
   then copied — network-volume small-file writes have failed with
   `Errno 5` on real hosts. Weights stay on the volume (`HF_HOME`).
5. **Monitor, don't poll by hand.** Arm a Monitor that tails `/run.log`
   for `=== config|GATE|PROBE|ABORTED|Traceback|EXIT|COPY_OK` and exits on
   `"running": false`. Monitors cap at 30 min — re-arm from the current
   line count on timeout. While waiting, do nothing destructive.
6. **On failure**: fetch the specific server log
   (`GET harness/results/<run>/logs/<config>.server.log`), diagnose, and
   report. Known classes: engine wheels vs host CUDA (vLLM ≥0.20 needs
   CUDA ≥13), venv bin dir missing from PATH (handled in `serve.py`),
   MooseFS `Errno 5` (results to local disk, above), context overflow
   (frozen `max_tokens`/`max_model_len` — amendment territory, STOP).
7. **Collect + re-verify locally.** Fetch `manifest.json`, then every
   `kind: results` file it lists, into `research/<slug>/runs/<run>/`;
   then run `python3 -m kvdrift.main verify --out ../research/<slug>/runs/<run>` from
   `harness/` — the verdict that counts is computed on the collected
   copies, never trusted from the pod. Also fetch `/run.log` as
   operational evidence.
8. **Terminate the pod** as soon as evidence is verified (ephemeral
   executor, never reused). Confirm with a pod listing.
9. **Land.** Run the frozen analyzer if the pre-registration names one;
   sweep for secrets (`grep` for the pod token and common credential
   patterns) — then commit evidence + analysis with a message stating the
   gate verdict, and push. A failed gate is landed and reported exactly
   like a passed one (fail-closed, evidence preserved).
10. **Report**: gate verdict per check, cost (pod lifetime × rate vs the
    stop-loss), and any anomalies for the audit panel.

Track the budget: if projected pod time exceeds the run's stop-loss,
terminate and report `attention` rather than continuing.
