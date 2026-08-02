"""CLI: `run` executes a config matrix; `verify` is the machine gate.

Gate checks (pre-registered; G0/G4 added for milestone 3):
  G0  pre-flight numerics probe per server config — a plain-text echo and a
      single tool call must both come back coherent BEFORE any trajectory
      runs, so a broken serving config fails as INFRA, not as data;
  G1  every config completed trajectories × turns, no gaps;
  G2  the prefix-cache toggle demonstrably engaged: reuse configs show
      cached-token evidence (API usage, server metrics, or — engines that
      expose neither — a paired reuse-vs-recompute deep-turn latency gap),
      recompute configs show zero;
  G3  the sha256 manifest exists, covers every result file, and re-hashes
      cleanly (code files included, so the exact harness is content-addressed);
  G4  optional matrix-frozen thresholds (`gate_thresholds` in the matrix
      file), e.g. minimum fp16 action-correct — the instrument-sanity floor.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .client import ChatClient
from .manifest import sha256_file, verify_manifest, write_manifest
from .runner import run_trajectory
from .scoring import extract_first_call, summarize
from .serve import EngineServer
from .warehouse import tool_schemas

CODE_DIR = Path(__file__).resolve().parent
HARNESS_DIR = CODE_DIR.parent


def _capture(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e:  # noqa: BLE001 - metadata capture must not kill the run
        return f"<capture failed: {e}>"


def _engine_version(bin_env: str, default_bin: str, module: str) -> str:
    """Version via the engine venv's own python (bin path sibling)."""
    import os
    bin_path = Path(os.environ.get(bin_env, default_bin))
    python = bin_path.parent / "python" if bin_path.is_absolute() else Path(sys.executable)
    return _capture([str(python), "-c", f"import {module}; print({module}.__version__)"])


def _fetch_text(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.read().decode()
    except Exception as e:  # noqa: BLE001
        return f"<fetch failed: {e}>"


COHERENCE_SENTENCE = "The quick brown fox jumps over the lazy dog."


def _norm_text(s: str) -> str:
    return " ".join(s.casefold().split())


def preflight_probe(client: ChatClient, cfg_name: str) -> dict:
    """G0: gate on COHERENCE only — a common-token sentence echo and a
    structurally valid tool call. Exact-copy fidelity of rare identifiers is
    the *measurand* of this study (quantized KV corrupts it), so it is
    recorded here as probe data (`copy_*` fields) but must not gate: only
    token-salad / no-structure configs are INFRA.

    Amendment A2: `text_ok` (the only gating flag) is word-overlap
    coherence — at least 6 of the sentence's 9 distinct words must appear in
    the reply. A corrupted-but-on-task echo ("The sentence brown fox jumps
    at the lazy dog.") passes; token salad does not. Tool-call structure is
    recorded (`tool_ok`) but no longer gates: whether a config can emit a
    parseable call at all is an H3 outcome the trajectories measure."""
    nonce = f"KVDRIFT-{cfg_name.upper().replace('_', '-')}-7429"
    out: dict = {"nonce": nonce, "text_ok": False, "tool_ok": False,
                 "copy_text_ok": None, "copy_tool_exact": None}
    try:
        r = client.chat(
            [{"role": "system", "content": "You follow instructions exactly."},
             {"role": "user", "content": f"Reply with exactly this sentence and nothing else: {COHERENCE_SENTENCE}"}],
            tools=None, max_tokens=64)
        out["coherence_raw"] = r["message"].get("content") or ""
        words = set(_norm_text(COHERENCE_SENTENCE).rstrip(".").split())
        got = set(_norm_text(out["coherence_raw"]).replace(".", " ").split())
        out["coherence_word_overlap"] = len(words & got)
        out["text_ok"] = out["coherence_word_overlap"] >= 6
    except Exception as e:  # noqa: BLE001
        out["text_error"] = str(e)
    try:
        r = client.chat(
            [{"role": "system", "content": "You follow instructions exactly."},
             {"role": "user", "content": f"Reply with exactly this token and nothing else: {nonce}"}],
            tools=None, max_tokens=64)
        out["copy_text_raw"] = r["message"].get("content") or ""
        out["copy_text_ok"] = nonce in out["copy_text_raw"]
    except Exception as e:  # noqa: BLE001
        out["copy_text_error"] = str(e)
    try:
        r = client.chat(
            [{"role": "system", "content": "You are a warehouse assistant with tool access."},
             {"role": "user", "content": "Look up SKU-1234."}],
            tools=tool_schemas(), max_tokens=128)
        call = extract_first_call(r["message"])
        out["tool_raw"] = {"name": call["name"], "args": call["args"] if call["parse_valid"] else call["raw_arguments"]}
        out["tool_ok"] = bool(call["parse_valid"] and call["name"] == "lookup_item")
        out["copy_tool_exact"] = bool(
            out["tool_ok"] and "1234" in str((call["args"] or {}).get("sku", "")))
    except Exception as e:  # noqa: BLE001
        out["tool_error"] = str(e)
    return out


def cmd_run(args: argparse.Namespace) -> int:
    matrix_path = Path(args.matrix)
    matrix = json.loads(matrix_path.read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = matrix["model"]
    served = matrix.get("served_name", "kvdrift-model")
    port = int(matrix.get("port", 8001))
    n_traj = int(matrix["trajectories"])
    n_turns = int(matrix["turns"])
    seed_base = int(matrix.get("seed_base", 1000))
    max_tokens = int(matrix.get("max_tokens", 400))

    run_meta = {
        "matrix_file": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "matrix": matrix,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "vllm": _engine_version("VLLM_BIN", "vllm", "vllm"),
        "lmdeploy": _engine_version("LMDEPLOY_BIN", "lmdeploy", "lmdeploy"),
        "gpu": _capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                         "--format=csv,noheader"]),
    }

    for cfg in matrix["configs"]:
        name = cfg["name"]
        engine = cfg.get("engine", "vllm")
        cfg_dir = out_dir / f"config_{name}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        print(f"[kvdrift] === config {name}: engine={engine} kv={cfg['kv']} "
              f"prefix_caching={cfg['prefix_caching']} ===", flush=True)
        server = EngineServer(
            engine=engine, model=model, served_name=served, port=port,
            kv=cfg["kv"], prefix_caching=bool(cfg["prefix_caching"]),
            max_model_len=int(matrix.get("max_model_len", 8192)),
            gpu_memory_utilization=float(matrix.get("gpu_memory_utilization", 0.9)),
            log_path=out_dir / "logs" / f"{name}.server.log",
            extra_args=cfg.get("extra_args"),
            server_env=cfg.get("server_env"),
        )
        t0 = time.monotonic()
        server.start()
        print(f"[kvdrift] server healthy in {time.monotonic() - t0:.0f}s", flush=True)
        try:
            client = ChatClient(server.base_url, served)
            probe = preflight_probe(client, name)
            (cfg_dir / "probe.json").write_text(json.dumps(probe, indent=2, sort_keys=True))
            if not probe["text_ok"]:
                print(f"[kvdrift] {name}: PROBE FAILED "
                      f"(coherence word overlap {probe.get('coherence_word_overlap')}) — "
                      "skipping trajectories, config marked INFRA", flush=True)
                (cfg_dir / "metrics.prom").write_text(_fetch_text(f"{server.base_url}/metrics"))
                continue
            summaries = []
            for i in range(n_traj):
                seed = seed_base + i
                try:
                    records = run_trajectory(client, sim_seed=seed, n_turns=n_turns,
                                             max_tokens=max_tokens)
                    traj_path = cfg_dir / f"trajectory_{seed}.jsonl"
                    with open(traj_path, "w") as f:
                        for r in records:
                            f.write(json.dumps(r, sort_keys=True) + "\n")
                except Exception as e:  # noqa: BLE001 - one bad trajectory (or one
                    # flaky filesystem write) must not kill the run; a missing
                    # file fails G1 (fail-closed).
                    print(f"[kvdrift] {name} traj {seed}: ABORTED: {e}", flush=True)
                    continue
                s = {"seed": seed, **summarize(records)}
                summaries.append(s)
                print(f"[kvdrift] {name} traj {seed}: "
                      f"action_correct={s['action_correct_rate']} "
                      f"args_fidelity={s['args_fidelity_mean']} "
                      f"recall={s['recall_hit_rate']}", flush=True)
            (cfg_dir / "summary.json").write_text(json.dumps({
                "config": cfg, "trajectories": summaries}, indent=2, sort_keys=True))
            (cfg_dir / "metrics.prom").write_text(_fetch_text(f"{server.base_url}/metrics"))
        finally:
            server.stop()

    run_meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True))
    write_manifest(out_dir, CODE_DIR, {
        "harness": "kv-drift harness",
        "created_utc": run_meta["finished_utc"],
        "matrix_sha256": run_meta["matrix_sha256"],
        "gpu": run_meta["gpu"],
        "versions": {"python": run_meta["python"], "vllm": run_meta["vllm"],
                     "lmdeploy": run_meta["lmdeploy"]},
    })
    print(f"[kvdrift] run complete: {out_dir}", flush=True)
    return 0


def _prom_counter_total(text: str, name: str) -> float:
    """Sum a Prometheus counter across label sets, matching the metric name
    exactly (`name{...}` or `name value`) — a bare prefix match would also
    pick up `_created` timestamp gauges and poison zero-checks."""
    total = 0.0
    found = False
    for line in text.splitlines():
        if line.startswith(f"{name}{{") or line.startswith(f"{name} "):
            try:
                total += float(line.rsplit(None, 1)[-1])
                found = True
            except ValueError:
                pass
    return total if found else -1.0


PROM_HIT_NAMES = ["vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total",
                  "lmdeploy:prefix_cache_hits_total"]

# Paired-latency fallback (pre-registered): among turns >= DEEP_TURN, the
# reuse config's median request latency must be under LATENCY_RATIO_MAX of
# its recompute sibling's — at depth the recompute leg re-prefills the whole
# conversation, so a working cache shows up as a large TTFT gap.
DEEP_TURN = 25
LATENCY_RATIO_MAX = 0.6


def _load_config_data(out_dir: Path, matrix: dict) -> dict[str, dict]:
    n_traj = int(matrix["trajectories"])
    seed_base = int(matrix.get("seed_base", 1000))
    data: dict[str, dict] = {}
    for cfg in matrix["configs"]:
        name = cfg["name"]
        cfg_dir = out_dir / f"config_{name}"
        records: list[dict] = []
        missing: list[str] = []
        for i in range(n_traj):
            p = cfg_dir / f"trajectory_{seed_base + i}.jsonl"
            if not p.is_file():
                missing.append(p.name)
                continue
            records.append([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
        probe = None
        if (cfg_dir / "probe.json").is_file():
            probe = json.loads((cfg_dir / "probe.json").read_text())
        metrics = (cfg_dir / "metrics.prom").read_text() if (cfg_dir / "metrics.prom").is_file() else ""
        data[name] = {"cfg": cfg, "dir": cfg_dir, "trajectories": records,
                      "missing": missing, "probe": probe, "metrics": metrics}
    return data


def cmd_verify(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    problems: list[str] = []
    matrix = json.loads((out_dir / "run_meta.json").read_text())["matrix"]
    n_traj, n_turns = int(matrix["trajectories"]), int(matrix["turns"])
    data = _load_config_data(out_dir, matrix)

    def flat(name: str) -> list[dict]:
        return [r for traj in data[name]["trajectories"] for r in traj]

    def cached_tokens(name: str, min_turn: int = 0) -> list[int]:
        return [((r.get("usage") or {}).get("cached_tokens") or 0)
                for r in flat(name) if r.get("turn", 0) >= min_turn]

    def deep_latency(name: str) -> float | None:
        vals = [r["latency_s"] for r in flat(name) if r.get("turn", 0) >= DEEP_TURN]
        return statistics.median(vals) if vals else None

    preflight_required = bool(matrix.get("preflight"))  # matrices frozen before G0 existed skip it
    for name, d in data.items():
        cfg = d["cfg"]
        # G0
        if preflight_required:
            if d["probe"] is None:
                problems.append(f"G0 {name}: probe.json missing")
            elif not d["probe"].get("text_ok"):
                problems.append(f"G0 {name}: pre-flight coherence probe failed "
                                f"(word overlap {d['probe'].get('coherence_word_overlap')})")
        # G1
        for m in d["missing"]:
            problems.append(f"G1 {name}: missing {m}")
        for traj in d["trajectories"]:
            if len(traj) != n_turns:
                problems.append(f"G1 {name}: trajectory with {len(traj)} turns, expected {n_turns}")
        if not (d["dir"] / "summary.json").is_file() and not d["missing"]:
            problems.append(f"G1 {name}: missing summary.json")
        # G2
        prom_hits = max(_prom_counter_total(d["metrics"], n) for n in PROM_HIT_NAMES)
        late = cached_tokens(name, min_turn=2)
        if cfg["prefix_caching"]:
            usage_ev = bool(late) and statistics.median(late) > 0
            metrics_ev = prom_hits > 0
            latency_ev = False
            sibling = name.replace("-reuse", "-recompute")
            if not (usage_ev or metrics_ev) and sibling in data and sibling != name:
                a, b = deep_latency(name), deep_latency(sibling)
                latency_ev = a is not None and b is not None and b > 0 and (a / b) < LATENCY_RATIO_MAX
                if not latency_ev:
                    problems.append(
                        f"G2 {name}: no cache evidence in usage or metrics, and deep-turn latency "
                        f"ratio vs {sibling} is {f'{a/b:.2f}' if a and b else 'n/a'} "
                        f"(needs < {LATENCY_RATIO_MAX})")
            elif not (usage_ev or metrics_ev):
                problems.append(f"G2 {name}: reuse config shows no cache-hit evidence and no sibling to compare")
        else:
            if any(c > 0 for c in cached_tokens(name)):
                problems.append(f"G2 {name}: recompute config reported cached_tokens > 0")
            if prom_hits > 0:
                problems.append(f"G2 {name}: recompute config has prefix-cache hits in server metrics")

    # G3
    problems += [f"G3 {p}" for p in verify_manifest(out_dir, CODE_DIR)]

    # G4 — matrix-frozen thresholds
    thresholds = matrix.get("gate_thresholds") or {}
    if "fp16_action_correct_min" in thresholds:
        floor = float(thresholds["fp16_action_correct_min"])
        for name, d in data.items():
            if d["cfg"]["kv"] != "fp16" or not d["trajectories"]:
                continue
            rs = [r for r in flat(name) if r.get("expected_tool") and r.get("scoreable")]
            rate = sum(bool(r["action_correct"]) for r in rs) / len(rs) if rs else 0.0
            if rate < floor:
                problems.append(f"G4 {name}: fp16 action_correct {rate:.3f} below floor {floor}")

    if problems:
        for p in problems:
            print(f"[kvdrift][verify] FAIL: {p}")
        print(f"[kvdrift][verify] GATE FAIL ({len(problems)} problem(s))")
        return 1
    if preflight_required:
        print("[kvdrift][verify] G0 pre-flight probes: PASS")
    print("[kvdrift][verify] G1 completeness: PASS")
    print("[kvdrift][verify] G2 prefix-cache toggle evidence: PASS")
    print("[kvdrift][verify] G3 content-addressed manifest: PASS")
    if thresholds:
        print("[kvdrift][verify] G4 matrix-frozen thresholds: PASS")
    return 0


def cmd_onset(args: argparse.Namespace) -> int:
    from .onset import run_onset, summarize_onset

    matrix_path = Path(args.matrix)
    matrix = json.loads(matrix_path.read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lengths = [int(x) for x in matrix["lengths"]]
    reps = int(matrix["reps"])
    modes = list(matrix.get("modes", ["near", "far"]))
    expected_rows = len(lengths) * reps * len(modes)

    run_meta = {
        "matrix_file": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "matrix": matrix,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "vllm": _engine_version("VLLM_BIN", "vllm", "vllm"),
        "lmdeploy": _engine_version("LMDEPLOY_BIN", "lmdeploy", "lmdeploy"),
        "gpu": _capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                         "--format=csv,noheader"]),
    }
    shortfall = False
    for cfg in matrix["configs"]:
        name = cfg["name"]
        cfg_dir = out_dir / f"config_{name}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        print(f"[kvdrift] === onset {name}: engine={cfg.get('engine', 'vllm')} kv={cfg['kv']} ===", flush=True)
        server = EngineServer(
            engine=cfg.get("engine", "vllm"), model=matrix["model"],
            served_name=matrix.get("served_name", "kvdrift-model"),
            port=int(matrix.get("port", 8001)),
            kv=cfg["kv"], prefix_caching=bool(cfg.get("prefix_caching", False)),
            max_model_len=int(matrix.get("max_model_len", 16384)),
            gpu_memory_utilization=float(matrix.get("gpu_memory_utilization", 0.9)),
            log_path=out_dir / "logs" / f"{name}.server.log",
            extra_args=cfg.get("extra_args"), server_env=cfg.get("server_env"))
        server.start()
        try:
            client = ChatClient(server.base_url, matrix.get("served_name", "kvdrift-model"))
            records = run_onset(client, lengths=lengths, reps=reps, modes=modes,
                                seed_base=int(matrix.get("seed_base", 5000)))
            with open(cfg_dir / "onset.jsonl", "w") as f:
                for r in records:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
            summary = summarize_onset(records)
            (cfg_dir / "summary.json").write_text(json.dumps(
                {"config": cfg, "by_length": summary}, indent=2, sort_keys=True))
            for row in summary:
                print(f"[kvdrift] {name} {row['mode']}@{row['approx_tokens']}: "
                      f"sku_exact={row['sku_exact_rate']} call={row['call_present_rate']}", flush=True)
            if len(records) != expected_rows:
                print(f"[kvdrift] {name}: SHORTFALL {len(records)}/{expected_rows} rows", flush=True)
                shortfall = True
        finally:
            server.stop()

    run_meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True))
    write_manifest(out_dir, CODE_DIR, {
        "harness": "kv-drift onset probe",
        "created_utc": run_meta["finished_utc"],
        "matrix_sha256": run_meta["matrix_sha256"],
        "gpu": run_meta["gpu"],
        "versions": {"python": run_meta["python"], "vllm": run_meta["vllm"],
                     "lmdeploy": run_meta["lmdeploy"]},
    })
    if shortfall:
        print("[kvdrift] ONSET: INCOMPLETE", flush=True)
        return 1
    print(f"[kvdrift] onset complete: {out_dir}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="kvdrift")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="execute a config matrix")
    p_run.add_argument("--matrix", required=True)
    p_run.add_argument("--out", required=True)
    p_run.set_defaults(fn=cmd_run)
    p_ver = sub.add_parser("verify", help="machine gate over a finished run")
    p_ver.add_argument("--out", required=True)
    p_ver.set_defaults(fn=cmd_verify)
    p_on = sub.add_parser("onset", help="context-onset copy-fidelity probe")
    p_on.add_argument("--matrix", required=True)
    p_on.add_argument("--out", required=True)
    p_on.set_defaults(fn=cmd_onset)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
