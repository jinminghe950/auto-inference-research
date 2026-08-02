"""Server lifecycle for one KV-precision × prefix-cache configuration.

Two engines, one interface:
  - vllm      — fp16 / fp8_e5m2 / fp8_e4m3 KV (`--kv-cache-dtype`)
  - lmdeploy  — fp16 / int8 / int4 KV (TurboMind `--quant-policy` 0/8/4)

vLLM 0.26 pins torch 2.11 while LMDeploy 0.15 caps torch at 2.10, so the two
engines cannot share a python env. The harness itself is stdlib-only and runs
on the system python; each engine's CLI binary is resolved via VLLM_BIN /
LMDEPLOY_BIN env vars pointing into per-engine venvs.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from .client import wait_healthy

VLLM_KV = {"fp16": "auto", "fp8_e5m2": "fp8_e5m2", "fp8_e4m3": "fp8_e4m3"}
LMDEPLOY_QP = {"fp16": 0, "int8": 8, "int4": 4}


def build_cmd(*, engine: str, model: str, served_name: str, port: int,
              kv: str, prefix_caching: bool, max_model_len: int,
              gpu_memory_utilization: float,
              extra_args: list[str] | None = None) -> list[str]:
    if engine == "vllm":
        if kv not in VLLM_KV:
            raise ValueError(f"vllm engine does not support kv={kv!r} (has {sorted(VLLM_KV)})")
        return [
            os.environ.get("VLLM_BIN", "vllm"), "serve", model,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--served-model-name", served_name,
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--kv-cache-dtype", VLLM_KV[kv],
            "--seed", "0",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching",
        ] + list(extra_args or [])
    if engine == "lmdeploy":
        if kv not in LMDEPLOY_QP:
            raise ValueError(f"lmdeploy engine does not support kv={kv!r} (has {sorted(LMDEPLOY_QP)})")
        cmd = [
            os.environ.get("LMDEPLOY_BIN", "lmdeploy"), "serve", "api_server", model,
            "--server-name", "127.0.0.1",
            "--server-port", str(port),
            "--model-name", served_name,
            "--session-len", str(max_model_len),
            "--cache-max-entry-count", str(gpu_memory_utilization * 0.5),
            "--quant-policy", str(LMDEPLOY_QP[kv]),
            "--tool-call-parser", "qwen",
        ]
        if prefix_caching:
            cmd.append("--enable-prefix-caching")
        return cmd + list(extra_args or [])
    raise ValueError(f"unknown engine {engine!r}")


class EngineServer:
    def __init__(self, *, engine: str, model: str, served_name: str, port: int,
                 kv: str, prefix_caching: bool,
                 max_model_len: int, gpu_memory_utilization: float,
                 log_path: Path, startup_timeout_s: float = 2400.0,
                 extra_args: list[str] | None = None,
                 server_env: dict[str, str] | None = None):
        self.engine = engine
        self.base_url = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        self.startup_timeout_s = startup_timeout_s
        self.cmd = build_cmd(
            engine=engine, model=model, served_name=served_name, port=port,
            kv=kv, prefix_caching=prefix_caching, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization, extra_args=extra_args)
        self.server_env = dict(server_env or {})
        # Engines launched from a venv by absolute path still expect their
        # venv's bin/ on PATH (torch shells out to `ninja` there, etc.).
        self.env = os.environ.copy()
        bin_dir = os.path.dirname(self.cmd[0])
        if bin_dir:
            self.env["PATH"] = f"{bin_dir}:{self.env.get('PATH', '')}"
            self.env["VIRTUAL_ENV"] = os.path.dirname(bin_dir)
        self.env.update(self.server_env)
        self.proc: subprocess.Popen | None = None
        self._log_file = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=self.env,
            start_new_session=True,  # own process group, so we can kill children
        )
        try:
            deadline = time.monotonic() + self.startup_timeout_s
            while True:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"{self.engine} exited during startup (code {self.proc.returncode}); see {self.log_path}"
                    )
                try:
                    wait_healthy(self.base_url, timeout_s=15.0, poll_s=3.0)
                    return
                except TimeoutError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"{self.engine} not healthy after {self.startup_timeout_s}s; see {self.log_path}"
                        ) from None
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self.proc.wait(timeout=30)
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        # Give the port and GPU memory a moment to be released.
        time.sleep(5)
