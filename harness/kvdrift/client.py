"""Minimal stdlib client for an OpenAI-compatible /v1/chat/completions."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class ChatClient:
    def __init__(self, base_url: str, model: str, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict] | None, *,
             temperature: float = 0.0, seed: int = 0, max_tokens: int = 400,
             retries: int = 3) -> dict:
        """Returns {message, usage, latency_s}. Raises on persistent failure.
        Pass tools=None for a plain-text request (no tools/tool_choice keys)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        body = json.dumps(payload).encode()
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode())
                latency = time.monotonic() - t0
                choice = data["choices"][0]
                usage = data.get("usage") or {}
                details = usage.get("prompt_tokens_details") or {}
                return {
                    "message": choice["message"],
                    "finish_reason": choice.get("finish_reason"),
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "cached_tokens": details.get("cached_tokens", 0),
                    },
                    "latency_s": round(latency, 3),
                }
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read(500).decode(errors="replace")
                except Exception:  # noqa: BLE001
                    detail = "<unreadable body>"
                if 400 <= e.code < 500:  # deterministic rejection — retrying can't help
                    raise RuntimeError(f"chat request rejected: {e} body={detail}") from e
                last_err = RuntimeError(f"{e} body={detail}")
                if attempt < retries:
                    time.sleep(2.0 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, KeyError) as e:
                last_err = e
                if attempt < retries:
                    time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"chat request failed after {retries + 1} attempts: {last_err}")


def get_json(url: str, timeout: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_healthy(base_url: str, timeout_s: float, poll_s: float = 5.0) -> None:
    """Block until GET /health returns 200 (vLLM readiness)."""
    deadline = time.monotonic() + timeout_s
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{base_url.rstrip('/')}/health")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    return
        except Exception as e:  # noqa: BLE001 - any failure means "not up yet"
            last = e
        time.sleep(poll_s)
    raise TimeoutError(f"server at {base_url} not healthy after {timeout_s}s (last error: {last})")
