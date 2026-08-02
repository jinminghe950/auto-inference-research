"""Pod-side control plane, served on the pod's exposed HTTP port.

Embedded base64 into the Runpod template's start command (see
make_start_cmd.py), so a fresh pod needs no credentials and no repo access:
the harness tarball is uploaded over Runpod's HTTPS proxy, runs are launched
and watched, and results are fetched back — all token-authenticated with a
per-pod random token passed via the KV_DRIFT_TOKEN env var.

Endpoints (all require header X-Auth-Token):
  GET  /health          -> ok
  GET  /status          -> {running, exit_code} of the last /run
  GET  /ls?path=rel     -> JSON dir listing under /workspace
  GET  /<rel-path>      -> raw file bytes from /workspace
  POST /upload          -> body = .tar.gz, extracted into /workspace
  POST /run             -> body = {"cmd": "..."} run via bash in /workspace,
                           output tee'd to /workspace/run.log
"""

import http.server
import io
import json
import os
import subprocess
import tarfile
import urllib.parse

TOKEN = os.environ["KV_DRIFT_TOKEN"]
ROOT = os.path.realpath(os.environ.get("KV_DRIFT_ROOT", "/workspace"))
PORT = int(os.environ.get("KV_DRIFT_PORT", "8000"))

proc = None


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _reply(self, code=200, body=b"", ctype="application/octet-stream"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._reply(code, json.dumps(obj), "application/json")

    def _authed(self):
        if self.headers.get("X-Auth-Token") != TOKEN:
            self._reply(403, "forbidden")
            return False
        return True

    def _safe_path(self, rel):
        p = os.path.realpath(os.path.join(ROOT, rel.lstrip("/")))
        return p if p == ROOT or p.startswith(ROOT + os.sep) else None

    def do_GET(self):
        if not self._authed():
            return
        url = urllib.parse.urlparse(self.path)
        if url.path == "/health":
            return self._reply(200, "ok", "text/plain")
        if url.path == "/status":
            code = proc.poll() if proc else None
            return self._json({"launched": proc is not None,
                               "running": proc is not None and code is None,
                               "exit_code": code})
        if url.path == "/ls":
            rel = urllib.parse.parse_qs(url.query).get("path", ["."])[0]
            p = self._safe_path(rel)
            if not p or not os.path.isdir(p):
                return self._json({"error": "not a directory"}, 404)
            entries = [{"name": e.name, "dir": e.is_dir(), "bytes": e.stat().st_size if e.is_file() else None}
                       for e in sorted(os.scandir(p), key=lambda e: e.name)]
            return self._json({"path": rel, "entries": entries})
        p = self._safe_path(urllib.parse.unquote(url.path))
        if not p or not os.path.isfile(p):
            return self._reply(404, "not found")
        with open(p, "rb") as f:
            return self._reply(200, f.read())

    def do_POST(self):
        global proc
        if not self._authed():
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        if self.path == "/upload":
            try:
                tf = tarfile.open(fileobj=io.BytesIO(body), mode="r:gz")
                try:
                    tf.extractall(ROOT, filter="data")
                except TypeError:  # python < 3.12 fallback
                    tf.extractall(ROOT)
            except Exception as e:
                return self._json({"error": str(e)}, 400)
            return self._json({"extracted": True})
        if self.path == "/run":
            if proc is not None and proc.poll() is None:
                return self._json({"error": "a run is already in progress"}, 409)
            try:
                cmd = json.loads(body)["cmd"]
            except Exception as e:
                return self._json({"error": str(e)}, 400)
            proc = subprocess.Popen(
                ["bash", "-lc", f"({cmd}) >{ROOT}/run.log 2>&1"],
                cwd=ROOT, start_new_session=True)
            return self._json({"started": True, "pid": proc.pid})
        return self._reply(404, "not found")

    def log_message(self, fmt, *args):  # keep pod logs quiet
        pass


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
