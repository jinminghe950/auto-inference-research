"""Content-addressed manifest: every result and code file, sha256-hashed.

"No hash, no verdict" — downstream gate evaluation cites these hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(out_dir: Path, code_dir: Path, meta: dict) -> dict:
    files = []
    for base, label in ((out_dir, "results"), (code_dir, "code")):
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.name == "manifest.json" or "__pycache__" in p.parts:
                continue
            files.append({
                "kind": label,
                "path": str(p.relative_to(base.parent if label == "results" else code_dir.parent)),
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            })
    return {"meta": meta, "files": files}


def write_manifest(out_dir: Path, code_dir: Path, meta: dict) -> Path:
    manifest = build_manifest(out_dir, code_dir, meta)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def verify_manifest(out_dir: Path, code_dir: Path) -> list[str]:
    """Re-hash everything the manifest names; return a list of problems."""
    problems: list[str] = []
    path = out_dir / "manifest.json"
    if not path.is_file():
        return [f"manifest missing: {path}"]
    manifest = json.loads(path.read_text())
    listed = {f["path"]: f for f in manifest.get("files", [])}
    if not listed:
        problems.append("manifest lists no files")
    for rel, entry in listed.items():
        base = out_dir.parent if entry["kind"] == "results" else code_dir.parent
        p = base / rel
        if not p.is_file():
            problems.append(f"manifested file missing on disk: {rel}")
            continue
        actual = sha256_file(p)
        if actual != entry["sha256"]:
            problems.append(f"hash mismatch for {rel}: manifest {entry['sha256'][:12]}… actual {actual[:12]}…")
    # Completeness: every result file on disk must be manifested.
    on_disk = {
        str(p.relative_to(out_dir.parent))
        for p in out_dir.rglob("*")
        if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts
    }
    for rel in sorted(on_disk - {r for r, e in listed.items() if e["kind"] == "results"}):
        problems.append(f"result file not in manifest: {rel}")
    return problems
