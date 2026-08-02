"""Emit the Runpod template start command that embeds control_plane.py.

Usage: python3 pod/make_start_cmd.py
Prints a single shell command suitable as the template's container start
command. Base64 avoids every quoting hazard in the template args string."""

import base64
from pathlib import Path

src = (Path(__file__).parent / "control_plane.py").read_bytes()
b64 = base64.b64encode(src).decode()
print(f"bash -c 'echo {b64} | base64 -d > /control_plane.py && exec python3 -u /control_plane.py'")
