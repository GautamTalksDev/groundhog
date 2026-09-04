"""Shared helpers for Groundhog Play step scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def bootstrap_path() -> Path:
    """resources/python — parent of steps/."""
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def emit(obj) -> None:
    sys.stdout.write(json.dumps(obj, default=_json_default))
    sys.stdout.write("\n")


def read_stdin_json():
    raw = sys.stdin.read()
    if not raw.strip():
        return None
    return json.loads(raw)


def _json_default(obj):
    if isinstance(obj, set):
        return sorted(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)


def session_file_dict(sf) -> dict:
    return {
        "path": sf.path,
        "harness": sf.harness,
        "mtime": sf.mtime,
        "size_bytes": sf.size_bytes,
        "project": getattr(sf, "project", "") or "",
    }


def discovery_dict(result) -> dict:
    return {
        "files": [session_file_dict(f) for f in result.files],
        "sources": dict(result.sources),
    }
