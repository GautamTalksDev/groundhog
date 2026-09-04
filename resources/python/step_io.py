#!/usr/bin/env python3
"""Shared IO: load JSON from path or literal, write artifact + stdout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_arg(value: str) -> Any:
    """Prefer file path; fall back to JSON literal."""
    path = Path(value)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def write_artifact(path: str | None, payload: Any) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload) + "\n", encoding="utf-8")
