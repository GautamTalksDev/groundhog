#!/usr/bin/env python3
"""@03 parse — join discoveries and normalize sessions."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import emit  # noqa: E402
from step_io import load_json_arg, write_artifact  # noqa: E402
from gh.discover import SessionFile  # noqa: E402
from gh.parse import parse_sessions  # noqa: E402

TIME_BUDGET = 20.0


def _files_from(payload: dict) -> list[SessionFile]:
    out: list[SessionFile] = []
    for item in payload.get("files") or []:
        out.append(
            SessionFile(
                path=item["path"],
                harness=item["harness"],
                mtime=float(item.get("mtime") or 0),
                size_bytes=int(item.get("size_bytes") or 0),
                project=item.get("project") or "",
            )
        )
    return out


def main(argv: list[str]) -> int:
    # argv: parse.py <claude_path> <codex_path> <cursor_path> [out_path]
    claude = load_json_arg(argv[1]) if len(argv) > 1 else {"files": [], "sources": {}}
    codex = load_json_arg(argv[2]) if len(argv) > 2 else {"files": [], "sources": {}}
    cursor = load_json_arg(argv[3]) if len(argv) > 3 else {"files": [], "sources": {}}
    out_path = argv[4] if len(argv) > 4 else None

    files = _files_from(claude) + _files_from(codex) + _files_from(cursor)
    sources = {}
    sources.update(claude.get("sources") or {})
    sources.update(codex.get("sources") or {})
    sources.update(cursor.get("sources") or {})

    deadline = time.monotonic() + TIME_BUDGET
    parsed = parse_sessions(files, deadline=deadline)

    sessions = []
    for s in parsed.sessions:
        sessions.append(
            {
                "session_id": s.session_id,
                "harness": s.harness,
                "project": s.project,
                "started_at": s.started_at,
                "ended_at": s.ended_at,
                "parse_status": s.parse_status,
                "turns": [
                    {
                        "role": t.role,
                        "text": t.text,
                        "timestamp": t.timestamp,
                        "input_tokens": t.input_tokens,
                        "output_tokens": t.output_tokens,
                        "cache_read_tokens": t.cache_read_tokens,
                        "model": t.model,
                    }
                    for t in s.turns
                ],
            }
        )

    payload = {
        "sources": sources,
        "sessions": sessions,
        "skipped": list(parsed.skipped),
        "malformed_lines": parsed.malformed_lines,
        "truncated": parsed.truncated,
        "files_read": parsed.files_read,
        "files_total": parsed.files_total,
    }
    write_artifact(out_path, payload)
    emit(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
