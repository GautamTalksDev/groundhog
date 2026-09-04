#!/usr/bin/env python3
"""@04 intents — extract substantive user asks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import emit  # noqa: E402
from step_io import load_json_arg, write_artifact  # noqa: E402
from gh.intents import extract_intents  # noqa: E402
from gh.parse import Session, Turn  # noqa: E402


def _hydrate(payload: dict) -> list[Session]:
    sessions: list[Session] = []
    for s in payload.get("sessions") or []:
        turns = [
            Turn(
                role=t["role"],
                text=t.get("text") or "",
                timestamp=t.get("timestamp"),
                input_tokens=t.get("input_tokens"),
                output_tokens=t.get("output_tokens"),
                cache_read_tokens=t.get("cache_read_tokens"),
                model=t.get("model"),
            )
            for t in s.get("turns") or []
        ]
        sessions.append(
            Session(
                session_id=s["session_id"],
                harness=s["harness"],
                project=s["project"],
                started_at=s.get("started_at"),
                ended_at=s.get("ended_at"),
                turns=turns,
                parse_status=s.get("parse_status") or "ok",
            )
        )
    return sessions


def main(argv: list[str]) -> int:
    payload = load_json_arg(argv[1]) if len(argv) > 1 else {}
    out_path = argv[2] if len(argv) > 2 else None
    sessions = _hydrate(payload)
    projects = sorted({s.project for s in sessions if s.project})
    intents = extract_intents(sessions, projects=projects)
    out = {
        "meta": {
            "sources": payload.get("sources") or {},
            "skipped": payload.get("skipped") or [],
            "malformed_lines": payload.get("malformed_lines") or 0,
            "truncated": payload.get("truncated") or False,
            "files_read": payload.get("files_read") or 0,
            "files_total": payload.get("files_total") or 0,
            "session_count": len(payload.get("sessions") or []),
        },
        "intents": [
            {
                "session_id": i.session_id,
                "harness": i.harness,
                "project": i.project,
                "timestamp": i.timestamp,
                "raw_text": i.raw_text,
                "normalized": i.normalized,
                "session_turn_count": i.session_turn_count,
                "session_tokens": i.session_tokens,
                "session_input_tokens": i.session_input_tokens,
                "session_output_tokens": i.session_output_tokens,
                "session_cache_read_tokens": i.session_cache_read_tokens,
                "session_model": i.session_model,
                "session_text_chars": i.session_text_chars,
            }
            for i in intents
        ],
    }
    write_artifact(out_path, out)
    emit(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
