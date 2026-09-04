#!/usr/bin/env python3
"""@05 cluster — group repeated intents."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import emit  # noqa: E402
from step_io import load_json_arg, write_artifact  # noqa: E402
from gh.cluster import cluster_intents  # noqa: E402
from gh.intents import Intent  # noqa: E402


def _hydrate(items: list) -> list[Intent]:
    return [
        Intent(
            session_id=i["session_id"],
            harness=i["harness"],
            project=i["project"],
            timestamp=i.get("timestamp"),
            raw_text=i["raw_text"],
            normalized=i["normalized"],
            session_turn_count=int(i.get("session_turn_count") or 0),
            session_tokens=i.get("session_tokens"),
            session_input_tokens=i.get("session_input_tokens"),
            session_output_tokens=i.get("session_output_tokens"),
            session_cache_read_tokens=i.get("session_cache_read_tokens"),
            session_model=i.get("session_model"),
            session_text_chars=int(i.get("session_text_chars") or 0),
        )
        for i in items
    ]


def main(argv: list[str]) -> int:
    # argv: cluster.py <intents_path> <min_runs> [out_path]
    payload = load_json_arg(argv[1]) if len(argv) > 1 else {}
    min_runs = int(argv[2]) if len(argv) > 2 else 3
    out_path = argv[3] if len(argv) > 3 else None
    clusters = cluster_intents(
        _hydrate(payload.get("intents") or []), min_runs=min_runs
    )
    out = {
        "meta": payload.get("meta") or {},
        "clusters": [
            {
                "id": c.id,
                "label": c.label,
                "projects": sorted(c.projects),
                "first_seen": c.first_seen,
                "last_seen": c.last_seen,
                "run_count": c.run_count,
                "cohesion": c.cohesion,
                "members": [
                    {
                        "session_id": m.session_id,
                        "harness": m.harness,
                        "project": m.project,
                        "timestamp": m.timestamp,
                        "raw_text": m.raw_text,
                        "normalized": m.normalized,
                        "session_turn_count": m.session_turn_count,
                        "session_tokens": m.session_tokens,
                        "session_input_tokens": m.session_input_tokens,
                        "session_output_tokens": m.session_output_tokens,
                        "session_cache_read_tokens": m.session_cache_read_tokens,
                        "session_model": m.session_model,
                        "session_text_chars": m.session_text_chars,
                    }
                    for m in c.members
                ],
            }
            for c in clusters
        ],
    }
    write_artifact(out_path, out)
    emit(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
