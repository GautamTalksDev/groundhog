#!/usr/bin/env python3
"""@08 report — stranger-facing text + JSON report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import emit  # noqa: E402
from step_io import load_json_arg, write_artifact  # noqa: E402
from gh.cost import (  # noqa: E402
    count_sessions_with_tokens,
    count_sessions_without_model,
    date_range_for_sessions,
    load_prices,
    project_costs_from_sessions,
)
from gh.discover import checked_locations  # noqa: E402
from gh.parse import Session, Turn  # noqa: E402
from gh.rank import Candidate, EvidenceItem, RankResult  # noqa: E402
from gh.render import (  # noqa: E402
    build_report,
    projects_from_session_costs,
    render_json,
    render_text,
)


def main(argv: list[str]) -> int:
    # argv: report.py <rank_path> <days> <top> <min_runs> <redact> [out_path]
    payload = load_json_arg(argv[1]) if len(argv) > 1 else {}
    days = int(argv[2]) if len(argv) > 2 else 14
    top = int(argv[3]) if len(argv) > 3 else 3
    min_runs = int(argv[4]) if len(argv) > 4 else 3
    redact_raw = (argv[5] if len(argv) > 5 else "true").strip().lower()
    redact = redact_raw not in ("0", "false", "no", "off")
    out_path = argv[6] if len(argv) > 6 else None
    parse_path = argv[7] if len(argv) > 7 else None

    meta = payload.get("meta") or {}
    candidates = []
    for c in payload.get("candidates") or []:
        candidates.append(
            Candidate(
                cluster_id=c["cluster_id"],
                label=c["label"],
                score=float(c.get("score") or 0),
                frequency=float(c.get("frequency") or 0),
                cost_score=float(c.get("cost_score") or 0),
                stability=float(c.get("stability") or 0),
                run_count=int(c.get("run_count") or 0),
                distinct_sessions=int(
                    c.get("distinct_sessions")
                    or len(c.get("session_ids") or [])
                    or c.get("run_count")
                    or 0
                ),
                usd=float(c.get("usd") or 0),
                cost_basis=c.get("cost_basis") or "unknown",
                recency_days=c.get("recency_days"),
                projects=set(c.get("projects") or []),
                evidence=[
                    EvidenceItem(
                        raw_text=ev["raw_text"],
                        timestamp=ev.get("timestamp"),
                        project=ev.get("project") or "",
                        session_id=ev.get("session_id") or "",
                    )
                    for ev in c.get("evidence") or []
                ],
                session_ids=list(c.get("session_ids") or []),
                first_seen=c.get("first_seen"),
                last_seen=c.get("last_seen"),
                input_tokens=int(c.get("input_tokens") or 0),
                output_tokens=int(c.get("output_tokens") or 0),
                cache_read_tokens=int(c.get("cache_read_tokens") or 0),
                priced=bool(c.get("priced")),
            )
        )

    session_projects = []
    sessions = []
    parse_payload: dict = {}
    if parse_path:
        parse_payload = load_json_arg(parse_path)
        sessions = _hydrate_sessions(parse_payload)
        if not candidates and sessions:
            prices = load_prices(
                Path(__file__).resolve().parent.parent / "prices.json"
            )
            session_projects = projects_from_session_costs(
                project_costs_from_sessions(sessions, prices)
            )

    report = build_report(
        days=days,
        min_runs=min_runs,
        top=top,
        session_count=int(meta.get("session_count") or 0),
        harness_statuses=meta.get("sources") or {},
        rank_result=RankResult(candidates=candidates),
        skipped=[tuple(x) for x in (meta.get("skipped") or [])],
        malformed_lines=int(meta.get("malformed_lines") or 0),
        locations_checked=checked_locations(),
        redact=redact,
        time_truncated=bool(meta.get("truncated")),
        files_read=int(
            meta.get("files_read")
            or parse_payload.get("files_read")
            or 0
        ),
        files_total=int(
            meta.get("files_total")
            or parse_payload.get("files_total")
            or 0
        ),
        session_projects=session_projects or None,
        tool_calls=int(
            meta.get("tool_calls")
            or parse_payload.get("tool_calls")
            or 0
        ),
        sessions_with_tokens=count_sessions_with_tokens(sessions),
        sessions_without_model=count_sessions_without_model(sessions),
        date_range=date_range_for_sessions(sessions),
    )
    out = {"text": render_text(report), "json": json.loads(render_json(report))}
    write_artifact(out_path, out)
    emit(out)
    return 0


def _hydrate_sessions(payload: dict) -> list[Session]:
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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
