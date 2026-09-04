#!/usr/bin/env python3
"""@08 report — stranger-facing text + JSON report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import emit  # noqa: E402
from step_io import load_json_arg, write_artifact  # noqa: E402
from gh.discover import checked_locations  # noqa: E402
from gh.rank import Candidate, EvidenceItem, RankResult  # noqa: E402
from gh.render import build_report, render_json, render_text  # noqa: E402


def main(argv: list[str]) -> int:
    # argv: report.py <rank_path> <days> <top> <min_runs> <redact> [out_path]
    payload = load_json_arg(argv[1]) if len(argv) > 1 else {}
    days = int(argv[2]) if len(argv) > 2 else 14
    top = int(argv[3]) if len(argv) > 3 else 3
    min_runs = int(argv[4]) if len(argv) > 4 else 3
    redact_raw = (argv[5] if len(argv) > 5 else "true").strip().lower()
    redact = redact_raw not in ("0", "false", "no", "off")
    out_path = argv[6] if len(argv) > 6 else None

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
            )
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
        files_read=int(meta.get("files_read") or 0),
        files_total=int(meta.get("files_total") or 0),
    )
    out = {"text": render_text(report), "json": json.loads(render_json(report))}
    write_artifact(out_path, out)
    emit(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
