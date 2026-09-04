#!/usr/bin/env python3
"""@07 rank — score Play candidates."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import emit  # noqa: E402
from step_io import load_json_arg, write_artifact  # noqa: E402
from gh.cluster import Cluster  # noqa: E402
from gh.cost import CostBreakdown  # noqa: E402
from gh.intents import Intent  # noqa: E402
from gh.rank import score_candidates  # noqa: E402


def _cluster(item: dict) -> Cluster:
    members = [
        Intent(
            session_id=m["session_id"],
            harness=m["harness"],
            project=m["project"],
            timestamp=m.get("timestamp"),
            raw_text=m["raw_text"],
            normalized=m["normalized"],
            session_turn_count=int(m.get("session_turn_count") or 0),
            session_tokens=m.get("session_tokens"),
            session_input_tokens=m.get("session_input_tokens"),
            session_output_tokens=m.get("session_output_tokens"),
            session_cache_read_tokens=m.get("session_cache_read_tokens"),
            session_model=m.get("session_model"),
            session_text_chars=int(m.get("session_text_chars") or 0),
        )
        for m in item.get("members") or []
    ]
    return Cluster(
        id=item["id"],
        members=members,
        label=item["label"],
        projects=set(item.get("projects") or []),
        first_seen=item.get("first_seen"),
        last_seen=item.get("last_seen"),
        run_count=int(item.get("run_count") or len(members)),
        cohesion=float(item.get("cohesion") or 0),
    )


def main(argv: list[str]) -> int:
    payload = load_json_arg(argv[1]) if len(argv) > 1 else {}
    out_path = argv[2] if len(argv) > 2 else None
    clusters = []
    costs = []
    for row in payload.get("priced") or []:
        clusters.append(_cluster(row["cluster"]))
        c = row["cost"]
        costs.append(
            CostBreakdown(
                input_tokens=int(c.get("input_tokens") or 0),
                output_tokens=int(c.get("output_tokens") or 0),
                cache_read_tokens=int(c.get("cache_read_tokens") or 0),
                usd=float(c.get("usd") or 0),
                basis=c.get("basis") or "unknown",
                price_model=c.get("price_model") or "default",
                priced=bool(c.get("priced")),
            )
        )
    ranked = score_candidates(clusters, costs)
    out = {
        "meta": payload.get("meta") or {},
        "candidates": [
            {
                "cluster_id": cand.cluster_id,
                "label": cand.label,
                "score": cand.score,
                "frequency": cand.frequency,
                "cost_score": cand.cost_score,
                "stability": cand.stability,
                "run_count": cand.run_count,
                "distinct_sessions": cand.distinct_sessions,
                "usd": cand.usd,
                "cost_basis": cand.cost_basis,
                "recency_days": cand.recency_days,
                "projects": sorted(cand.projects),
                "session_ids": cand.session_ids,
                "first_seen": cand.first_seen,
                "last_seen": cand.last_seen,
                "input_tokens": cand.input_tokens,
                "output_tokens": cand.output_tokens,
                "cache_read_tokens": cand.cache_read_tokens,
                "priced": cand.priced,
                "evidence": [
                    {
                        "raw_text": ev.raw_text,
                        "timestamp": ev.timestamp,
                        "project": ev.project,
                        "session_id": ev.session_id,
                    }
                    for ev in cand.evidence
                ],
            }
            for cand in ranked.candidates
        ],
    }
    write_artifact(out_path, out)
    emit(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
