#!/usr/bin/env python3
"""@06 cost — dollar costs with labeled basis."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import emit  # noqa: E402
from step_io import load_json_arg, write_artifact  # noqa: E402
from gh.cluster import Cluster  # noqa: E402
from gh.cost import cost_breakdown_dict, cost_for_cluster, load_prices  # noqa: E402
from gh.intents import Intent  # noqa: E402


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
    prices_path = Path(__file__).resolve().parent.parent / "prices.json"
    prices = load_prices(prices_path)
    priced = []
    for item in payload.get("clusters") or []:
        cluster = _cluster(item)
        cost = cost_for_cluster(cluster, prices)
        priced.append({"cluster": item, "cost": cost_breakdown_dict(cost)})
    out = {"meta": payload.get("meta") or {}, "priced": priced}
    write_artifact(out_path, out)
    emit(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
