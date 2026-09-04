"""Tests for gh.render — stranger-facing copy and empty state."""

from __future__ import annotations

import json
import unittest

from gh.rank import Candidate, EvidenceItem, RankResult
from gh.render import (
    ProjectView,
    build_report,
    render_json,
    render_text,
    stability_sentence,
    _empty_repeat_message,
    _format_timespan,
)


def _cand(**kwargs) -> Candidate:
    return Candidate(
        cluster_id=kwargs.get("cluster_id", "c1"),
        label=kwargs.get(
            "label", "Re-run the garak smoke report and compare to baseline"
        ),
        score=kwargs.get("score", 0.8),
        frequency=kwargs.get("frequency", 0.9),
        cost_score=kwargs.get("cost_score", 0.7),
        stability=kwargs.get("stability", 1.0),
        run_count=kwargs.get("run_count", 3),
        distinct_sessions=kwargs.get("distinct_sessions", kwargs.get("run_count", 3)),
        usd=kwargs.get("usd", 0.32),
        cost_basis=kwargs.get("cost_basis", "measured"),
        recency_days=1.0,
        projects=set(kwargs.get("projects", ["solen-kernel"])),
        evidence=[
            EvidenceItem(
                raw_text="Re-run garak smoke report vs baseline",
                timestamp="2026-09-02T20:00:00Z",
                project="solen-kernel",
                session_id="s1",
            )
        ],
        session_ids=["s1", "s2", "s3"],
        first_seen=kwargs.get("first_seen", "2026-08-29T18:00:00Z"),
        last_seen=kwargs.get("last_seen", "2026-09-02T20:00:00Z"),
        input_tokens=8000,
        output_tokens=2500,
        cache_read_tokens=1500,
        priced=kwargs.get("priced", kwargs.get("cost_basis", "measured") == "measured"),
    )


class StabilityCopyTests(unittest.TestCase):
    def test_wording_bands(self) -> None:
        self.assertEqual(
            stability_sentence(1.0), "Solved the same way every time."
        )
        self.assertEqual(
            stability_sentence(0.6), "Went smoothly most times."
        )
        self.assertEqual(
            stability_sentence(0.2), "Varied a lot between runs."
        )


class RenderTextTests(unittest.TestCase):
    def test_structure_and_no_jargon(self) -> None:
        report = build_report(
            days=14,
            min_runs=3,
            top=3,
            session_count=14,
            harness_statuses={"claude_code": "found", "codex": "absent"},
            rank_result=RankResult(candidates=[_cand()]),
            skipped=[],
            malformed_lines=0,
        )
        text = render_text(report)
        self.assertIn("GROUNDHOG · 14 sessions · last 14 days · Claude Code", text)
        self.assertIn("YOU KEEP REDOING THIS", text)
        self.assertIn("WHERE YOUR TOKENS WENT", text)
        self.assertIn("NOT COUNTED", text)
        self.assertIn("Codex history not found on this machine", text)
        self.assertIn("Local only · read your session files", text)
        self.assertIn("Solved the same way every time.", text)
        self.assertIn("from your logs", text)
        for banned in (
            "cluster",
            "cohesion",
            "medoid",
            "TF-IDF",
            "tf-idf",
            "centroid",
            "frequency=",
            "stability=0",
        ):
            self.assertNotIn(banned, text)

    def test_empty_state(self) -> None:
        report = build_report(
            days=14,
            min_runs=3,
            top=3,
            session_count=2,
            harness_statuses={"claude_code": "found", "codex": "absent"},
            rank_result=RankResult(candidates=[]),
            skipped=[],
        )
        text = render_text(report)
        self.assertIn("YOU KEEP REDOING THIS", text)
        self.assertIn(
            "No chore repeated across 3+ separate sessions in the last 14 days.",
            text,
        )
        self.assertIn("Try --days 30 or --min-runs 2.", text)
        self.assertIn("NOT COUNTED", text)
        self.assertIn("WHERE YOUR TOKENS WENT", text)

    def test_empty_suggestion_days_14_min_runs_3(self) -> None:
        msg = _empty_repeat_message(14, 3)
        self.assertIn(
            "No chore repeated across 3+ separate sessions in the last 14 days.",
            msg,
        )
        self.assertIn("Try --days 30 or --min-runs 2.", msg)
        self.assertNotIn("several months", msg)

    def test_empty_suggestion_days_60_min_runs_3(self) -> None:
        msg = _empty_repeat_message(60, 3)
        self.assertIn(
            "No chore repeated across 3+ separate sessions in the last 60 days.",
            msg,
        )
        self.assertIn("several months of history", msg)
        self.assertNotIn("Try --days 30", msg)
        self.assertIn("Try --min-runs 2.", msg)

    def test_empty_suggestion_days_60_min_runs_2(self) -> None:
        msg = _empty_repeat_message(60, 2)
        self.assertIn(
            "No chore repeated across 2+ separate sessions in the last 60 days.",
            msg,
        )
        self.assertIn("several months of history", msg)
        self.assertNotIn("Try --", msg)

    def test_empty_state_wide_window_does_not_suggest_stale_knobs(self) -> None:
        report = build_report(
            days=60,
            min_runs=3,
            top=3,
            session_count=10,
            harness_statuses={"cursor": "found", "claude_code": "absent"},
            rank_result=RankResult(candidates=[]),
            skipped=[],
        )
        text = render_text(report)
        self.assertIn(
            "No chore repeated across 3+ separate sessions in the last 60 days.",
            text,
        )
        self.assertIn("several months of history", text)
        self.assertNotIn("Try --days 30", text)
        self.assertIn("Try --min-runs 2.", text)
        self.assertIn("(no project totals in this window)", text)

    def test_empty_candidates_still_show_session_project_totals(self) -> None:
        report = build_report(
            days=60,
            min_runs=3,
            top=3,
            session_count=178,
            harness_statuses={"cursor": "found", "claude_code": "absent"},
            rank_result=RankResult(candidates=[]),
            skipped=[],
            session_projects=[
                ProjectView(
                    project="GASKET",
                    tokens=12_000,
                    usd=0.42,
                    basis="estimated",
                    run_count=40,
                    input_tokens=12_000,
                    priced=False,
                ),
                ProjectView(
                    project="Lading",
                    tokens=8_000,
                    usd=0.21,
                    basis="estimated",
                    run_count=20,
                    input_tokens=8_000,
                    priced=False,
                ),
            ],
        )
        text = render_text(report)
        self.assertIn("WHERE YOUR TOKENS WENT", text)
        self.assertIn("GASKET", text)
        self.assertIn("Lading", text)
        self.assertNotIn("$0.42", text)
        self.assertNotIn("$0.21", text)
        self.assertNotIn("(no project totals in this window)", text)

    def test_nothing_skipped_when_clean(self) -> None:
        report = build_report(
            days=14,
            min_runs=3,
            top=3,
            session_count=3,
            harness_statuses={"claude_code": "found", "codex": "found"},
            rank_result=RankResult(candidates=[_cand()]),
            skipped=[],
        )
        text = render_text(report)
        self.assertIn("nothing skipped", text)

    def test_unpriced_candidate_omits_dollar_column(self) -> None:
        report = build_report(
            days=14,
            min_runs=3,
            top=3,
            session_count=5,
            harness_statuses={"cursor": "found"},
            rank_result=RankResult(
                candidates=[_cand(cost_basis="estimated", priced=False, usd=0.79)]
            ),
            skipped=[],
            sessions_without_model=5,
        )
        text = render_text(report)
        self.assertIn("tokens", text)
        self.assertNotIn("$0.79", text)
        self.assertIn("5 sessions had no model id and therefore no cost", text)

    def test_priced_candidate_prints_dollars(self) -> None:
        report = build_report(
            days=14,
            min_runs=3,
            top=3,
            session_count=5,
            harness_statuses={"claude_code": "found"},
            rank_result=RankResult(candidates=[_cand(priced=True, usd=0.32)]),
            skipped=[],
        )
        text = render_text(report)
        self.assertIn("$0.32", text)
        self.assertIn("from your logs", text)


class RenderJsonTests(unittest.TestCase):
    def test_mirrors_scores_and_sessions(self) -> None:
        report = build_report(
            days=14,
            min_runs=3,
            top=3,
            session_count=14,
            harness_statuses={"claude_code": "found", "codex": "absent"},
            rank_result=RankResult(candidates=[_cand()]),
            skipped=[],
        )
        payload = json.loads(render_json(report))
        self.assertEqual(payload["schema_version"], "1")
        self.assertEqual(payload["candidates"][0]["session_ids"], ["s1", "s2", "s3"])
        self.assertIn("frequency", payload["candidates"][0]["components"])
        self.assertTrue(payload["not_counted"])
        self.assertIn("Codex history not found", payload["not_counted"][0])


class SameDaySpanTests(unittest.TestCase):
    def test_format_timespan_same_and_range(self) -> None:
        self.assertEqual(
            _format_timespan(5, "2026-08-24", "2026-08-24"),
            "5 times on 2026-08-24",
        )
        self.assertEqual(
            _format_timespan(3, "2026-08-20", "2026-08-29"),
            "3 times · 2026-08-20 → 2026-08-29",
        )

    def test_render_uses_on_for_same_day(self) -> None:
        report = build_report(
            days=30,
            min_runs=3,
            top=3,
            session_count=5,
            harness_statuses={"cursor": "found", "claude_code": "absent"},
            rank_result=RankResult(
                candidates=[
                    _cand(
                        first_seen="2026-08-24T12:00:00Z",
                        last_seen="2026-08-24T18:00:00Z",
                        run_count=5,
                        label="Harden crates/gw-dom for reliability",
                    )
                ]
            ),
            skipped=[],
        )
        text = render_text(report)
        self.assertIn("5 times on 2026-08-24", text)
        self.assertNotIn("2026-08-24 → 2026-08-24", text)

    def test_timespan_uses_distinct_sessions_not_member_count(self) -> None:
        report = build_report(
            days=30,
            min_runs=3,
            top=3,
            session_count=5,
            harness_statuses={"cursor": "found", "claude_code": "absent"},
            rank_result=RankResult(
                candidates=[
                    _cand(
                        first_seen="2026-08-20T12:00:00Z",
                        last_seen="2026-08-29T18:00:00Z",
                        run_count=8,
                        distinct_sessions=3,
                        label="Harden crates/gw-dom for reliability",
                    )
                ]
            ),
            skipped=[],
        )
        text = render_text(report)
        self.assertIn("3 times · 2026-08-20 → 2026-08-29", text)
        self.assertNotIn("8 times", text)


if __name__ == "__main__":
    unittest.main()
