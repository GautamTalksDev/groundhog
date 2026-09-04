"""Verdict classes: emptiness must not render as a clean null."""

from __future__ import annotations

import unittest

from gh.rank import RankResult
from gh.render import (
    VERDICT_INSUFFICIENT,
    VERDICT_NO_HISTORY,
    VERDICT_NULL,
    VERDICT_PARTIAL,
    VERDICT_REPEATED,
    VERDICT_SELFCHECK_FAILED,
    build_report,
    classify_verdict,
    render_text,
)

from test_render import _cand


def _report(**kwargs):
    defaults = dict(
        days=30,
        min_runs=3,
        top=3,
        session_count=10,
        harness_statuses={"cursor": "found", "claude_code": "absent"},
        rank_result=RankResult(candidates=[]),
        skipped=[],
        malformed_lines=0,
        locations_checked=["/tmp/.cursor/projects", "/tmp/.claude/projects"],
        files_read=10,
        files_total=10,
        time_truncated=False,
    )
    defaults.update(kwargs)
    return build_report(**defaults)


class ClassifyVerdictTests(unittest.TestCase):
    def test_no_supported_history(self) -> None:
        verdict = classify_verdict(
            harness_statuses={
                "claude_code": "absent",
                "codex": "absent",
                "cursor": "absent",
            },
            session_count=0,
            min_runs=3,
            candidate_count=0,
            skipped=[],
            malformed_lines=0,
            time_truncated=False,
        )
        self.assertEqual(verdict, VERDICT_NO_HISTORY)

    def test_insufficient_history(self) -> None:
        verdict = classify_verdict(
            harness_statuses={"cursor": "found"},
            session_count=2,
            min_runs=3,
            candidate_count=0,
            skipped=[],
            malformed_lines=0,
            time_truncated=False,
        )
        self.assertEqual(verdict, VERDICT_INSUFFICIENT)

    def test_defensible_null(self) -> None:
        verdict = classify_verdict(
            harness_statuses={"cursor": "found"},
            session_count=10,
            min_runs=3,
            candidate_count=0,
            skipped=[],
            malformed_lines=0,
            time_truncated=False,
        )
        self.assertEqual(verdict, VERDICT_NULL)

    def test_repeated_work(self) -> None:
        verdict = classify_verdict(
            harness_statuses={"cursor": "found"},
            session_count=10,
            min_runs=3,
            candidate_count=2,
            skipped=[],
            malformed_lines=0,
            time_truncated=False,
        )
        self.assertEqual(verdict, VERDICT_REPEATED)

    def test_unreadable_file_forces_partial_even_with_clusters(self) -> None:
        verdict = classify_verdict(
            harness_statuses={"cursor": "found"},
            session_count=10,
            min_runs=3,
            candidate_count=3,
            skipped=[("/tmp/locked.jsonl", "unreadable: Permission denied")],
            malformed_lines=0,
            time_truncated=False,
        )
        self.assertEqual(verdict, VERDICT_PARTIAL)

    def test_malformed_forces_partial(self) -> None:
        verdict = classify_verdict(
            harness_statuses={"cursor": "found"},
            session_count=10,
            min_runs=3,
            candidate_count=0,
            skipped=[],
            malformed_lines=4,
            time_truncated=False,
        )
        self.assertEqual(verdict, VERDICT_PARTIAL)

    def test_truncated_forces_partial(self) -> None:
        verdict = classify_verdict(
            harness_statuses={"cursor": "found"},
            session_count=10,
            min_runs=3,
            candidate_count=0,
            skipped=[],
            malformed_lines=0,
            time_truncated=True,
        )
        self.assertEqual(verdict, VERDICT_PARTIAL)


class RenderVerdictTests(unittest.TestCase):
    def test_each_verdict_is_first_line_under_header(self) -> None:
        cases = [
            (
                VERDICT_NO_HISTORY,
                dict(
                    session_count=0,
                    harness_statuses={
                        "claude_code": "absent",
                        "codex": "absent",
                        "cursor": "absent",
                    },
                    files_read=0,
                    files_total=0,
                ),
            ),
            (
                VERDICT_INSUFFICIENT,
                dict(session_count=2, rank_result=RankResult(candidates=[])),
            ),
            (
                VERDICT_NULL,
                dict(session_count=10, rank_result=RankResult(candidates=[])),
            ),
            (
                VERDICT_REPEATED,
                dict(
                    session_count=10,
                    rank_result=RankResult(candidates=[_cand()]),
                ),
            ),
            (
                VERDICT_PARTIAL,
                dict(
                    session_count=10,
                    rank_result=RankResult(candidates=[_cand()]),
                    skipped=[("/tmp/x.jsonl", "unreadable: Permission denied")],
                ),
            ),
        ]
        for expected, kwargs in cases:
            with self.subTest(verdict=expected):
                text = render_text(_report(**kwargs))
                header, rest = text.split("\n", 1)
                self.assertTrue(header.startswith("GROUNDHOG"))
                body = rest.lstrip("\n")
                first = body.split("\n", 1)[0]
                self.assertEqual(first, expected)

    def test_partial_with_clusters_is_not_a_null(self) -> None:
        text = render_text(
            _report(
                rank_result=RankResult(candidates=[_cand()]),
                skipped=[("/tmp/x.jsonl", "unreadable: Permission denied")],
            )
        )
        self.assertIn(VERDICT_PARTIAL, text)
        self.assertNotIn(VERDICT_NULL, text)
        self.assertIn("COVERAGE", text)
        self.assertIn("files skipped", text)
        self.assertIn("Re-run the garak smoke report", text)

    def test_coverage_ledger_fields_present(self) -> None:
        text = render_text(
            _report(
                tool_calls=58578,
                sessions_with_tokens=0,
                date_range="2026-08-05 → 2026-09-04",
            )
        )
        for field in (
            "directories checked",
            "agents detected",
            "files discovered",
            "files parsed",
            "files skipped",
            "sessions analyzed",
            "tool calls analyzed",
            "date range covered",
            "sessions with token counts",
            "threshold used",
        ):
            self.assertIn(field, text)
        self.assertIn("58,578", text)


class SelfCheckVerdictTests(unittest.TestCase):
    def test_failed_selfcheck_is_the_verdict_and_keeps_findings(self) -> None:
        report = _report(
            rank_result=RankResult(candidates=[_cand()]),
            selfcheck_ok=False,
            selfcheck_passed=6,
            selfcheck_total=7,
            selfcheck_ms=18.0,
            selfcheck_headline=(
                "Self-check: FAILED (6/7) — "
                "THIS ANALYZER IS NOT BEHAVING AS BUILT"
            ),
            selfcheck_failures=[
                (
                    "cluster_three_sessions",
                    "expected a cluster of 3 distinct sessions, got 0",
                )
            ],
            selfcheck_coverage="6/7 failed · 18ms",
        )
        self.assertEqual(report.verdict, VERDICT_SELFCHECK_FAILED)
        self.assertEqual(report.would_have_been, VERDICT_REPEATED)
        text = render_text(report)
        self.assertTrue(text.startswith("Self-check: FAILED (6/7)"))
        self.assertIn(VERDICT_SELFCHECK_FAILED, text)
        self.assertIn("(would have been: REPEATED WORK FOUND)", text)
        self.assertIn("cluster_three_sessions:", text)
        self.assertIn("Re-run the garak smoke report", text)
        self.assertIn("6/7 failed · 18ms", text)


if __name__ == "__main__":
    unittest.main()
