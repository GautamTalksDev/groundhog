"""Runtime self-check: bundled cases through the real analyzer."""

from __future__ import annotations

import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

from gh.cluster import Cluster
from gh.cost import CostBreakdown
from gh.discover import DiscoveryResult, SessionFile
from gh.intents import Intent
from gh.parse import ParseResult, Session, Turn
from gh.rank import RankResult
from gh.redact import _SECRET_PATTERNS
from gh.render import (
    VERDICT_NULL,
    VERDICT_PARTIAL,
    VERDICT_REPEATED,
    VERDICT_SELFCHECK_FAILED,
    build_report,
    render_text,
)
from gh.selfcheck import (
    CASE_NAMES,
    _SELFCHECK_MS_BUDGET,
    CaseResult,
    SelfCheckResult,
    fixture_blob,
    report_kwargs,
    run_selfcheck,
)

from test_render import _cand


def _cluster(members: list[Intent]) -> Cluster:
    return Cluster(
        id="c1",
        members=members,
        label=members[0].raw_text[:70],
        projects={m.project for m in members},
        first_seen="2026-09-01T12:00:00Z",
        last_seen="2026-09-01T12:00:00Z",
        run_count=len(members),
        cohesion=0.9,
    )


def _session(text: str = "hello") -> Session:
    return Session(
        session_id="s1",
        harness="claude_code",
        project="demo",
        started_at="2026-09-01T12:00:00Z",
        ended_at="2026-09-01T12:01:00Z",
        turns=[Turn("user", text, "2026-09-01T12:00:00Z", None, None, None, None)],
        parse_status="ok",
    )


def _failed_result(name: str, detail: str) -> SelfCheckResult:
    cases = [
        CaseResult(n, n != name, "ok" if n != name else detail)
        for n in CASE_NAMES
    ]
    return SelfCheckResult(
        passed=sum(1 for c in cases if c.passed),
        total=len(cases),
        elapsed_ms=12.0,
        cases=cases,
    )


def _report_with_selfcheck(selfcheck: SelfCheckResult, **kwargs):
    defaults = dict(
        days=30,
        min_runs=3,
        top=3,
        session_count=10,
        harness_statuses={"cursor": "found", "claude_code": "absent"},
        rank_result=RankResult(candidates=[_cand()]),
        skipped=[],
        malformed_lines=0,
        locations_checked=["/tmp/.cursor/projects"],
        files_read=10,
        files_total=10,
        **report_kwargs(selfcheck),
    )
    defaults.update(kwargs)
    return build_report(**defaults)


class SelfCheckPassTests(unittest.TestCase):
    def test_all_bundled_cases_pass(self) -> None:
        result = run_selfcheck()
        failed = [(c.name, c.detail) for c in result.failures]
        self.assertTrue(result.ok, failed)
        self.assertEqual(result.total, len(CASE_NAMES))
        self.assertEqual(result.passed, result.total)
        self.assertEqual(tuple(c.name for c in result.cases), CASE_NAMES)

    def test_runtime_under_budget(self) -> None:
        result = run_selfcheck()
        self.assertLess(
            result.elapsed_ms,
            _SELFCHECK_MS_BUDGET,
            f"self-check took {result.elapsed_ms:.1f}ms",
        )

    def test_headline_on_success(self) -> None:
        result = run_selfcheck()
        self.assertEqual(
            result.headline(),
            f"Self-check: PASSED ({result.total}/{result.total} "
            "bundled analyzer cases)",
        )


class SelfCheckFixtureSafetyTests(unittest.TestCase):
    def test_no_credential_shaped_fixture_text(self) -> None:
        blob = fixture_blob()
        for pattern in _SECRET_PATTERNS:
            match = pattern.search(blob)
            self.assertIsNone(
                match,
                "credential-shaped fixture text matched "
                f"{pattern.pattern!r}: {(match.group(0) if match else '')!r}",
            )

    def test_fixtures_are_not_on_a_discovery_root(self) -> None:
        here = Path(__file__).resolve()
        selfcheck_mod = (
            here.parent.parent / "gh" / "selfcheck.py"
        ).resolve()
        self.assertTrue(selfcheck_mod.is_file())
        text = str(selfcheck_mod)
        for fragment in (
            "/.claude/projects",
            "/.codex/sessions",
            "/.codex/history",
            "/.cursor/projects",
        ):
            self.assertNotIn(fragment, text)
        # Bundled cases write under tempfile, never $HOME harness trees.
        source = selfcheck_mod.read_text(encoding="utf-8")
        self.assertIn("TemporaryDirectory", source)
        self.assertNotIn("Path.home()", source)
        self.assertNotIn("discover_sessions(", source)


class SelfCheckRefusalTests(unittest.TestCase):
    def test_failed_selfcheck_forces_refusal_verdict(self) -> None:
        result = _failed_result(
            "cluster_three_sessions",
            "expected a cluster of 3 distinct sessions, got 0 cluster(s)",
        )
        report = _report_with_selfcheck(result)
        self.assertEqual(report.verdict, VERDICT_SELFCHECK_FAILED)
        self.assertEqual(report.would_have_been, VERDICT_REPEATED)
        text = render_text(report)
        first = text.split("\n", 1)[0]
        self.assertTrue(
            first.startswith("Self-check: FAILED"),
            first,
        )
        self.assertIn(VERDICT_SELFCHECK_FAILED, text)
        self.assertIn("(would have been: REPEATED WORK FOUND)", text)
        self.assertIn("cluster_three_sessions:", text)
        self.assertIn("got 0 cluster(s)", text)
        # Findings are still shown. A failed check is not a clean null.
        self.assertIn("Re-run the garak smoke report", text)
        self.assertIn("YOU KEEP REDOING THIS", text)
        header_block = text.split("YOU KEEP REDOING THIS", 1)[0]
        self.assertNotIn(VERDICT_NULL, header_block)
        self.assertNotIn(
            "\n" + VERDICT_REPEATED + "\n",
            text.split("(would have been", 1)[0],
        )

    def test_failed_selfcheck_does_not_look_like_partial_clean(self) -> None:
        result = _failed_result(
            "unreadable_dir_partial",
            "expected PARTIAL, got DEFENSIBLE NULL",
        )
        report = _report_with_selfcheck(
            result,
            rank_result=RankResult(candidates=[]),
            skipped=[("/tmp/x", "unreadable directory: Permission denied")],
        )
        self.assertEqual(report.verdict, VERDICT_SELFCHECK_FAILED)
        self.assertEqual(report.would_have_been, VERDICT_PARTIAL)
        text = render_text(report)
        self.assertIn("FINDINGS BELOW CANNOT BE RELIED ON", text)
        self.assertIn("self-check", text.lower())

    def test_coverage_ledger_includes_selfcheck_cost(self) -> None:
        result = run_selfcheck()
        report = _report_with_selfcheck(result)
        text = render_text(report)
        self.assertIn("self-check", text)
        self.assertIn(f"{result.passed}/{result.total}", text)
        self.assertRegex(text, r"self-check\s+\d+/\d+ passed · \d+ms")


class SelfCheckEachCaseCanFailTests(unittest.TestCase):
    """Mutate the real analyzer until each named case fires, then restore."""

    def test_cluster_three_sessions_fails_when_cluster_returns_empty(self) -> None:
        with mock.patch("gh.selfcheck.cluster_intents", return_value=[]):
            result = run_selfcheck()
        case = next(c for c in result.cases if c.name == "cluster_three_sessions")
        self.assertFalse(case.passed, case.detail)
        self.assertIn("got 0 cluster", case.detail)

    def test_no_cluster_one_session_fails_when_same_session_clusters(self) -> None:
        def _fake_cluster(intents, min_runs=3):
            if not intents:
                return []
            sessions = {m.session_id for m in intents}
            if len(sessions) == 1 and len(intents) >= 3:
                fake = _cluster(list(intents)[:3])
                fake.distinct_sessions = 3
                return [fake]
            return cluster_intents_original(intents, min_runs=min_runs)

        from gh.selfcheck import cluster_intents as cluster_intents_original

        with mock.patch("gh.selfcheck.cluster_intents", side_effect=_fake_cluster):
            result = run_selfcheck()
        case = next(c for c in result.cases if c.name == "no_cluster_one_session")
        self.assertFalse(case.passed, case.detail)
        self.assertIn("got 1 cluster", case.detail)

    def test_malformed_jsonl_fails_when_parse_drops_the_count(self) -> None:
        real = __import__("gh.parse", fromlist=["parse_sessions"]).parse_sessions

        def _silent(files, deadline=None):
            parsed = real(files, deadline=deadline)
            parsed.malformed_lines = 0
            return parsed

        with mock.patch("gh.selfcheck.parse_sessions", side_effect=_silent):
            result = run_selfcheck()
        case = next(c for c in result.cases if c.name == "malformed_jsonl_skipped")
        self.assertFalse(case.passed, case.detail)
        self.assertIn("malformed_lines", case.detail)

    def test_unreadable_dir_fails_when_discover_hides_it(self) -> None:
        clean = DiscoveryResult(
            files=[],
            sources={"claude_code": "found"},
            skipped=[],
        )
        with mock.patch("gh.selfcheck.discover_harness", return_value=clean):
            result = run_selfcheck()
        case = next(c for c in result.cases if c.name == "unreadable_dir_partial")
        self.assertFalse(case.passed, case.detail)
        self.assertIn("PARTIAL", case.detail)

    def test_no_model_id_fails_when_cost_claims_priced(self) -> None:
        fake = CostBreakdown(
            input_tokens=5000,
            output_tokens=200,
            cache_read_tokens=0,
            usd=0.15,
            basis="measured",
            priced=True,
        )
        with mock.patch("gh.selfcheck.cost_for_session", return_value=fake):
            result = run_selfcheck()
        case = next(c for c in result.cases if c.name == "no_model_id_no_dollars")
        self.assertFalse(case.passed, case.detail)
        self.assertIn("priced=True", case.detail)

    def test_user_query_fails_when_wrapper_is_left_in_place(self) -> None:
        wrapped = (
            "<user_query>Re-run the garak smoke report and compare it "
            "to baseline.report.jsonl</user_query>"
        )
        parsed = ParseResult(
            sessions=[_session(wrapped)],
            malformed_lines=0,
        )
        with mock.patch("gh.selfcheck.parse_sessions", return_value=parsed):
            with mock.patch("gh.selfcheck.extract_intents", return_value=[]):
                result = run_selfcheck()
        case = next(c for c in result.cases if c.name == "user_query_unwrapped")
        self.assertFalse(case.passed, case.detail)
        self.assertIn("wrapper still present", case.detail)

    def test_symlink_fails_when_discover_keeps_the_escape(self) -> None:
        real = __import__(
            "gh.discover", fromlist=["discover_harness"]
        ).discover_harness

        def _keep_escape(harness, days, home=None):
            result = real(harness, days, home=home)
            if harness != "cursor" or home is None:
                return result
            leak = SessionFile(
                path=str(Path(home) / "outside.jsonl"),
                harness="cursor",
                mtime=0.0,
                size_bytes=1,
            )
            result.files = [leak]
            result.skipped = []
            return result

        with mock.patch("gh.selfcheck.discover_harness", side_effect=_keep_escape):
            result = run_selfcheck()
        case = next(c for c in result.cases if c.name == "symlink_outside_refused")
        self.assertFalse(case.passed, case.detail)

    def test_mutated_cluster_module_turns_the_scan_red(self) -> None:
        """A real cluster.py change must not render as a clean result."""
        from groundhog import main

        with mock.patch("gh.selfcheck.cluster_intents", return_value=[]):
            buf = StringIO()
            with tempfile.TemporaryDirectory() as tmp:
                with mock.patch.dict(os.environ, {"HOME": tmp}):
                    with mock.patch("sys.stdout", buf):
                        code = main(["--days", "14"])
            self.assertEqual(code, 0)
            text = buf.getvalue()
        self.assertTrue(
            text.startswith("Self-check: FAILED"),
            text.split("\n", 1)[0],
        )
        self.assertIn(VERDICT_SELFCHECK_FAILED, text)
        self.assertIn("cluster_three_sessions", text)
        self.assertIn("FINDINGS BELOW CANNOT BE RELIED ON", text)
        self.assertNotIn("\n" + VERDICT_NULL + "\n", text.split("would have been", 1)[0])


class SelfCheckUsesRealAnalyzerTests(unittest.TestCase):
    def test_cluster_case_calls_cluster_intents(self) -> None:
        with mock.patch(
            "gh.selfcheck.cluster_intents",
            wraps=__import__("gh.cluster", fromlist=["cluster_intents"]).cluster_intents,
        ) as wrapped:
            run_selfcheck()
        self.assertGreaterEqual(wrapped.call_count, 2)

    def test_cost_case_calls_cost_for_session(self) -> None:
        with mock.patch(
            "gh.selfcheck.cost_for_session",
            wraps=__import__("gh.cost", fromlist=["cost_for_session"]).cost_for_session,
        ) as wrapped:
            run_selfcheck()
        self.assertGreaterEqual(wrapped.call_count, 1)

    def test_discover_cases_call_discover_harness(self) -> None:
        with mock.patch(
            "gh.selfcheck.discover_harness",
            wraps=__import__(
                "gh.discover", fromlist=["discover_harness"]
            ).discover_harness,
        ) as wrapped:
            run_selfcheck()
        self.assertGreaterEqual(wrapped.call_count, 2)


if __name__ == "__main__":
    unittest.main()
