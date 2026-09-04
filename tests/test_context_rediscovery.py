"""Tests for context rediscovery: exploration prefix before the first edit."""

from __future__ import annotations

import unittest

from gh.context_rediscovery import (
    MIN_RESOLVABLE,
    RediscoveryReport,
    exploration_prefix,
    has_explore_loop,
    is_mutating,
    measure_context_rediscovery,
)
from gh.parse import Session, ToolCall, Turn
from gh.rank import RankResult
from gh.render import VERDICT_NULL, build_report, render_text


def _call(name: str, path: str | None = None, command: str | None = None) -> ToolCall:
    return ToolCall(name=name, path=path, command=command)


def _session(
    sid: str,
    calls: list[ToolCall],
    *,
    harness: str = "cursor",
    project: str = "demo",
) -> Session:
    return Session(
        session_id=sid,
        harness=harness,
        project=project,
        started_at="2026-09-01T12:00:00Z",
        ended_at="2026-09-01T12:10:00Z",
        turns=[Turn("user", "do the thing", None, None, None, None, None)],
        parse_status="ok",
        tool_calls=calls,
    )


class PrefixDetectionTests(unittest.TestCase):
    def test_glob_grep_reads_then_edit(self) -> None:
        calls = [
            _call("Glob"),
            _call("Grep"),
            _call("Read", path="/repo/a.py"),
            _call("Read", path="/repo/b.py"),
            _call("StrReplace", path="/repo/a.py"),
            _call("Read", path="/repo/c.py"),
        ]
        prefix = exploration_prefix(calls)
        self.assertIsNotNone(prefix)
        self.assertEqual(len(prefix), 4)
        self.assertTrue(has_explore_loop(prefix))
        self.assertTrue(is_mutating(_call("StrReplace")))
        self.assertTrue(is_mutating(_call("Write")))
        self.assertTrue(is_mutating(_call("Edit")))

    def test_readonly_shell_stays_in_prefix(self) -> None:
        calls = [
            _call("Shell", command="ls -la && git status"),
            _call("Read", path="/repo/main.go"),
            _call("Shell", command="rm -rf /tmp/out"),
        ]
        prefix = exploration_prefix(calls)
        self.assertEqual(len(prefix), 2)
        self.assertFalse(is_mutating(_call("Shell", command="git log --oneline")))
        self.assertTrue(is_mutating(_call("Shell", command="git commit -am 'x'")))

    def test_no_mutation_returns_none_prefix(self) -> None:
        calls = [
            _call("Glob"),
            _call("Read", path="/repo/a.py"),
            _call("Read", path="/repo/b.py"),
        ]
        self.assertIsNone(exploration_prefix(calls))

    def test_no_mutation_sessions_excluded_from_median(self) -> None:
        sessions = [
            _session("m1", [_call("Read", path="/a"), _call("Write", path="/a")]),
            _session("m2", [_call("Read", path="/b"), _call("StrReplace")]),
            _session("m3", [_call("Glob"), _call("Write")]),
            _session("m4", [_call("Read", path="/c"), _call("Edit")]),
            _session("m5", [_call("Read", path="/d"), _call("Write")]),
            _session("look", [_call("Glob"), _call("Grep"), _call("Read", path="/z")]),
        ]
        result = measure_context_rediscovery(sessions)
        self.assertEqual(result.no_mutation_sessions, 1)
        self.assertEqual(result.resolvable_sessions, 5)
        self.assertTrue(result.sufficient)
        self.assertIsNotNone(result.median_prefix)
        # look-only session must not pull the median toward 3.
        self.assertLessEqual(result.median_prefix, 1.0)

    def test_insufficient_sample_refuses_percentages(self) -> None:
        sessions = [
            _session(
                f"s{i}",
                [_call("Read", path=f"/f{i}"), _call("Write")],
            )
            for i in range(MIN_RESOLVABLE - 1)
        ]
        result = measure_context_rediscovery(sessions)
        self.assertEqual(result.resolvable_sessions, 4)
        self.assertFalse(result.sufficient)
        self.assertIsNone(result.pattern_pct)
        self.assertIsNone(result.median_prefix)
        self.assertIsNone(result.explore_pct)

    def test_harness_without_tools_is_excluded(self) -> None:
        sessions = [
            _session("c1", [], harness="claude_code"),
            _session(
                "k1",
                [_call("Read", path="/a"), _call("Write")],
                harness="cursor",
            ),
        ]
        result = measure_context_rediscovery(sessions)
        self.assertIn("claude_code", result.harnesses_excluded)
        self.assertEqual(result.sessions_with_tools, 1)


class RenderRediscoveryTests(unittest.TestCase):
    def test_section_renders_under_defensible_null(self) -> None:
        calls = [
            _call("Glob"),
            _call("Grep"),
            _call("Read", path="/home/u/projects/demo/internal/log/merkle.go"),
            _call("Read", path="/home/u/projects/demo/cmd/root.go"),
            _call("StrReplace"),
        ]
        sessions = [_session(f"s{i}", calls, project="GASKET") for i in range(5)]
        rd = measure_context_rediscovery(sessions)
        report = build_report(
            days=30,
            min_runs=3,
            top=3,
            session_count=5,
            harness_statuses={"cursor": "found"},
            rank_result=RankResult(candidates=[]),
            skipped=[],
            rediscovery=rd,
        )
        text = render_text(report)
        self.assertEqual(report.verdict, VERDICT_NULL)
        self.assertIn("THE WORK YOUR AGENT REDOES EVERY SESSION", text)
        self.assertIn("of sessions begin by re-deriving the same context", text)
        idx_rd = text.index("THE WORK YOUR AGENT REDOES EVERY SESSION")
        idx_chores = text.index("YOU KEEP REDOING THIS")
        self.assertLess(idx_rd, idx_chores)

    def test_file_paths_are_redacted(self) -> None:
        secret_path = "/home/u/projects/demo/sk-abcdefghijklmnopqrstuvwxyz012345.txt"
        calls = [
            _call("Glob"),
            _call("Grep"),
            _call("Read", path=secret_path),
            _call("Read", path=secret_path),
            _call("Write"),
        ]
        sessions = [_session(f"s{i}", calls) for i in range(5)]
        rd = measure_context_rediscovery(sessions)
        report = build_report(
            days=14,
            min_runs=3,
            top=3,
            session_count=5,
            harness_statuses={"cursor": "found"},
            rank_result=RankResult(candidates=[]),
            skipped=[],
            rediscovery=rd,
            redact=True,
        )
        text = render_text(report)
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz012345", text)
        self.assertIn("<redacted>", text)

    def test_insufficient_copy_names_the_count(self) -> None:
        rd = RediscoveryReport(
            resolvable_sessions=3,
            no_mutation_sessions=2,
            sessions_with_tools=5,
            sufficient=False,
            notes=["2 sessions had no mutating call; not folded into the median"],
        )
        report = build_report(
            days=14,
            min_runs=3,
            top=3,
            session_count=5,
            harness_statuses={"cursor": "found"},
            rank_result=RankResult(candidates=[]),
            skipped=[],
            rediscovery=rd,
        )
        text = render_text(report)
        self.assertIn("not enough to report rates (need 5)", text)
        self.assertNotIn("begin by re-deriving", text)


if __name__ == "__main__":
    unittest.main()
