"""Tests for gh.rank — scoring components and ordering."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from gh.cluster import Cluster
from gh.cost import CostBreakdown
from gh.intents import Intent, normalize_intent
from gh.rank import score_candidates


def _intent(
    text: str,
    *,
    session_id: str,
    turns: int,
    project: str = "demo",
    ts: str = "2026-09-02T12:00:00Z",
) -> Intent:
    return Intent(
        session_id=session_id,
        harness="claude_code",
        project=project,
        timestamp=ts,
        raw_text=text,
        normalized=normalize_intent(text),
        session_turn_count=turns,
        session_tokens=1000,
    )


def _cluster(
    cid: str,
    members: list[Intent],
    *,
    last_seen: str,
) -> Cluster:
    return Cluster(
        id=cid,
        members=members,
        label=members[0].raw_text[:90],
        projects={m.project for m in members},
        first_seen=members[-1].timestamp,
        last_seen=last_seen,
        run_count=len(members),
        cohesion=0.8,
    )


class RankTests(unittest.TestCase):
    def test_stable_expensive_outranks_volatile_cheap(self) -> None:
        now = datetime(2026, 9, 3, 18, 0, tzinfo=timezone.utc)
        stable = _cluster(
            "stable",
            [
                _intent("Run garak smoke and compare baseline", session_id="a1", turns=4, ts="2026-09-02T10:00:00Z"),
                _intent("Re-run garak smoke report vs baseline", session_id="a2", turns=4, ts="2026-09-02T18:00:00Z"),
                _intent("Run garak smoke again compare baseline", session_id="a3", turns=4, ts="2026-09-03T09:00:00Z"),
            ],
            last_seen="2026-09-03T09:00:00Z",
        )
        volatile = _cluster(
            "volatile",
            [
                _intent("Write stranger README one screen", session_id="b1", turns=3, ts="2026-09-01T10:00:00Z"),
                _intent("Rewrite stranger facing README", session_id="b2", turns=40, ts="2026-09-02T10:00:00Z"),
                _intent("Write one-screen README again", session_id="b3", turns=3, ts="2026-09-03T10:00:00Z"),
            ],
            last_seen="2026-09-03T10:00:00Z",
        )
        costs = [
            CostBreakdown(5000, 2000, 500, 0.40, "measured", "claude-opus-4"),
            CostBreakdown(1000, 400, 100, 0.05, "measured", "claude-sonnet-4"),
        ]
        result = score_candidates([stable, volatile], costs, now=now)
        self.assertEqual(result.candidates[0].cluster_id, "stable")
        self.assertGreater(
            result.candidates[0].stability, result.candidates[1].stability
        )
        self.assertTrue(result.candidates[0].evidence)
        self.assertIn("a1", result.candidates[0].session_ids)

    def test_recency_penalty(self) -> None:
        now = datetime(2026, 9, 3, 18, 0, tzinfo=timezone.utc)
        old = _cluster(
            "old",
            [
                _intent("Deploy canary to staging west", session_id="o1", turns=5, ts="2026-08-20T10:00:00Z"),
                _intent("Deploy canary build staging west", session_id="o2", turns=5, ts="2026-08-21T10:00:00Z"),
                _intent("Deploy the canary to staging", session_id="o3", turns=5, ts="2026-08-22T10:00:00Z"),
            ],
            last_seen="2026-08-22T10:00:00Z",
        )
        cost = CostBreakdown(8000, 3000, 1000, 0.5, "measured", "claude-opus-4")
        result = score_candidates([old], [cost], now=now)
        self.assertGreater(result.candidates[0].recency_days or 0, 7)
        # Without penalty score would be 1*1*1=1; with penalty 0.5.
        self.assertAlmostEqual(result.candidates[0].score, 0.5, places=4)

    def test_frequency_uses_distinct_sessions_not_member_count(self) -> None:
        now = datetime(2026, 9, 3, 18, 0, tzinfo=timezone.utc)
        same_session = _cluster(
            "dense",
            [
                _intent(
                    "Fix the flaky authentication test in the login suite",
                    session_id="only",
                    turns=4,
                    ts="2026-09-02T10:00:00Z",
                ),
                _intent(
                    "Please fix flaky auth test inside login test suite",
                    session_id="only",
                    turns=4,
                    ts="2026-09-02T11:00:00Z",
                ),
                _intent(
                    "Repair the flaky login authentication test suite failure",
                    session_id="only",
                    turns=4,
                    ts="2026-09-02T12:00:00Z",
                ),
                _intent(
                    "Fix flaky auth tests that keep failing in login suite",
                    session_id="only",
                    turns=4,
                    ts="2026-09-02T13:00:00Z",
                ),
                _intent(
                    "Can you fix the flaky authentication tests in login",
                    session_id="only",
                    turns=4,
                    ts="2026-09-03T09:00:00Z",
                ),
            ],
            last_seen="2026-09-03T09:00:00Z",
        )
        across_sessions = _cluster(
            "repeated",
            [
                _intent(
                    "Fix the flaky authentication test in the login suite",
                    session_id="a",
                    turns=4,
                    ts="2026-09-01T10:00:00Z",
                ),
                _intent(
                    "Please fix flaky auth test inside login test suite",
                    session_id="b",
                    turns=4,
                    ts="2026-09-02T10:00:00Z",
                ),
                _intent(
                    "Repair the flaky login authentication test suite failure",
                    session_id="c",
                    turns=4,
                    ts="2026-09-03T09:00:00Z",
                ),
            ],
            last_seen="2026-09-03T09:00:00Z",
        )
        cost = CostBreakdown(2000, 800, 100, 0.20, "measured", "claude-sonnet-4")
        result = score_candidates(
            [same_session, across_sessions], [cost, cost], now=now
        )
        by_id = {c.cluster_id: c for c in result.candidates}
        self.assertEqual(by_id["dense"].distinct_sessions, 1)
        self.assertEqual(by_id["repeated"].distinct_sessions, 3)
        self.assertGreater(by_id["repeated"].frequency, by_id["dense"].frequency)


if __name__ == "__main__":
    unittest.main()
