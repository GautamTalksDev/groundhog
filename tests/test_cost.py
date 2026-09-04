"""Tests for gh.cost — measured / estimated / unknown bases."""

from __future__ import annotations

import unittest
from pathlib import Path

from gh.cluster import Cluster
from gh.cost import (
    cost_for_cluster,
    cost_for_session,
    load_prices,
    project_costs_from_sessions,
    resolve_model_price,
)
from gh.intents import Intent, normalize_intent
from gh.parse import Session, Turn

PRICES = load_prices(Path(__file__).resolve().parent.parent / "prices.json")


def _intent(**kwargs) -> Intent:
    text = kwargs.get("raw_text", "Fix the flaky auth test in login suite")
    return Intent(
        session_id=kwargs.get("session_id", "s1"),
        harness="claude_code",
        project=kwargs.get("project", "demo"),
        timestamp="2026-09-01T12:00:00Z",
        raw_text=text,
        normalized=normalize_intent(text),
        session_turn_count=2,
        session_tokens=kwargs.get("session_tokens"),
        session_input_tokens=kwargs.get("session_input_tokens"),
        session_output_tokens=kwargs.get("session_output_tokens"),
        session_cache_read_tokens=kwargs.get("session_cache_read_tokens"),
        session_model=kwargs.get("session_model"),
        session_text_chars=kwargs.get("session_text_chars", 0),
    )


def _cluster(members: list[Intent]) -> Cluster:
    return Cluster(
        id="c1",
        members=members,
        label=members[0].raw_text[:90],
        projects={m.project for m in members},
        first_seen="2026-09-01T12:00:00Z",
        last_seen="2026-09-01T12:00:00Z",
        run_count=len(members),
        cohesion=0.9,
    )


class ResolveModelTests(unittest.TestCase):
    def test_longest_prefix(self) -> None:
        rates, key = resolve_model_price("claude-opus-4-20250514", PRICES)
        self.assertEqual(key, "claude-opus-4")
        self.assertGreater(rates["input"], 0)

    def test_unknown_uses_default(self) -> None:
        _rates, key = resolve_model_price("totally-unknown-model-xyz", PRICES)
        self.assertEqual(key, "default")


class CostBasisTests(unittest.TestCase):
    def test_measured(self) -> None:
        intent = _intent(
            session_input_tokens=1_000_000,
            session_output_tokens=0,
            session_cache_read_tokens=0,
            session_model="claude-sonnet-4-20250514",
            session_tokens=1_000_000,
        )
        cost = cost_for_cluster(_cluster([intent]), PRICES)
        self.assertEqual(cost.basis, "measured")
        self.assertEqual(cost.input_tokens, 1_000_000)
        self.assertEqual(cost.price_model, "claude-sonnet-4")
        # $3 / MTok input for sonnet-4 in prices.json
        self.assertAlmostEqual(cost.usd, 3.0, places=4)

    def test_estimated(self) -> None:
        # 400 chars → 100 tokens at 4 chars/token; no measured usage.
        intent = _intent(
            session_input_tokens=None,
            session_output_tokens=None,
            session_cache_read_tokens=None,
            session_model="claude-sonnet-4",
            session_text_chars=400,
            session_tokens=None,
        )
        cost = cost_for_cluster(_cluster([intent]), PRICES)
        self.assertEqual(cost.basis, "estimated")
        self.assertEqual(cost.input_tokens, 100)
        self.assertEqual(cost.output_tokens, 0)
        self.assertGreater(cost.usd, 0.0)

    def test_unknown(self) -> None:
        intent = _intent(
            session_input_tokens=None,
            session_output_tokens=None,
            session_cache_read_tokens=None,
            session_text_chars=0,
            session_tokens=None,
        )
        cost = cost_for_cluster(_cluster([intent]), PRICES)
        self.assertEqual(cost.basis, "unknown")
        self.assertEqual(cost.usd, 0.0)
        self.assertEqual(cost.input_tokens, 0)

    def test_unknown_model_labeled_default(self) -> None:
        intent = _intent(
            session_input_tokens=500_000,
            session_output_tokens=0,
            session_cache_read_tokens=0,
            session_model="vendor-mystery-9",
            session_tokens=500_000,
        )
        cost = cost_for_cluster(_cluster([intent]), PRICES)
        self.assertEqual(cost.basis, "measured")
        self.assertEqual(cost.price_model, "default")
        self.assertAlmostEqual(cost.usd, 1.5, places=4)

    def test_dedupes_same_session(self) -> None:
        a = _intent(
            session_id="same",
            session_input_tokens=1000,
            session_output_tokens=0,
            session_cache_read_tokens=0,
            session_model="claude-sonnet-4",
        )
        b = _intent(
            session_id="same",
            session_input_tokens=1000,
            session_output_tokens=0,
            session_cache_read_tokens=0,
            session_model="claude-sonnet-4",
            raw_text="Also implement Codex session discovery under home",
        )
        cost = cost_for_cluster(_cluster([a, b]), PRICES)
        self.assertEqual(cost.input_tokens, 1000)


class SessionProjectCostTests(unittest.TestCase):
    def test_rollups_all_sessions_when_grouped_by_project(self) -> None:
        sessions = [
            Session(
                session_id="a",
                harness="cursor",
                project="GASKET",
                started_at="2026-08-31T12:00:00Z",
                ended_at="2026-08-31T12:10:00Z",
                turns=[
                    Turn("user", "Explore the repo for CP-5", None, None, None, None, None),
                    Turn(
                        "assistant",
                        "Looking.",
                        None,
                        1_000_000,
                        0,
                        0,
                        "claude-sonnet-4-20250514",
                    ),
                ],
                parse_status="ok",
            ),
            Session(
                session_id="b",
                harness="cursor",
                project="Keyring",
                started_at="2026-08-31T13:00:00Z",
                ended_at="2026-08-31T13:10:00Z",
                turns=[
                    Turn("user", "Explore the repo for CP-4", None, None, None, None, None),
                    Turn(
                        "assistant",
                        "Looking.",
                        None,
                        500_000,
                        0,
                        0,
                        "claude-sonnet-4-20250514",
                    ),
                ],
                parse_status="ok",
            ),
            Session(
                session_id="c",
                harness="cursor",
                project="GASKET",
                started_at="2026-09-01T12:00:00Z",
                ended_at="2026-09-01T12:10:00Z",
                turns=[
                    Turn("user", "xxxx", None, None, None, None, None),
                    Turn(
                        "assistant",
                        "ok",
                        None,
                        500_000,
                        0,
                        0,
                        "claude-sonnet-4-20250514",
                    ),
                ],
                parse_status="ok",
            ),
        ]
        costs = project_costs_from_sessions(sessions, PRICES)
        by_name = {c.project: c for c in costs}
        self.assertEqual(set(by_name), {"GASKET", "Keyring"})
        self.assertEqual(by_name["GASKET"].session_count, 2)
        self.assertEqual(by_name["GASKET"].input_tokens, 1_500_000)
        self.assertEqual(by_name["GASKET"].basis, "measured")
        self.assertAlmostEqual(by_name["GASKET"].usd, 4.5, places=4)
        self.assertEqual(by_name["Keyring"].session_count, 1)
        self.assertEqual(by_name["Keyring"].input_tokens, 500_000)

    def test_cost_for_session_estimates_from_text(self) -> None:
        session = Session(
            session_id="e",
            harness="cursor",
            project="Lading",
            started_at="2026-08-24T12:00:00Z",
            ended_at="2026-08-24T12:01:00Z",
            turns=[
                Turn("user", "a" * 400, None, None, None, None, None),
            ],
            parse_status="ok",
        )
        cost = cost_for_session(session, PRICES)
        self.assertEqual(cost.basis, "estimated")
        self.assertEqual(cost.input_tokens, 100)
        self.assertGreater(cost.usd, 0.0)


if __name__ == "__main__":
    unittest.main()
