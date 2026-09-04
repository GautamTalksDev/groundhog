"""Tests for gh.cluster — TF-IDF agglomeration precision/recall."""

from __future__ import annotations

import unittest

from gh.cluster import (
    SIMILARITY_THRESHOLD,
    choose_cluster_label,
    clean_label,
    cluster_intents,
    label_imperative_score,
)
from gh.intents import Intent, normalize_intent


def _intent(text: str, project: str = "demo", session_id: str = "s") -> Intent:
    return Intent(
        session_id=session_id,
        harness="claude_code",
        project=project,
        timestamp="2026-09-01T12:00:00Z",
        raw_text=text,
        normalized=normalize_intent(text),
        session_turn_count=2,
        session_tokens=1000,
    )


class ClusterSameTaskTests(unittest.TestCase):
    def test_five_phrasings_one_cluster(self) -> None:
        phrasings = [
            "Fix the flaky authentication test in the login suite",
            "Please fix flaky auth test inside login test suite",
            "Repair the flaky login authentication test suite failure",
            "Fix flaky auth tests that keep failing in login suite",
            "Can you fix the flaky authentication tests in login",
        ]
        intents = [
            _intent(p, session_id=f"same-{i}") for i, p in enumerate(phrasings)
        ]
        clusters = cluster_intents(intents, min_runs=3)
        self.assertEqual(len(clusters), 1, clusters)
        self.assertEqual(clusters[0].run_count, 5)
        self.assertGreaterEqual(clusters[0].cohesion, 0.35)
        self.assertGreaterEqual(SIMILARITY_THRESHOLD, 0.4)


class ClusterUnrelatedTests(unittest.TestCase):
    def test_five_unrelated_stay_separate(self) -> None:
        tasks = [
            "Write a stranger-facing README for the billing product",
            "Implement JSONL session parsing into Turn records",
            "Re-run the garak smoke report against baseline output",
            "Deploy the kubernetes-api canary to staging us-east",
            "Design a new Postgres schema for inventory reservations",
        ]
        intents = [
            _intent(t, project=f"proj-{i}", session_id=f"u-{i}")
            for i, t in enumerate(tasks)
        ]
        # min_runs=1 so we observe raw separation before precision drop.
        clusters = cluster_intents(intents, min_runs=1)
        self.assertEqual(len(clusters), 5, [c.label for c in clusters])
        for cluster in clusters:
            self.assertEqual(cluster.run_count, 1)


class PrecisionFilterTests(unittest.TestCase):
    def test_min_runs_drops_pairs(self) -> None:
        intents = [
            _intent("Fix the flaky auth test in login suite", session_id="a"),
            _intent("Fix flaky authentication test in login", session_id="b"),
            _intent("Deploy canary to staging region west", session_id="c"),
            _intent("Deploy the canary build to staging west", session_id="d"),
        ]
        clusters = cluster_intents(intents, min_runs=3)
        self.assertEqual(clusters, [])


class ClusterLabelTests(unittest.TestCase):
    def test_prefers_imperative_over_confirmed_reply(self) -> None:
        members = [
            _intent(
                '**Confirmed. "No commit found for SHA" on all three.** '
                "The provenance is fabricated, not real.",
                session_id="a",
            ),
            _intent(
                "Investigate the fabricated provenance SHAs in the Lading "
                "manifest components and confirm which commits are missing.",
                session_id="b",
            ),
            _intent(
                "Check the three Lading component SHAs against git history "
                "and report which ones are fabricated.",
                session_id="c",
            ),
        ]
        # Dummy equal vectors so medoid is first; imperative scoring must still win.
        vectors = [{"x": 1.0}, {"x": 1.0}, {"x": 1.0}]
        label = choose_cluster_label(members, vectors)
        self.assertNotIn("Confirmed", label)
        self.assertFalse(label.startswith("**"))
        self.assertTrue(
            label.lower().startswith("investigate")
            or label.lower().startswith("check")
        )

    def test_clean_label_strips_and_word_boundary(self) -> None:
        raw = (
            '**Confirmed. "No commit found for SHA" on all three.** The '
            "provenance is fabricated, not a real git object anywhere."
        )
        cleaned = clean_label(raw, limit=70)
        self.assertFalse(cleaned.startswith("**"))
        self.assertFalse(cleaned.startswith('"'))
        self.assertLessEqual(len(cleaned), 71)  # 70 + ellipsis char
        if cleaned.endswith("…"):
            before = cleaned[:-1]
            self.assertFalse(before.endswith(" "))
            # Last char before ellipsis should end a whole word.
            self.assertRegex(before, r"[A-Za-z0-9)\]\"']$")

        locative = clean_label(
            "In /home/dev/Glasswake, harden crates/gw-dom for reliability "
            "and add property tests covering selector matching.",
            limit=70,
        )
        self.assertTrue(locative.lower().startswith("harden"))
        self.assertNotIn("/home/", locative)
        self.assertLessEqual(len(locative.rstrip("…")), 70)

    def test_falls_back_to_medoid_when_nothing_imperative(self) -> None:
        members = [
            _intent("**Confirmed. All three SHAs are fake.**", session_id="a"),
            _intent("Fair — that matches what I saw earlier today.", session_id="b"),
            _intent("You're right about the provenance story.", session_id="c"),
        ]
        vectors = [{"a": 1.0}, {"b": 1.0}, {"c": 1.0}]
        # Medoid with equal isolation is members[0]; score all negative.
        for m in members:
            self.assertLessEqual(label_imperative_score(m.raw_text), 0)
        label = choose_cluster_label(members, vectors)
        self.assertTrue(label)
        self.assertFalse(label.startswith("**"))


if __name__ == "__main__":
    unittest.main()
