"""Tests for gh.intents — substantive asks, rejections, normalization."""

from __future__ import annotations

import unittest

from gh.intents import (
    extract_intents,
    is_new_task,
    is_substantive,
    normalize_intent,
)
from gh.parse import Session, Turn


def _turn(role: str, text: str, **kwargs) -> Turn:
    return Turn(
        role=role,
        text=text,
        timestamp=kwargs.get("timestamp", "2026-08-20T10:00:00Z"),
        input_tokens=kwargs.get("input_tokens"),
        output_tokens=kwargs.get("output_tokens"),
        cache_read_tokens=kwargs.get("cache_read_tokens"),
        model=kwargs.get("model"),
    )


def _session(turns: list[Turn], session_id: str = "s1") -> Session:
    return Session(
        session_id=session_id,
        harness="claude_code",
        project="groundhog",
        started_at="2026-08-20T10:00:00Z",
        ended_at="2026-08-20T11:00:00Z",
        turns=turns,
        parse_status="ok",
    )


class SubstantiveTests(unittest.TestCase):
    def test_rejects_short_and_continuations(self) -> None:
        self.assertFalse(is_substantive("ok"))
        self.assertFalse(is_substantive("yes"))
        self.assertFalse(is_substantive("continue"))
        self.assertFalse(is_substantive("go on"))
        self.assertFalse(is_substantive("try again"))
        self.assertFalse(is_substantive("fix it"))
        self.assertFalse(is_substantive("thanks"))
        self.assertFalse(is_substantive("short ask"))  # < 25 chars after strip

    def test_rejects_stack_trace_blob(self) -> None:
        blob = (
            "Traceback (most recent call last):\n"
            '  File "/home/dev/app/main.py", line 42, in <module>\n'
            "    run()\n"
            '  File "/home/dev/app/main.py", line 10, in run\n'
            "    explode()\n"
            "ValueError: boom\n"
            "Caused by: KeyError: 'x'\n"
        )
        self.assertFalse(is_substantive(blob))

    def test_accepts_real_task(self) -> None:
        self.assertTrue(
            is_substantive("Fix the flaky auth test in login.spec.ts")
        )


class ConversationalReplyTests(unittest.TestCase):
    def test_rejects_markdown_emphasis_block(self) -> None:
        self.assertFalse(
            is_substantive(
                '**Confirmed. "No commit found for SHA" on all three.** '
                "The provenance is fabricated, not a real git object."
            )
        )

    def test_rejects_multi_paragraph_analysis(self) -> None:
        analysis = (
            "The honesty path fired correctly — that's the hard part working, "
            "and rank 2 recovering real steps proves recovery works.\n\n"
            "But this output has just exposed something more important than "
            "the feature, and you should see it clearly now.\n\n"
            "Your ranker is surfacing conversation density, not repetition, "
            "which is the opposite of the promise on the Play page."
        )
        self.assertFalse(is_substantive(analysis))

    def test_rejects_fenced_prompt_quote(self) -> None:
        pasted = (
            "Paste into Cursor:\n\n"
            "```\n"
            "Fix a correctness bug: clusters can currently be formed from "
            "multiple intents within a single session.\n"
            "```\n"
        )
        self.assertFalse(is_substantive(pasted))

    def test_rejects_over_length_ceiling(self) -> None:
        padded = "Fix the flaky auth test in login.spec.ts. " + ("word " * 400)
        self.assertGreater(len(padded), 1200)
        self.assertFalse(is_substantive(padded))

    def test_keeps_short_imperative_task(self) -> None:
        self.assertTrue(
            is_substantive("Fix the flaky auth test in login.spec.ts")
        )


class NewTaskTests(unittest.TestCase):
    def test_corrections_rejected_new_tasks_kept(self) -> None:
        self.assertFalse(is_new_task("actually make it blue instead"))
        self.assertFalse(is_new_task("wait, change that filename"))
        self.assertTrue(
            is_new_task(
                "Implement session discovery for Codex history files next"
            )
        )


class NormalizeTests(unittest.TestCase):
    def test_paths_shas_urls_numbers_quotes(self) -> None:
        raw = (
            'Update /home/dev/acme/app.py and https://example.com/x '
            "after commit a1b2c3d4e5f67890 using id "
            "123e4567-e89b-12d3-a456-426614174000 for ticket 12345 "
            'named "Billing Worker"'
        )
        norm = normalize_intent(raw)
        self.assertNotIn("<path>", norm)
        self.assertIn("<url>", norm)
        self.assertIn("<sha>", norm)
        self.assertIn("<id>", norm)
        self.assertIn("<num>", norm)
        self.assertIn("<str>", norm)
        self.assertNotIn("/home/dev", norm)
        self.assertNotIn("a1b2c3d", norm)

    def test_same_task_different_wording_shares_tokens(self) -> None:
        a = normalize_intent(
            "Fix the flaky authentication test in the login suite"
        )
        b = normalize_intent(
            "Please fix flaky auth test inside login test suite"
        )
        ta, tb = set(a.split()), set(b.split())
        overlap = ta & tb
        # Most meaningful tokens should align after stopword stripping.
        self.assertGreaterEqual(len(overlap), 3)
        self.assertIn("fix", overlap)
        self.assertTrue({"flaky", "login", "test", "suite"} & overlap)


class ExtractTests(unittest.TestCase):
    def test_first_substantive_plus_later_new_task(self) -> None:
        session = _session(
            [
                _turn("user", "ok"),
                _turn(
                    "user",
                    "Fix the flaky auth test in login.spec.ts",
                    input_tokens=100,
                    output_tokens=50,
                ),
                _turn("assistant", "Working on it.", output_tokens=20),
                _turn("user", "try again"),
                _turn("user", "actually rename the helper too"),
                _turn(
                    "user",
                    "Also implement Codex session discovery under ~/.codex",
                ),
            ]
        )
        intents = extract_intents([session])
        self.assertEqual(len(intents), 2)
        self.assertIn("flaky auth", intents[0].raw_text)
        self.assertIn("Codex session discovery", intents[1].raw_text)
        self.assertEqual(intents[0].session_tokens, 170)
        self.assertTrue(intents[0].normalized)

    def test_no_tokens_stays_none(self) -> None:
        session = _session(
            [_turn("user", "Write a README for the groundhog CLI tool")]
        )
        intents = extract_intents([session])
        self.assertEqual(len(intents), 1)
        self.assertIsNone(intents[0].session_tokens)


class CursorWrapTests(unittest.TestCase):
    def test_extracts_inner_user_query(self) -> None:
        wrapped = (
            "<timestamp>Friday, Sep 4, 2026, 1:33 AM (UTC-4)</timestamp>\n"
            "<user_query>\n"
            "Fix the flaky auth test in login.spec.ts\n"
            "</user_query>"
        )
        session = _session(
            [_turn("user", wrapped, timestamp=None)],
            session_id="cursor-1",
        )
        session.harness = "cursor"
        intents = extract_intents([session])
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].raw_text, "Fix the flaky auth test in login.spec.ts")
        self.assertNotIn("<user_query>", intents[0].raw_text)
        self.assertNotIn("<timestamp>", intents[0].raw_text)

    def test_parses_timestamp_tag(self) -> None:
        wrapped = (
            "<timestamp>Friday, Sep 4, 2026, 1:33 AM (UTC-4)</timestamp>\n"
            "<user_query>\n"
            "Fix the flaky auth test in login.spec.ts\n"
            "</user_query>"
        )
        session = _session(
            [_turn("user", wrapped, timestamp=None)],
        )
        session.started_at = "2026-01-01T00:00:00Z"
        intents = extract_intents([session])
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].timestamp, "2026-09-04T01:33:00-04:00")

    def test_unparseable_timestamp_falls_back(self) -> None:
        wrapped = (
            "<timestamp>not a real date</timestamp>\n"
            "<user_query>\n"
            "Fix the flaky auth test in login.spec.ts\n"
            "</user_query>"
        )
        session = _session(
            [_turn("user", wrapped, timestamp="2026-08-20T10:00:00Z")],
        )
        intents = extract_intents([session])
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].timestamp, "2026-08-20T10:00:00Z")

    def test_neither_tag_passes_through_unchanged(self) -> None:
        plain = "Fix the flaky auth test in login.spec.ts"
        session = _session(
            [_turn("user", plain, timestamp="2026-08-20T10:00:00Z")],
        )
        intents = extract_intents([session])
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].raw_text, plain)
        self.assertEqual(intents[0].timestamp, "2026-08-20T10:00:00Z")
        self.assertIn("flaky", intents[0].normalized)

    def test_timestamp_without_user_query_is_stripped(self) -> None:
        wrapped = (
            "<timestamp>Monday, Aug 31, 2026, 3:13 PM (UTC-4)</timestamp>\n"
            "Explore the GASKET repo for checkpoint wire proxy implementation"
        )
        session = _session([_turn("user", wrapped, timestamp=None)])
        intents = extract_intents([session])
        self.assertEqual(len(intents), 1)
        self.assertNotIn("<", intents[0].raw_text)
        self.assertNotIn("timestamp", intents[0].raw_text.lower())
        self.assertTrue(
            intents[0].raw_text.startswith("Explore the GASKET repo")
        )
        self.assertEqual(intents[0].timestamp, "2026-08-31T15:13:00-04:00")

    def test_normalized_truncates_at_400_raw_text_does_not(self) -> None:
        prefix = "Please implement the billing client retry logic now. "
        late_token = "zebraquark"
        # Pad so the late token starts after character 400.
        pad = "padding " * 80
        query = prefix + pad + late_token
        self.assertGreater(len(query), 400)
        self.assertGreater(query.find(late_token), 400)
        wrapped = (
            "<timestamp>Friday, Sep 4, 2026, 1:33 AM (UTC-4)</timestamp>\n"
            f"<user_query>\n{query}\n</user_query>"
        )
        session = _session([_turn("user", wrapped, timestamp=None)])
        intents = extract_intents([session])
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].raw_text, query)
        self.assertIn(late_token, intents[0].raw_text)
        self.assertIn("billing", intents[0].normalized)
        self.assertNotIn(late_token, intents[0].normalized)


class SignalExtractionTests(unittest.TestCase):
    def test_strips_leading_locative_and_drops_path_placeholder(self) -> None:
        gasket = normalize_intent(
            "In /home/gautamtalksdev/projects/GASKET, explore for CP-5 "
            "wire proxy implementation"
        )
        keyring = normalize_intent(
            "In /home/gautamtalksdev/projects/Keyring, explore for CP-5 "
            "wire proxy implementation"
        )
        cursor = normalize_intent(
            "In Cursor, explore for CP-5 wire proxy implementation"
        )
        self.assertEqual(gasket, keyring)
        self.assertEqual(gasket, cursor)
        self.assertNotIn("<path>", gasket)
        self.assertNotIn("home", gasket)
        self.assertTrue(gasket.startswith("explore"))
        self.assertIn("proxy", gasket)

    def test_rejects_meta_comments_file_fetches_and_short_asks(self) -> None:
        self.assertFalse(is_substantive("What is Glasswakew"))
        self.assertFalse(
            is_substantive(
                "Fair — I mixed three different kinds of instruction "
                "together. Here it is as plain steps."
            )
        )
        self.assertFalse(is_substantive("You're right, that's the better approach"))
        self.assertFalse(is_substantive("Sorry, I meant the other file"))
        self.assertFalse(
            is_substantive(
                "RESULTS-FPR.md and RESULTS-KT1-v2.md Send me this file please"
            )
        )
        self.assertTrue(
            is_substantive("Fix the flaky auth test in login.spec.ts")
        )

    def test_boilerplate_stripped_from_normalized_not_raw_text(self) -> None:
        template = (
            "Implement the plan as specified, it is attached for your "
            "reference. Do NOT edit the plan file itself.\n\n"
            "To-do's from the plan have already been created. Do not create "
            "them again. Mark them as in_progress as you work, starting with "
            "the first one. Don't stop until you have completed all the to-dos.\n\n"
        )
        task = "Explore the repo for CP-5 wire proxy implementation."
        session = _session([_turn("user", template + task)])
        intents = extract_intents([session])
        self.assertEqual(len(intents), 1)
        self.assertIn("Implement the plan as specified", intents[0].raw_text)
        self.assertIn(
            "Don't stop until you have completed all the to-dos",
            intents[0].raw_text,
        )
        self.assertNotIn("specified", intents[0].normalized)
        self.assertNotIn("attached", intents[0].normalized)
        self.assertNotIn("reference", intents[0].normalized)
        self.assertNotIn("completed", intents[0].normalized)
        self.assertIn("explore", intents[0].normalized)
        self.assertIn("proxy", intents[0].normalized)
        self.assertEqual(
            intents[0].normalized,
            normalize_intent(task),
        )

    def test_normalized_keeps_only_first_30_tokens(self) -> None:
        tail = " ".join(f"token{i:02d}" for i in range(40))
        text = "Please implement the billing retry logic now. " + tail
        norm = normalize_intent(text)
        tokens = norm.split()
        self.assertLessEqual(len(tokens), 30)
        self.assertIn("token00", norm)
        self.assertNotIn("token39", norm)
        self.assertNotIn("token30", norm)


class CrossProjectNormalizeTests(unittest.TestCase):
    def test_explore_checkpoint_overlap_and_unrelated_task_does_not(self) -> None:
        projects = ["GASKET", "Keyring", "PLIMSOLL", "Glasswake"]
        gasket = normalize_intent(
            "Explore /home/gautamtalksdev/projects/GASKET for CP-5 "
            "implementation context. Return cmd/gasket structure and how "
            "the check is wired.",
            projects=projects,
        )
        plimsoll = normalize_intent(
            "Explore /home/gautamtalksdev/projects/PLIMSOLL for CP-10 "
            "implementation context. Return internal/log structure and how "
            "the check is wired.",
            projects=projects,
        )
        other = normalize_intent(
            "Harden crates/gw-dom for reliability. Add property tests "
            "for the selector matching engine.",
            projects=projects,
        )
        ga, pl, ot = set(gasket.split()), set(plimsoll.split()), set(other.split())
        overlap_same = ga & pl
        overlap_diff = ga & ot
        self.assertIn("<checkpoint>", overlap_same)
        self.assertIn("explore", overlap_same)
        self.assertIn("implementation", overlap_same)
        self.assertIn("context", overlap_same)
        self.assertNotIn("gasket", ga)
        self.assertNotIn("plimsoll", pl)
        self.assertIn("<subpath>", ga)
        self.assertIn("<subpath>", pl)
        self.assertGreaterEqual(len(overlap_same), 5)
        self.assertLess(len(overlap_diff), len(overlap_same))
        self.assertNotIn("explore", ot)


if __name__ == "__main__":
    unittest.main()
