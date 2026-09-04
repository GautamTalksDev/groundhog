"""Tests for gh.parse — fixture-driven session normalization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gh.discover import SessionFile
from gh.parse import first_present, parse_sessions, unwrap_cursor_text

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _sf(name: str, harness: str) -> SessionFile:
    path = FIXTURES / name
    st = path.stat()
    return SessionFile(
        path=str(path),
        harness=harness,
        mtime=st.st_mtime,
        size_bytes=st.st_size,
    )


class FirstPresentTests(unittest.TestCase):
    def test_dotted_path_and_fallback(self) -> None:
        obj = {"message": {"content": "hello"}, "text": "ignored"}
        self.assertEqual(first_present(obj, ["missing", "message.content"]), "hello")
        self.assertIsNone(first_present(obj, ["nope", "also.nope"]))

    def test_missing_intermediate_is_not_fatal(self) -> None:
        obj = {"usage": None}
        self.assertIsNone(first_present(obj, ["usage.input_tokens"]))


class ClaudeCleanTests(unittest.TestCase):
    def test_parses_turns_tokens_and_project(self) -> None:
        result = parse_sessions([_sf("claude_clean.jsonl", "claude_code")])
        self.assertEqual(result.skipped, [])
        self.assertEqual(len(result.sessions), 1)
        session = result.sessions[0]
        self.assertEqual(session.session_id, "claude-clean-001")
        self.assertEqual(session.harness, "claude_code")
        self.assertEqual(session.project, "acme-api")
        self.assertEqual(session.parse_status, "ok")
        self.assertEqual(len(session.turns), 4)

        user0, asst0, user1, asst1 = session.turns
        self.assertEqual(user0.role, "user")
        self.assertIn("flaky auth test", user0.text)
        self.assertIsNone(user0.input_tokens)

        self.assertEqual(asst0.role, "assistant")
        self.assertEqual(asst0.input_tokens, 1200)
        self.assertEqual(asst0.output_tokens, 340)
        self.assertEqual(asst0.cache_read_tokens, 800)
        self.assertEqual(asst0.model, "claude-opus-4-20250514")

        # Content-block array concatenated for user turn.
        self.assertIn("rerun the suite", user1.text)

        # camelCase usage variant on second assistant turn.
        self.assertEqual(asst1.input_tokens, 1500)
        self.assertEqual(asst1.output_tokens, 220)
        self.assertEqual(asst1.cache_read_tokens, 900)


class CodexCleanTests(unittest.TestCase):
    def test_parses_codex_shapes(self) -> None:
        result = parse_sessions([_sf("codex_clean.jsonl", "codex")])
        self.assertEqual(result.skipped, [])
        self.assertEqual(len(result.sessions), 1)
        session = result.sessions[0]
        self.assertEqual(session.session_id, "codex-clean-001")
        self.assertEqual(session.harness, "codex")
        self.assertEqual(session.project, "billing-service")
        self.assertEqual(len(session.turns), 4)

        self.assertEqual(session.turns[0].role, "user")
        self.assertIn("OpenAPI client", session.turns[0].text)

        asst0 = session.turns[1]
        self.assertEqual(asst0.role, "assistant")
        self.assertEqual(asst0.input_tokens, 900)
        self.assertEqual(asst0.output_tokens, 410)
        self.assertEqual(asst0.cache_read_tokens, 100)
        self.assertEqual(asst0.model, "o3")

        # prompt/completion token aliases, cache absent -> None.
        asst1 = session.turns[3]
        self.assertEqual(asst1.input_tokens, 1100)
        self.assertEqual(asst1.output_tokens, 180)
        self.assertIsNone(asst1.cache_read_tokens)


class MalformedTests(unittest.TestCase):
    def test_skips_bad_lines_keeps_usable_turns(self) -> None:
        result = parse_sessions([_sf("malformed.jsonl", "claude_code")])
        self.assertEqual(result.skipped, [])
        self.assertEqual(len(result.sessions), 1)
        session = result.sessions[0]
        self.assertGreaterEqual(result.malformed_lines, 1)
        self.assertIn("malformed", session.parse_status)

        # Truncated line skipped; tool_use-only and odd content shapes yield
        # no text turn; final assistant text kept with tokens None.
        roles = [t.role for t in session.turns]
        self.assertEqual(roles[0], "user")
        self.assertEqual(roles[-1], "assistant")
        last = session.turns[-1]
        self.assertIn("Cleanup done", last.text)
        self.assertIsNone(last.input_tokens)
        self.assertIsNone(last.output_tokens)
        self.assertIsNone(last.cache_read_tokens)

    def test_unreadable_file_is_skipped(self) -> None:
        missing = SessionFile(
            path=str(FIXTURES / "does_not_exist.jsonl"),
            harness="claude_code",
            mtime=0.0,
            size_bytes=0,
        )
        result = parse_sessions([missing])
        self.assertEqual(result.sessions, [])
        self.assertEqual(len(result.skipped), 1)
        self.assertIn("unreadable", result.skipped[0][1])


class CursorTagStripTests(unittest.TestCase):
    def test_parse_strips_timestamp_from_turn_text(self) -> None:
        record = {
            "role": "user",
            "timestamp": "2026-08-31T19:13:00Z",
            "message": {
                "content": (
                    "<timestamp>Monday, Aug 31, 2026, 3:13 PM (UTC-4)</timestamp>\n"
                    "Explore the GASKET repo for checkpoint wire proxy "
                    "implementation"
                )
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cursor.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            st = path.stat()
            result = parse_sessions(
                [
                    SessionFile(
                        path=str(path),
                        harness="cursor",
                        mtime=st.st_mtime,
                        size_bytes=st.st_size,
                    )
                ]
            )
        self.assertEqual(len(result.sessions), 1)
        turn = result.sessions[0].turns[0]
        self.assertNotIn("<", turn.text)
        self.assertNotIn("timestamp", turn.text.lower())
        self.assertTrue(turn.text.startswith("Explore the GASKET repo"))
        self.assertEqual(turn.timestamp, "2026-08-31T15:13:00-04:00")

    def test_unwrap_drops_timestamp_without_user_query(self) -> None:
        query, ts = unwrap_cursor_text(
            "<timestamp>Monday, Aug 31, 2026, 3:13 PM (UTC-4)</timestamp>\n"
            "Explore the Keyring repo for checkpoint wire proxy implementation"
        )
        self.assertEqual(
            query,
            "Explore the Keyring repo for checkpoint wire proxy implementation",
        )
        self.assertNotIn("<", query)
        self.assertEqual(ts, "2026-08-31T15:13:00-04:00")


if __name__ == "__main__":
    unittest.main()
