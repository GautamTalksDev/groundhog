"""Tests for Cursor session discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gh.discover import cursor_project_name, discover_harness


FIXTURE_HOME = Path(__file__).resolve().parent / "fixtures" / "cursor_home"


class CursorProjectNameTests(unittest.TestCase):
    def test_final_segment_after_projects_preserves_case(self) -> None:
        self.assertEqual(
            cursor_project_name("home-tester-projects-DemoApp"),
            "DemoApp",
        )
        self.assertEqual(
            cursor_project_name("home-gautamtalksdev-projects-Glasswake"),
            "Glasswake",
        )
        self.assertEqual(
            cursor_project_name("home-tester-projects-mcp-pin"),
            "mcp-pin",
        )

    def test_fallback_to_raw_directory_name(self) -> None:
        self.assertEqual(cursor_project_name("1781912036518"), "1781912036518")
        self.assertEqual(cursor_project_name("home-tester-projects-"), "home-tester-projects-")


class CursorDiscoverTests(unittest.TestCase):
    def test_finds_nested_transcripts_and_project_name(self) -> None:
        result = discover_harness("cursor", days=36500, home=FIXTURE_HOME)
        self.assertEqual(result.sources["cursor"], "found")
        self.assertEqual(len(result.files), 2)

        by_name = {Path(f.path).name: f for f in result.files}
        self.assertEqual(set(by_name), {"parent.jsonl", "child.jsonl"})

        parent = by_name["parent.jsonl"]
        child = by_name["child.jsonl"]
        self.assertTrue(parent.path.endswith("agent-transcripts/parent.jsonl"))
        self.assertTrue(
            child.path.replace("\\", "/").endswith(
                "agent-transcripts/subagents/child.jsonl"
            )
        )
        for sf in result.files:
            self.assertEqual(sf.harness, "cursor")
            self.assertEqual(sf.project, "DemoApp")

    def test_missing_dir_is_absent_and_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = discover_harness("cursor", days=30, home=Path(tmp))
        self.assertEqual(result.sources["cursor"], "absent")
        self.assertEqual(result.files, [])


if __name__ == "__main__":
    unittest.main()
