"""Tests for Cursor session discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gh.discover import SKIP_SYMLINK_OUTSIDE, cursor_project_name, discover_harness


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
        self.assertEqual(result.skipped, [])

    def test_symlink_escaping_root_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            outside = tmp_path / "outside.jsonl"
            outside.write_text('{"role":"user","message":{"content":"secret"}}\n')
            transcripts = (
                tmp_path
                / "home"
                / ".cursor"
                / "projects"
                / "home-x-projects-Evil"
                / "agent-transcripts"
            )
            transcripts.mkdir(parents=True)
            leak = transcripts / "leak.jsonl"
            leak.symlink_to(outside)
            result = discover_harness(
                "cursor", days=36500, home=tmp_path / "home"
            )
        self.assertEqual(result.files, [])
        self.assertEqual(len(result.skipped), 1)
        path, reason = result.skipped[0]
        self.assertTrue(path.endswith("leak.jsonl"))
        self.assertEqual(reason, SKIP_SYMLINK_OUTSIDE)
        self.assertNotIn(str(outside), path)

    def test_symlink_inside_root_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            transcripts = (
                tmp_path
                / "home"
                / ".cursor"
                / "projects"
                / "home-x-projects-Ok"
                / "agent-transcripts"
            )
            transcripts.mkdir(parents=True)
            real = transcripts / "real.jsonl"
            real.write_text('{"role":"user","message":{"content":"ok"}}\n')
            alias = transcripts / "alias.jsonl"
            alias.symlink_to(real)
            result = discover_harness(
                "cursor", days=36500, home=tmp_path / "home"
            )
        names = {Path(f.path).name for f in result.files}
        self.assertEqual(names, {"real.jsonl", "alias.jsonl"})
        self.assertEqual(result.skipped, [])


if __name__ == "__main__":
    unittest.main()
