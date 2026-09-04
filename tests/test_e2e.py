"""End-to-end pipeline tests with HOME overridden to fixtures."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gh.redact import redact_text
from groundhog import main


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _write_home(root: Path) -> Path:
    """Build a fake HOME with Claude sessions + one corrupted file."""
    proj = (
        root
        / ".claude"
        / "projects"
        / "-home-dev-demo"
    )
    proj.mkdir(parents=True)

    # Clean repeated chore (3 sessions) so ranking has something to show.
    for i, text in enumerate(
        [
            "Re-run the garak smoke report and compare it to baseline.report.jsonl",
            "Run garak smoke again and compare the report to baseline.report.jsonl",
            "Re-run garak smoke report and compare against the baseline jsonl output",
        ],
        1,
    ):
        path = proj / f"sess-{i}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "type": "user",
                    "sessionId": f"e2e-{i}",
                    "cwd": "/home/dev/solen-kernel",
                    "timestamp": f"2026-09-0{i}T12:00:00Z",
                    "message": {"role": "user", "content": text},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "assistant",
                    "sessionId": f"e2e-{i}",
                    "timestamp": f"2026-09-0{i}T12:01:00Z",
                    "model": "claude-opus-4-20250514",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Running."}],
                    },
                    "usage": {
                        "input_tokens": 2000,
                        "output_tokens": 400,
                        "cache_read_input_tokens": 100,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    # Corrupted / truncated file — must not crash the pipeline.
    (proj / "corrupted.jsonl").write_text(
        '{"type":"user","message":{"role":"user","content":"partial\n'
        "THIS IS NOT JSON\n"
        '{"type":"user","sessionId":"bad","message":{"role":"user","content":'
        '"Use key sk-abcdefghijklmnopqrstuvwxyz012345 and Bearer '
        'supersecret_token_value_here_abc"}}\n',
        encoding="utf-8",
    )

    # Also copy the project's malformed fixture for variety.
    malformed = FIXTURES / "malformed.jsonl"
    if malformed.exists():
        (proj / "malformed-copy.jsonl").write_text(
            malformed.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return root


class E2ETests(unittest.TestCase):
    def test_pipeline_with_fixture_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_home(Path(tmp))
            with mock.patch.dict(os.environ, {"HOME": str(home)}):
                code = main(
                    ["--days", "30", "--min-runs", "3", "--top", "3"]
                )
            self.assertEqual(code, 0)

    def test_pipeline_prints_readable_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_home(Path(tmp))
            out = Path(tmp) / "report.txt"
            with mock.patch.dict(os.environ, {"HOME": str(home)}):
                code = main(
                    [
                        "--days",
                        "30",
                        "--min-runs",
                        "3",
                        "--out",
                        str(out),
                    ]
                )
            self.assertEqual(code, 0)
            text = out.read_text(encoding="utf-8")
            self.assertIn("GROUNDHOG", text)
            self.assertIn("YOU KEEP REDOING THIS", text)
            self.assertIn("NOT COUNTED", text)
            self.assertIn("garak", text.lower())
            self.assertIn("Local only", text)
            # Secrets from corrupted file must not leak when redact is on.
            self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz012345", text)
            self.assertNotIn("Bearer supersecret_token_value_here_abc", text)

    def test_empty_home_exit_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HOME": tmp}):
                out = Path(tmp) / "empty.txt"
                code = main(["--out", str(out)])
            self.assertEqual(code, 0)
            text = out.read_text(encoding="utf-8")
            self.assertIn("No session files found", text)
            self.assertIn("Checked:", text)
            self.assertIn(".claude/projects", text)
            self.assertIn("NOT COUNTED", text)

    def test_redact_helper(self) -> None:
        raw = (
            "deploy with sk-abcdefghijklmnopqrstuvwxyz012345 "
            "and ghp_ABCDEFGHIJKLMNOPQRSTUVWX and "
            "AKIAIOSFODNN7EXAMPLE plus Bearer tok_abc_def_ghi_jkl"
        )
        cleaned = redact_text(raw, limit=120)
        self.assertNotIn("sk-abcdefghij", cleaned)
        self.assertNotIn("ghp_", cleaned)
        self.assertNotIn("AKIA", cleaned)
        self.assertNotIn("Bearer tok", cleaned)
        self.assertIn("<redacted>", cleaned)
        self.assertLessEqual(len(cleaned), 120)


if __name__ == "__main__":
    unittest.main()
