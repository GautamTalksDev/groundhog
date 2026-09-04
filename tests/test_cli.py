"""CLI argument validation: --days, --top, and --min-runs must be positive."""

from __future__ import annotations

import unittest
from io import StringIO
from unittest.mock import patch

from groundhog import main


class CliValidationTests(unittest.TestCase):
    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        out = StringIO()
        err = StringIO()
        with patch("sys.stdout", out), patch("sys.stderr", err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_rejects_zero_and_negative_on_all_three_flags(self) -> None:
        cases = [
            (["--days", "0"], "--days"),
            (["--days", "-5"], "--days"),
            (["--top", "0"], "--top"),
            (["--top", "-1"], "--top"),
            (["--min-runs", "0"], "--min-runs"),
            (["--min-runs", "-2"], "--min-runs"),
        ]
        for argv, flag in cases:
            with self.subTest(argv=argv):
                code, _out, err = self._run(argv)
                self.assertNotEqual(code, 0, msg=err)
                self.assertIn(flag, err)
                self.assertIn("positive integer", err)
