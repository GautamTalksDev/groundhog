"""Tests for gh.suggest — parameters, tool recovery, honesty."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gh.cluster import Cluster
from gh.discover import SessionFile
from gh.intents import Intent
from gh.rank import Candidate
from gh.suggest import (
    common_and_varied_steps,
    infer_param_kind,
    parameterize_task,
    recover_tool_steps,
    suggest_scaffold,
    summarize_tool_input,
    ToolStep,
)


def _intent(text: str, session_id: str = "s1", project: str = "demo") -> Intent:
    return Intent(
        session_id=session_id,
        harness="cursor",
        project=project,
        timestamp="2026-08-24T12:00:00Z",
        raw_text=text,
        normalized=text.lower(),
        session_turn_count=4,
        session_tokens=None,
    )


def _candidate(cluster: Cluster) -> Candidate:
    return Candidate(
        cluster_id=cluster.id,
        label=cluster.label,
        score=0.5,
        frequency=0.5,
        cost_score=0.5,
        stability=1.0,
        run_count=cluster.run_count,
        distinct_sessions=len({m.session_id for m in cluster.members}),
        usd=0.0,
        cost_basis="estimated",
        recency_days=1.0,
        projects=set(cluster.projects),
        evidence=[],
        session_ids=sorted({m.session_id for m in cluster.members}),
        first_seen=cluster.first_seen,
        last_seen=cluster.last_seen,
    )


def _write_session(path: Path, tool_names: list[str]) -> None:
    lines = []
    for name in tool_names:
        lines.append(
            json.dumps(
                {
                    "role": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": name,
                                "input": {"path": f"/tmp/{name.lower()}.txt"},
                            }
                        ]
                    },
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ParameterizeTests(unittest.TestCase):
    def test_extracts_varying_slots_with_types(self) -> None:
        texts = [
            "Explore /home/dev/GASKET for CP-5 wire proxy implementation",
            "Explore /home/dev/Keyring for CP-4 wire proxy implementation",
            "Explore /home/dev/PLIMSOLL for CP-10 wire proxy implementation",
        ]
        description, params = parameterize_task(texts)
        self.assertIn("Explore", description)
        self.assertIn("wire proxy implementation", description)
        self.assertGreaterEqual(len(params), 1)
        kinds = {p.kind for p in params}
        self.assertTrue(kinds & {"path", "number", "identifier", "free text"})
        # Constant chore words survive.
        self.assertIn("Explore", description)
        self.assertTrue(any("{" in description for _ in [0]) or params)

    def test_infer_param_kinds(self) -> None:
        self.assertEqual(infer_param_kind(["/home/dev/app"]), "path")
        self.assertEqual(infer_param_kind(["README.md"]), "filename")
        self.assertEqual(infer_param_kind(["42", "7"]), "number")
        self.assertEqual(
            infer_param_kind(["a1b2c3d4e5f67890abcdef"]), "identifier"
        )


class ToolRecoveryTests(unittest.TestCase):
    def test_recovers_tool_use_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sess.jsonl"
            _write_session(path, ["Glob", "Read", "Shell"])
            steps = recover_tool_steps(str(path))
            self.assertEqual([s.name for s in steps], ["Glob", "Read", "Shell"])
            self.assertTrue(all(s.summary for s in steps))

    def test_summarize_prefers_useful_keys(self) -> None:
        self.assertIn(
            "command=",
            summarize_tool_input({"command": "ls -la", "timeout": 30}),
        )
        self.assertIn("path=", summarize_tool_input({"path": "/a/b.py"}))

    def test_common_vs_varied(self) -> None:
        a = [
            ToolStep("Glob", "path=a", key="Glob|path"),
            ToolStep("Read", "path=b", key="Read|path"),
            ToolStep("Shell", "command=x", key="Shell|command"),
        ]
        b = [
            ToolStep("Glob", "path=a", key="Glob|path"),
            ToolStep("Read", "path=b", key="Read|path"),
        ]
        c = [
            ToolStep("Glob", "path=a", key="Glob|path"),
            ToolStep("Read", "path=b", key="Read|path"),
            ToolStep("Grep", "pattern=z", key="Grep|pattern"),
        ]
        common, varied = common_and_varied_steps([a, b, c])
        common_names = {s.name for s in common}
        self.assertIn("Glob", common_names)
        self.assertIn("Read", common_names)
        varied_names = {s.name for s in varied}
        self.assertTrue({"Shell", "Grep"} & varied_names)


class HonestyTests(unittest.TestCase):
    def test_small_sample_note(self) -> None:
        members = [
            _intent("Fix the flaky auth test in login.spec.ts", "a"),
            _intent("Fix the flaky auth test in login.spec.ts", "b"),
        ]
        cluster = Cluster(
            id="c1",
            members=members,
            label="Fix the flaky auth test",
            projects={"demo"},
            first_seen="2026-08-24T12:00:00Z",
            last_seen="2026-08-24T12:00:00Z",
            run_count=2,
            cohesion=1.0,
        )
        result = suggest_scaffold(
            _candidate(cluster), cluster, session_files=[], rank=1
        )
        self.assertIn("too small to generalize", result.markdown)
        self.assertIn("starting point recovered from your transcripts", result.markdown)

    def test_sparse_tools_not_invented(self) -> None:
        members = [
            _intent("Explore the repo for checkpoint context", f"s{i}")
            for i in range(3)
        ]
        cluster = Cluster(
            id="c2",
            members=members,
            label="Explore the repo for checkpoint context",
            projects={"demo"},
            first_seen="2026-08-14T12:00:00Z",
            last_seen="2026-08-14T12:00:00Z",
            run_count=3,
            cohesion=0.8,
        )
        # Session files exist but have no tool_use blocks.
        with tempfile.TemporaryDirectory() as tmp:
            files = []
            for i in range(3):
                path = Path(tmp) / f"s{i}.jsonl"
                path.write_text(
                    json.dumps(
                        {
                            "role": "assistant",
                            "message": {
                                "content": [{"type": "text", "text": "Done."}]
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                st = path.stat()
                files.append(
                    SessionFile(
                        path=str(path),
                        harness="cursor",
                        mtime=st.st_mtime,
                        size_bytes=st.st_size,
                        project="demo",
                    )
                )
            result = suggest_scaffold(
                _candidate(cluster), cluster, session_files=files, rank=1
            )
        self.assertIn("Not recovered", result.markdown)
        self.assertIn("Not inventing steps", result.markdown)
        # Must not fabricate a numbered procedure from nothing.
        self.assertNotRegex(result.markdown, r"## Proposed procedure.*\n1\. `")


if __name__ == "__main__":
    unittest.main()
