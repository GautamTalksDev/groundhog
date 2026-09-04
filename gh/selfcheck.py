"""Runtime self-check: bundled fixtures through the real analyzer.

Runs at the start of every scan, before any user data is read. Cases call
``gh.cluster.cluster_intents``, ``gh.intents.extract_intents``,
``gh.cost.cost_for_session``, ``gh.discover.discover_harness``, and the
parse/verdict helpers those stages already use. A case that stops passing
means the analyzer changed — the check fails rather than being rewritten.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from gh.cluster import cluster_intents
from gh.cost import cost_for_session, load_prices
from gh.discover import (
    SKIP_SYMLINK_OUTSIDE,
    SKIP_UNREADABLE_DIR,
    SessionFile,
    discover_harness,
    is_unreadable_dir_reason,
)
from gh.intents import extract_intents
from gh.parse import parse_sessions
from gh.render import VERDICT_PARTIAL, classify_verdict

# Phrasings known to cluster when they come from distinct sessions.
# No credential-shaped tokens. Not stored under any harness discovery root.
_ASK_A = (
    "Re-run the garak smoke report and compare it to baseline.report.jsonl"
)
_ASK_B = (
    "Run garak smoke again and compare the report to baseline.report.jsonl"
)
_ASK_C = (
    "Re-run garak smoke report and compare against the baseline jsonl output"
)

_SELFCHECK_MS_BUDGET = 250.0


@dataclass
class CaseResult:
    """One bundled case and what the real analyzer produced."""

    name: str
    passed: bool
    detail: str


@dataclass
class SelfCheckResult:
    """Outcome of running every bundled analyzer case."""

    passed: int
    total: int
    elapsed_ms: float
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.passed == self.total

    @property
    def failures(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.passed]

    def headline(self) -> str:
        if self.ok:
            return (
                f"Self-check: PASSED ({self.passed}/{self.total} "
                "bundled analyzer cases)"
            )
        return (
            f"Self-check: FAILED ({self.passed}/{self.total}) — "
            "THIS ANALYZER IS NOT BEHAVING AS BUILT"
        )

    def coverage_value(self) -> str:
        status = "passed" if self.ok else "failed"
        ms = max(0, int(round(self.elapsed_ms)))
        return f"{self.passed}/{self.total} {status} · {ms}ms"


def report_kwargs(result: Optional[SelfCheckResult]) -> dict:
    """Keyword args ``build_report`` needs to surface a self-check."""
    if result is None:
        return {}
    return {
        "selfcheck_ok": result.ok,
        "selfcheck_passed": result.passed,
        "selfcheck_total": result.total,
        "selfcheck_ms": result.elapsed_ms,
        "selfcheck_headline": result.headline(),
        "selfcheck_failures": [(c.name, c.detail) for c in result.failures],
        "selfcheck_coverage": result.coverage_value(),
    }


def result_to_dict(result: SelfCheckResult) -> dict:
    """JSON-ready snapshot for the Play artifact and the report payload."""
    return {
        "ok": result.ok,
        "passed": result.passed,
        "total": result.total,
        "elapsed_ms": round(result.elapsed_ms, 3),
        "headline": result.headline(),
        "coverage": result.coverage_value(),
        "cases": [asdict(c) for c in result.cases],
        "failures": [asdict(c) for c in result.failures],
    }


def result_from_dict(payload: Optional[dict]) -> Optional[SelfCheckResult]:
    """Rehydrate a Play artifact. Missing/broken payload → None."""
    if not payload or not isinstance(payload, dict):
        return None
    cases = [
        CaseResult(
            name=str(item.get("name") or "unnamed"),
            passed=bool(item.get("passed")),
            detail=str(item.get("detail") or ""),
        )
        for item in (payload.get("cases") or [])
        if isinstance(item, dict)
    ]
    total = int(payload.get("total") or len(cases) or 0)
    passed = int(payload.get("passed") or sum(1 for c in cases if c.passed))
    return SelfCheckResult(
        passed=passed,
        total=total,
        elapsed_ms=float(payload.get("elapsed_ms") or 0.0),
        cases=cases,
    )


def failed_selfcheck(reason: str) -> SelfCheckResult:
    """Self-check could not run at all — treat as a hard fail."""
    return SelfCheckResult(
        passed=0,
        total=1,
        elapsed_ms=0.0,
        cases=[CaseResult("selfcheck_runner", False, reason)],
    )


def run_selfcheck() -> SelfCheckResult:
    """Run every bundled case through the real analyzer. Never raises."""
    started = time.monotonic()
    try:
        cases = [_run_named(name, fn) for name, fn in _CASES]
    except Exception as exc:  # noqa: BLE001 — never crash a scan
        elapsed = (time.monotonic() - started) * 1000.0
        result = failed_selfcheck(f"raised {type(exc).__name__}: {exc}")
        result.elapsed_ms = elapsed
        return result
    elapsed = (time.monotonic() - started) * 1000.0
    passed = sum(1 for c in cases if c.passed)
    return SelfCheckResult(
        passed=passed,
        total=len(cases),
        elapsed_ms=elapsed,
        cases=cases,
    )


def _run_named(name: str, fn: Callable[[], tuple[bool, str]]) -> CaseResult:
    try:
        ok, detail = fn()
        return CaseResult(name=name, passed=bool(ok), detail=detail)
    except Exception as exc:  # noqa: BLE001 — a raised case is a failed case
        return CaseResult(
            name=name,
            passed=False,
            detail=f"raised {type(exc).__name__}: {exc}",
        )


def _write_jsonl(path: Path, lines: list[str]) -> SessionFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    st = path.stat()
    return SessionFile(
        path=str(path),
        harness="claude_code",
        mtime=st.st_mtime,
        size_bytes=st.st_size,
    )


def _user_line(session_id: str, text: str, ts: str) -> str:
    return json.dumps(
        {
            "type": "user",
            "sessionId": session_id,
            "cwd": "/tmp/selfcheck-demo",
            "timestamp": ts,
            "message": {"role": "user", "content": text},
        }
    )


def _assistant_line(
    session_id: str,
    text: str,
    ts: str,
    *,
    model: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> str:
    rec = {
        "type": "assistant",
        "sessionId": session_id,
        "timestamp": ts,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }
    if model:
        rec["model"] = model
    if input_tokens is not None or output_tokens is not None:
        rec["usage"] = {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
        }
    return json.dumps(rec)


def _case_cluster_three_sessions() -> tuple[bool, str]:
    """A cluster of 3 intents across 3 distinct sessions must be found."""
    with tempfile.TemporaryDirectory(prefix="gh-selfcheck-") as tmp:
        root = Path(tmp)
        files = []
        for i, ask in enumerate((_ASK_A, _ASK_B, _ASK_C), 1):
            sid = f"selfcheck-across-{i}"
            files.append(
                _write_jsonl(
                    root / f"{sid}.jsonl",
                    [
                        _user_line(sid, ask, f"2026-09-0{i}T12:00:00Z"),
                        _assistant_line(
                            sid,
                            "Running.",
                            f"2026-09-0{i}T12:01:00Z",
                            model="claude-sonnet-4-20250514",
                            input_tokens=100,
                            output_tokens=20,
                        ),
                    ],
                )
            )
        parsed = parse_sessions(files)
        intents = extract_intents(parsed.sessions)
        clusters = cluster_intents(intents, min_runs=3)
    n = len(clusters)
    distinct = clusters[0].distinct_sessions if clusters else 0
    if n >= 1 and distinct >= 3:
        return True, f"{n} cluster(s), {distinct} distinct sessions"
    return (
        False,
        f"expected a cluster of 3 distinct sessions, "
        f"got {n} cluster(s) (distinct_sessions={distinct}, "
        f"intents={len(intents)})",
    )


def _case_no_cluster_one_session() -> tuple[bool, str]:
    """3 intents inside 1 session must not cluster."""
    sid = "selfcheck-one-session"
    with tempfile.TemporaryDirectory(prefix="gh-selfcheck-") as tmp:
        path = Path(tmp) / f"{sid}.jsonl"
        lines: list[str] = []
        for i, ask in enumerate((_ASK_A, _ASK_B, _ASK_C), 1):
            lines.append(_user_line(sid, ask, f"2026-09-01T12:0{i}:00Z"))
            lines.append(
                _assistant_line(
                    sid,
                    "Running.",
                    f"2026-09-01T12:0{i}:30Z",
                    model="claude-sonnet-4-20250514",
                    input_tokens=50,
                    output_tokens=10,
                )
            )
        parsed = parse_sessions([_write_jsonl(path, lines)])
        intents = extract_intents(parsed.sessions)
        clusters = cluster_intents(intents, min_runs=3)
    n_intents = len(intents)
    sessions = {m.session_id for m in intents}
    n_clusters = len(clusters)
    if n_intents < 3:
        return (
            False,
            f"expected 3 intents in 1 session, got {n_intents} "
            f"(sessions={sorted(sessions)})",
        )
    if len(sessions) != 1:
        return (
            False,
            f"expected 1 session id, got {sorted(sessions)}",
        )
    if n_clusters != 0:
        distinct = clusters[0].distinct_sessions if clusters else 0
        return (
            False,
            f"expected no cluster from 3 intents in 1 session, "
            f"got {n_clusters} cluster(s) "
            f"(distinct_sessions={distinct}, run_count="
            f"{clusters[0].run_count if clusters else 0})",
        )
    return True, f"{n_intents} intents in 1 session, 0 clusters"


def _case_malformed_jsonl() -> tuple[bool, str]:
    """A malformed JSONL line must be skipped and counted, not fatal."""
    sid = "selfcheck-malformed"
    with tempfile.TemporaryDirectory(prefix="gh-selfcheck-") as tmp:
        path = Path(tmp) / f"{sid}.jsonl"
        lines = [
            _user_line(sid, _ASK_A, "2026-09-01T12:00:00Z"),
            "THIS LINE IS TRUNCATED AND NOT VALID JSON",
            _assistant_line(
                sid,
                "Cleanup done.",
                "2026-09-01T12:01:00Z",
                model="claude-sonnet-4-20250514",
                input_tokens=40,
                output_tokens=8,
            ),
        ]
        parsed = parse_sessions([_write_jsonl(path, lines)])
    if parsed.malformed_lines < 1:
        return (
            False,
            f"expected malformed_lines >= 1, got {parsed.malformed_lines} "
            f"(sessions={len(parsed.sessions)}, skipped={parsed.skipped})",
        )
    if not parsed.sessions:
        return (
            False,
            f"malformed line was fatal: no sessions kept "
            f"(malformed_lines={parsed.malformed_lines}, "
            f"skipped={parsed.skipped})",
        )
    return (
        True,
        f"malformed_lines={parsed.malformed_lines}, "
        f"sessions={len(parsed.sessions)}",
    )


def _case_unreadable_dir_partial() -> tuple[bool, str]:
    """An unreadable directory must force PARTIAL."""
    with tempfile.TemporaryDirectory(prefix="gh-selfcheck-") as tmp:
        home = Path(tmp)
        root = home / ".claude" / "projects"
        root.mkdir(parents=True)
        locked = root / "locked-subdir"
        locked.mkdir()
        (locked / "hidden.jsonl").write_text("{}\n", encoding="utf-8")
        (root / "ok.jsonl").write_text(
            _user_line("selfcheck-ok", _ASK_A, "2026-09-01T12:00:00Z") + "\n",
            encoding="utf-8",
        )
        result = None
        used = "chmod"
        try:
            os.chmod(locked, 0o000)
            listed = True
            try:
                os.listdir(locked)
            except OSError:
                listed = False
            if listed:
                used = "not-a-directory"
                os.chmod(locked, 0o755)
                # Platform still lists chmod 000 dirs (e.g. root). A file
                # where the harness root should be is the same unreadable
                # status path in discover_harness.
                for child in root.iterdir():
                    if child.is_dir():
                        os.chmod(child, 0o755)
                _replace_root_with_file(root)
            result = discover_harness("claude_code", days=36500, home=home)
        finally:
            try:
                if locked.exists() and locked.is_dir():
                    os.chmod(locked, 0o755)
            except OSError:
                pass
        if result is None:
            return False, "discover_harness produced no result"
        status = result.sources.get("claude_code", "")
        dir_skips = [
            item
            for item in result.skipped
            if is_unreadable_dir_reason(item[1])
        ]
        unreadable = str(status).startswith("unreadable") or bool(dir_skips)
        verdict = classify_verdict(
            harness_statuses=result.sources,
            session_count=1,
            min_runs=3,
            candidate_count=3,
            skipped=list(result.skipped),
            malformed_lines=0,
            time_truncated=False,
        )
        if unreadable and verdict == VERDICT_PARTIAL:
            return (
                True,
                f"PARTIAL via {used} "
                f"(status={status!r}, dir_skips={len(dir_skips)})",
            )
        return (
            False,
            f"expected PARTIAL from unreadable directory, "
            f"got verdict={verdict!r} status={status!r} "
            f"skipped={result.skipped!r} (via {used})",
        )


def _replace_root_with_file(root: Path) -> None:
    """Turn a harness root into a file so discover reports it unreadable."""
    for child in list(root.iterdir()):
        if child.is_dir():
            for nested in child.iterdir():
                nested.unlink()
            child.rmdir()
        else:
            child.unlink()
    root.rmdir()
    root.write_text("not-a-directory\n", encoding="utf-8")


def _case_no_model_id_no_dollars() -> tuple[bool, str]:
    """A session with no model id must produce tokens and no dollar figure."""
    sid = "selfcheck-no-model"
    with tempfile.TemporaryDirectory(prefix="gh-selfcheck-") as tmp:
        path = Path(tmp) / f"{sid}.jsonl"
        parsed = parse_sessions(
            [
                _write_jsonl(
                    path,
                    [
                        _user_line(sid, _ASK_A, "2026-09-01T12:00:00Z"),
                        _assistant_line(
                            sid,
                            "Running.",
                            "2026-09-01T12:01:00Z",
                            model=None,
                            input_tokens=5000,
                            output_tokens=200,
                        ),
                    ],
                )
            ]
        )
        if not parsed.sessions:
            return False, f"parse produced no session: skipped={parsed.skipped}"
        prices = load_prices()
        cost = cost_for_session(parsed.sessions[0], prices)
    tokens = cost.input_tokens + cost.output_tokens + cost.cache_read_tokens
    if tokens <= 0:
        return (
            False,
            f"expected tokens > 0 with no model id, got tokens={tokens} "
            f"priced={cost.priced} basis={cost.basis}",
        )
    if cost.priced:
        return (
            False,
            f"expected no dollar figure (priced=False) with no model id, "
            f"got priced=True usd={cost.usd} tokens={tokens}",
        )
    return True, f"tokens={tokens} priced={cost.priced} usd={cost.usd}"


def _case_user_query_unwrapped() -> tuple[bool, str]:
    """A <user_query> wrapper must be unwrapped."""
    inner = _ASK_A
    wrapped = f"<user_query>{inner}</user_query>"
    sid = "selfcheck-unwrap"
    with tempfile.TemporaryDirectory(prefix="gh-selfcheck-") as tmp:
        path = Path(tmp) / f"{sid}.jsonl"
        rec = {
            "role": "user",
            "sessionId": sid,
            "timestamp": "2026-09-01T12:00:00Z",
            "message": {"content": wrapped},
        }
        parsed = parse_sessions(
            [_write_jsonl(path, [json.dumps(rec)])]
        )
        intents = extract_intents(parsed.sessions)
    if not parsed.sessions or not parsed.sessions[0].turns:
        return False, f"parse produced no turns: skipped={parsed.skipped}"
    turn_text = parsed.sessions[0].turns[0].text
    intent_text = intents[0].raw_text if intents else ""
    haystacks = (turn_text, intent_text)
    if any("<user_query>" in (h or "") for h in haystacks):
        return (
            False,
            f"wrapper still present: turn={turn_text!r} "
            f"intent={intent_text!r}",
        )
    if inner not in turn_text:
        return (
            False,
            f"inner ask missing after unwrap: turn={turn_text!r}",
        )
    return True, f"unwrapped to {turn_text!r}"


def _case_symlink_outside_refused() -> tuple[bool, str]:
    """A symlink resolving outside the root must be refused."""
    marker = "SELFCHECK_OUTSIDE_MARKER_not-a-secret"
    with tempfile.TemporaryDirectory(prefix="gh-selfcheck-") as tmp:
        tmp_path = Path(tmp)
        outside = tmp_path / "outside.jsonl"
        outside.write_text(
            _user_line("selfcheck-leak", marker, "2026-09-01T12:00:00Z")
            + "\n",
            encoding="utf-8",
        )
        transcripts = (
            tmp_path
            / "home"
            / ".cursor"
            / "projects"
            / "home-x-projects-Selfcheck"
            / "agent-transcripts"
        )
        transcripts.mkdir(parents=True)
        leak = transcripts / "leak.jsonl"
        leak.symlink_to(outside)
        result = discover_harness(
            "cursor", days=36500, home=tmp_path / "home"
        )
    paths = [f.path for f in result.files]
    reasons = [r for _, r in result.skipped]
    if any(str(outside) in p for p in paths):
        return (
            False,
            f"outside target was kept in files: {paths}",
        )
    if leak.name in {Path(p).name for p in paths}:
        return (
            False,
            f"escaping symlink was kept: files={paths} skipped={result.skipped}",
        )
    if SKIP_SYMLINK_OUTSIDE not in reasons:
        return (
            False,
            f"expected skip reason {SKIP_SYMLINK_OUTSIDE!r}, "
            f"got files={paths} skipped={result.skipped}",
        )
    return True, f"refused ({SKIP_SYMLINK_OUTSIDE})"


# Ordered cases. Names are stable — tests and failure copy depend on them.
_CASES: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
    ("cluster_three_sessions", _case_cluster_three_sessions),
    ("no_cluster_one_session", _case_no_cluster_one_session),
    ("malformed_jsonl_skipped", _case_malformed_jsonl),
    ("unreadable_dir_partial", _case_unreadable_dir_partial),
    ("no_model_id_no_dollars", _case_no_model_id_no_dollars),
    ("user_query_unwrapped", _case_user_query_unwrapped),
    ("symlink_outside_refused", _case_symlink_outside_refused),
]

CASE_NAMES = tuple(name for name, _ in _CASES)


def fixture_blob() -> str:
    """Concatenation of bundled fixture text — for credential scans in tests."""
    return "\n".join(
        [
            _ASK_A,
            _ASK_B,
            _ASK_C,
            "THIS LINE IS TRUNCATED AND NOT VALID JSON",
            "SELFCHECK_OUTSIDE_MARKER_not-a-secret",
            "<user_query>",
            "</user_query>",
            SKIP_UNREADABLE_DIR,
        ]
    )
