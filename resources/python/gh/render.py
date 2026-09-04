"""Render text and JSON reports from ranked candidates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from gh import SCHEMA_VERSION
from gh.cost import ProjectCost
from gh.discover import (
    SKIP_SYMLINK_OUTSIDE,
    SKIP_UNREADABLE_DIR,
    is_unreadable_dir_reason,
)
from gh.rank import Candidate, RankResult
from gh.context_rediscovery import RediscoveryReport

# Friendly harness labels for strangers (never leak snake_case internals).
_HARNESS_LABELS = {
    "claude_code": "Claude Code",
    "codex": "Codex",
    "cursor": "Cursor",
}

_BASIS_LABELS = {
    "measured": "from your logs",
    "estimated": "estimated",
    "unknown": "unknown",
}

VERDICT_REPEATED = "REPEATED WORK FOUND"
VERDICT_NULL = "DEFENSIBLE NULL"
VERDICT_PARTIAL = "PARTIAL SCAN — NOT A CLEAN RESULT"
VERDICT_INSUFFICIENT = "INSUFFICIENT HISTORY"
VERDICT_NO_HISTORY = "NO SUPPORTED HISTORY FOUND"


@dataclass
class EvidenceView:
    raw_text: str
    date: str
    project: str
    session_id: str


@dataclass
class CandidateView:
    rank: int
    label: str
    run_count: int
    distinct_sessions: int
    first_seen: str
    last_seen: str
    projects: list[str]
    tokens: int
    usd: float
    basis: str
    stability_sentence: str
    evidence: list[EvidenceView]
    # JSON-only extras
    cluster_id: str = ""
    score: float = 0.0
    frequency: float = 0.0
    cost_score: float = 0.0
    stability: float = 0.0
    recency_days: Optional[float] = None
    session_ids: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    priced: bool = False


@dataclass
class ProjectView:
    project: str
    tokens: int
    usd: float
    basis: str
    run_count: int = 0
    candidates: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    priced: bool = False


@dataclass
class Coverage:
    """What the scan actually touched — shown under the verdict."""

    directories_checked: int = 0
    agents_detected: list[str] = field(default_factory=list)
    files_discovered: int = 0
    files_parsed: int = 0
    files_skipped: int = 0
    sessions_analyzed: int = 0
    tool_calls_analyzed: int = 0
    date_range: str = "none"
    sessions_with_tokens: int = 0
    threshold: int = 0


@dataclass
class Report:
    """Shared data model for text and JSON output."""

    days: int
    min_runs: int
    top: int
    session_count: int
    harnesses_found: list[str]
    harness_statuses: dict[str, str]
    candidates: list[CandidateView]
    projects: list[ProjectView]
    not_counted: list[str]
    schema_version: str = SCHEMA_VERSION
    total_candidates: int = 0
    sessions_skipped: list[tuple[str, str]] = field(default_factory=list)
    malformed_lines: int = 0
    locations_checked: list[str] = field(default_factory=list)
    zero_sessions: bool = False
    redacted: bool = True
    verdict: str = VERDICT_NO_HISTORY
    coverage: Coverage = field(default_factory=Coverage)
    sessions_without_model: int = 0
    rediscovery: Optional[RediscoveryReport] = None


def build_report(
    *,
    days: int,
    min_runs: int,
    top: int,
    session_count: int,
    harness_statuses: dict[str, str],
    rank_result: RankResult,
    skipped: list[tuple[str, str]],
    malformed_lines: int = 0,
    extra_not_counted: list[str] | None = None,
    locations_checked: list[str] | None = None,
    redact: bool = True,
    time_truncated: bool = False,
    files_read: int = 0,
    files_total: int = 0,
    session_projects: list[ProjectView] | None = None,
    tool_calls: int = 0,
    sessions_with_tokens: int = 0,
    sessions_without_model: int = 0,
    date_range: str = "none",
    rediscovery: Optional[RediscoveryReport] = None,
) -> Report:
    """Assemble the shared report model from pipeline outputs."""
    from gh.redact import EVIDENCE_LIMIT, redact_text

    found = [
        _harness_label(name)
        for name, status in harness_statuses.items()
        if status == "found"
    ]
    all_candidates = list(rank_result.candidates)
    shown = all_candidates[: max(0, top)]

    views = [
        _candidate_view(i, cand, redact=redact, evidence_limit=EVIDENCE_LIMIT)
        for i, cand in enumerate(shown, 1)
    ]
    if redact:
        for view in views:
            view.label = redact_text(view.label, limit=None)

    projects = _projects_from_candidates(shown)
    if not projects and session_projects:
        projects = list(session_projects)
    not_counted = _not_counted_lines(
        harness_statuses=harness_statuses,
        skipped=skipped,
        malformed_lines=malformed_lines,
        all_candidates=all_candidates,
        shown_count=len(shown),
        top=top,
        candidates_shown=shown,
        extra=(extra_not_counted or []) + list(
            (rediscovery.notes if rediscovery else [])
        ),
        time_truncated=time_truncated,
        files_read=files_read,
        files_total=files_total,
        session_projects=projects if not shown else None,
        sessions_without_model=sessions_without_model,
    )

    real_skipped = _real_skipped(skipped)
    unread = max(0, files_total - files_read) if time_truncated else 0
    symlink_skips = [
        item for item in real_skipped if item[1] == SKIP_SYMLINK_OUTSIDE
    ]
    dir_skips = [
        item for item in real_skipped if is_unreadable_dir_reason(item[1])
    ]
    opened_skips = [
        item
        for item in real_skipped
        if item[1] != SKIP_SYMLINK_OUTSIDE
        and not is_unreadable_dir_reason(item[1])
    ]
    files_skipped = len(real_skipped) - len(dir_skips) + unread
    files_parsed = max(0, files_read - len(opened_skips))
    coverage = Coverage(
        directories_checked=len(locations_checked or []),
        agents_detected=found,
        files_discovered=files_total + len(symlink_skips),
        files_parsed=files_parsed,
        files_skipped=files_skipped,
        sessions_analyzed=session_count,
        tool_calls_analyzed=tool_calls,
        date_range=date_range or "none",
        sessions_with_tokens=sessions_with_tokens,
        threshold=min_runs,
    )
    verdict = classify_verdict(
        harness_statuses=harness_statuses,
        session_count=session_count,
        min_runs=min_runs,
        candidate_count=len(all_candidates),
        skipped=skipped,
        malformed_lines=malformed_lines,
        time_truncated=time_truncated,
        extra_notes=extra_not_counted or [],
    )

    return Report(
        days=days,
        min_runs=min_runs,
        top=top,
        session_count=session_count,
        harnesses_found=found,
        harness_statuses=dict(harness_statuses),
        candidates=views,
        projects=projects,
        not_counted=not_counted,
        total_candidates=len(all_candidates),
        sessions_skipped=list(skipped),
        malformed_lines=malformed_lines,
        locations_checked=list(locations_checked or []),
        zero_sessions=session_count == 0,
        redacted=redact,
        verdict=verdict,
        coverage=coverage,
        sessions_without_model=sessions_without_model,
        rediscovery=rediscovery,
    )


def classify_verdict(
    *,
    harness_statuses: dict[str, str],
    session_count: int,
    min_runs: int,
    candidate_count: int,
    skipped: list[tuple[str, str]],
    malformed_lines: int,
    time_truncated: bool,
    extra_notes: list[str] | None = None,
) -> str:
    """Pick the honest scan class. Partial never renders as a null."""
    statuses = list(harness_statuses.values()) if harness_statuses else []
    if statuses and all(status == "absent" for status in statuses):
        return VERDICT_NO_HISTORY
    if not statuses:
        return VERDICT_NO_HISTORY
    if _is_partial(
        harness_statuses=harness_statuses,
        skipped=skipped,
        malformed_lines=malformed_lines,
        time_truncated=time_truncated,
        extra_notes=extra_notes or [],
    ):
        return VERDICT_PARTIAL
    if session_count < min_runs:
        return VERDICT_INSUFFICIENT
    if candidate_count > 0:
        return VERDICT_REPEATED
    return VERDICT_NULL


def _is_partial(
    *,
    harness_statuses: dict[str, str],
    skipped: list[tuple[str, str]],
    malformed_lines: int,
    time_truncated: bool,
    extra_notes: list[str],
) -> bool:
    for status in harness_statuses.values():
        if str(status).startswith("unreadable"):
            return True
    if _real_skipped(skipped):
        return True
    if malformed_lines > 0:
        return True
    if time_truncated:
        return True
    for note in extra_notes:
        if note and "failed" in note.lower():
            return True
    return False


def _real_skipped(skipped: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [
        (path, reason)
        for path, reason in skipped
        if path != "(remaining files)" and "time budget hit" not in reason
    ]


def _coverage_lines(coverage: Coverage) -> list[str]:
    agents = ", ".join(coverage.agents_detected) or "none"
    rows = [
        ("directories checked", str(coverage.directories_checked)),
        ("agents detected", agents),
        ("files discovered", str(coverage.files_discovered)),
        ("files parsed", str(coverage.files_parsed)),
        ("files skipped", str(coverage.files_skipped)),
        ("sessions analyzed", str(coverage.sessions_analyzed)),
        ("tool calls analyzed", f"{coverage.tool_calls_analyzed:,}"),
        ("date range covered", coverage.date_range),
        ("sessions with token counts", str(coverage.sessions_with_tokens)),
        (
            "threshold used",
            f"{coverage.threshold} distinct session"
            f"{'s' if coverage.threshold != 1 else ''}",
        ),
    ]
    width = max(len(label) for label, _ in rows)
    lines = ["COVERAGE"]
    for label, value in rows:
        lines.append(f"  {label.ljust(width)}  {value}")
    return lines


def _rediscovery_lines(report: Report) -> list[str]:
    """Always emit the rediscovery section, including on a defensible null."""
    from gh.redact import redact_text

    rd = report.rediscovery
    lines = ["THE WORK YOUR AGENT REDOES EVERY SESSION"]
    if rd is None:
        lines.append("  (not measured)")
        lines.append("")
        return lines
    if rd.sessions_with_tools == 0 and not rd.harnesses_excluded:
        if rd.notes and "no sessions" in rd.notes[0]:
            lines.append("  No sessions to measure.")
        else:
            lines.append("  No tool-use blocks in this window.")
        lines.append("")
        return lines
    if not rd.sufficient:
        n = rd.resolvable_sessions
        lines.append(
            f"  {n} session{'s' if n != 1 else ''} had a first edit — "
            "not enough to report rates (need 5)."
        )
        for note in rd.notes:
            lines.append(f"  {note[0].upper() + note[1:]}" if note else "")
        lines.append("")
        return lines

    pct = _fmt_pct(rd.pattern_pct)
    median = _fmt_count(rd.median_prefix)
    p90 = _fmt_count(rd.p90_prefix)
    explore = _fmt_pct(rd.explore_pct)
    lines.append(
        f"  {pct} of sessions begin by re-deriving the same context"
    )
    lines.append(
        f"  Median {median} tool calls before the first edit (p90 {p90})"
    )
    lines.append(
        f"  {explore} of all tool calls happen before any change is made"
    )
    if rd.top_files:
        lines.append("  Files re-read across the most sessions:")
        shown = [
            (_display_path(item.path, redact=report.redacted), item.sessions)
            for item in rd.top_files
        ]
        width = max(len(path) for path, _ in shown)
        for path, count in shown:
            if report.redacted:
                path = redact_text(path, limit=None)
            lines.append(
                f"    {path.ljust(width)}  re-read in {count} session"
                f"{'s' if count != 1 else ''}"
            )
    if rd.per_project:
        bits = []
        for proj in rd.per_project[:8]:
            name = proj.project
            if report.redacted:
                name = redact_text(name, limit=None)
            bits.append(
                f"{name} {proj.sessions} session"
                f"{'s' if proj.sessions != 1 else ''}, "
                f"median {proj.median_prefix} calls to first edit"
            )
        lines.append("  Per project: " + "; ".join(bits))
    if rd.no_mutation_sessions:
        n = rd.no_mutation_sessions
        lines.append(
            f"  {n} session{'s' if n != 1 else ''} never made an edit "
            "(not folded into the median)"
        )
    lines.append("")
    return lines


def _rediscovery_json(rd: Optional[RediscoveryReport]) -> Optional[dict[str, Any]]:
    if rd is None:
        return None
    return {
        "resolvable_sessions": rd.resolvable_sessions,
        "no_mutation_sessions": rd.no_mutation_sessions,
        "sessions_with_tools": rd.sessions_with_tools,
        "harnesses_excluded": rd.harnesses_excluded,
        "sufficient": rd.sufficient,
        "pattern_pct": rd.pattern_pct,
        "median_prefix": rd.median_prefix,
        "p90_prefix": rd.p90_prefix,
        "explore_pct": rd.explore_pct,
        "top_files": [
            {"path": f.path, "sessions": f.sessions} for f in rd.top_files
        ],
        "per_project": [
            {
                "project": p.project,
                "sessions": p.sessions,
                "median_prefix": p.median_prefix,
            }
            for p in rd.per_project
        ],
        "notes": rd.notes,
    }


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{int(round(value))}%"


def _fmt_count(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return str(int(round(value)))


def _display_path(path: str, *, redact: bool = True) -> str:
    text = (path or "").replace("\\", "/")
    marker = "/projects/"
    if marker in text:
        text = text.split(marker, 1)[1]
    else:
        parts = [p for p in text.split("/") if p]
        if len(parts) > 3:
            text = "/".join(parts[-3:])
    return text


def render_text(report: Report) -> str:
    """Stranger-facing terminal report. Keep it to one screen."""
    from gh.redact import redact_report_strings

    harnesses = (
        " · ".join(report.harnesses_found)
        if report.harnesses_found
        else "no tools found"
    )
    lines: list[str] = [
        f"GROUNDHOG · {report.session_count} sessions · "
        f"last {report.days} days · {harnesses}",
        "",
        report.verdict,
        "",
    ]
    lines.extend(_coverage_lines(report.coverage))
    lines.append("")
    lines.extend(_rediscovery_lines(report))
    lines.append("YOU KEEP REDOING THIS")

    if report.zero_sessions:
        lines.append(
            f"No session files found in the last {report.days} days."
        )
        if report.locations_checked:
            lines.append("Checked:")
            for loc in report.locations_checked:
                lines.append(f"  {loc}")
    elif not report.candidates:
        lines.append(_empty_repeat_message(report.days, report.min_runs))
    else:
        for cand in report.candidates:
            projects = ", ".join(cand.projects) or "unknown"
            lines.append("")
            lines.append(f"{cand.rank}. {cand.label}")
            lines.append(
                f"   {_format_timespan(cand.distinct_sessions, cand.first_seen, cand.last_seen)}"
                f" · {projects}"
            )
            token_line = f"   ~{_fmt_tokens(cand.tokens)} tokens"
            if cand.priced:
                token_line += (
                    f" · ~${cand.usd:.2f} ({_basis_label(cand.basis)})"
                )
            lines.append(token_line)
            lines.append(f"   {cand.stability_sentence}")
            if cand.evidence:
                lines.append("   Seen as:")
                for ev in cand.evidence[:2]:
                    quote = _quote(ev.raw_text, 64)
                    lines.append(f'     "{quote}"   {ev.date}')

    lines.append("")
    lines.append("WHERE YOUR TOKENS WENT")
    if report.projects:
        name_width = max(len(p.project) for p in report.projects)
        token_width = max(
            len(_fmt_tokens(p.tokens)) for p in report.projects
        )
        for proj in report.projects:
            row = (
                f"  {proj.project.ljust(name_width)}   "
                f"{_fmt_tokens(proj.tokens).rjust(token_width)}"
            )
            if proj.priced:
                row += f"   ${proj.usd:.2f}   {_basis_label(proj.basis)}"
            lines.append(row)
    else:
        lines.append("  (no project totals in this window)")

    lines.append("")
    lines.append("NOT COUNTED")
    for item in report.not_counted:
        lines.append(f"  · {item}")
    lines.append("")
    lines.append(
        "Local only · read your session files · wrote nothing · sent nothing"
    )
    lines.append("")
    text = "\n".join(lines)
    if report.redacted:
        text = redact_report_strings(text)
    return text


def render_json(report: Report) -> str:
    """JSON mirror of the text report, plus scores and session ids."""
    payload: dict[str, Any] = {
        "schema_version": report.schema_version,
        "days": report.days,
        "min_runs": report.min_runs,
        "top": report.top,
        "session_count": report.session_count,
        "harnesses_found": report.harnesses_found,
        "harness_statuses": {
            _harness_label(k): v for k, v in report.harness_statuses.items()
        },
        "candidates": [
            {
                "rank": c.rank,
                "label": c.label,
                "run_count": c.run_count,
                "distinct_sessions": c.distinct_sessions,
                "first_seen": c.first_seen,
                "last_seen": c.last_seen,
                "projects": c.projects,
                "tokens": c.tokens,
                "usd": round(c.usd, 6) if c.priced else None,
                "priced": c.priced,
                "basis": c.basis,
                "evidence": [asdict(ev) for ev in c.evidence],
                "cluster_id": c.cluster_id,
                "score": c.score,
                "components": {
                    "frequency": c.frequency,
                    "cost": c.cost_score,
                    "stability": c.stability,
                },
                "recency_days": c.recency_days,
                "session_ids": c.session_ids,
                "token_detail": {
                    "input": c.input_tokens,
                    "output": c.output_tokens,
                    "cache_read": c.cache_read_tokens,
                    "basis": c.basis,
                },
            }
            for c in report.candidates
        ],
        "projects": [
            {
                "project": p.project,
                "tokens": p.tokens,
                "usd": round(p.usd, 6) if p.priced else None,
                "priced": p.priced,
                "basis": p.basis,
                "run_count": p.run_count,
                "candidates": p.candidates,
                "token_detail": {
                    "input": p.input_tokens,
                    "output": p.output_tokens,
                    "cache_read": p.cache_read_tokens,
                },
            }
            for p in report.projects
        ],
        "not_counted": report.not_counted,
        "total_candidates": report.total_candidates,
        "empty": not bool(report.candidates),
        "zero_sessions": report.zero_sessions,
        "locations_checked": report.locations_checked,
        "redacted": report.redacted,
        "verdict": report.verdict,
        "coverage": {
            "directories_checked": report.coverage.directories_checked,
            "agents_detected": report.coverage.agents_detected,
            "files_discovered": report.coverage.files_discovered,
            "files_parsed": report.coverage.files_parsed,
            "files_skipped": report.coverage.files_skipped,
            "sessions_analyzed": report.coverage.sessions_analyzed,
            "tool_calls_analyzed": report.coverage.tool_calls_analyzed,
            "date_range": report.coverage.date_range,
            "sessions_with_tokens": report.coverage.sessions_with_tokens,
            "threshold": report.coverage.threshold,
        },
        "rediscovery": _rediscovery_json(report.rediscovery),
    }
    text = json.dumps(payload, indent=2) + "\n"
    if report.redacted:
        from gh.redact import redact_report_strings

        text = redact_report_strings(text)
    return text


def stability_sentence(stability: float) -> str:
    """Translate a 0–1 stability score into plain language."""
    if stability >= 0.85:
        return "Solved the same way every time."
    if stability >= 0.5:
        return "Went smoothly most times."
    return "Varied a lot between runs."


def _candidate_view(
    rank: int,
    cand: Candidate,
    *,
    redact: bool = True,
    evidence_limit: int = 120,
) -> CandidateView:
    from gh.redact import redact_text

    tokens = cand.input_tokens + cand.output_tokens + cand.cache_read_tokens
    evidence = []
    for ev in cand.evidence:
        raw = ev.raw_text
        if redact:
            raw = redact_text(raw, limit=evidence_limit)
        evidence.append(
            EvidenceView(
                raw_text=raw,
                date=_short_date(ev.timestamp),
                project=ev.project,
                session_id=ev.session_id,
            )
        )
    label = cand.label
    if redact:
        label = redact_text(label, limit=None)
    return CandidateView(
        rank=rank,
        label=label,
        run_count=cand.run_count,
        distinct_sessions=cand.distinct_sessions,
        first_seen=_short_date(cand.first_seen),
        last_seen=_short_date(cand.last_seen),
        projects=sorted(cand.projects),
        tokens=tokens,
        usd=cand.usd,
        basis=cand.cost_basis,
        stability_sentence=stability_sentence(cand.stability),
        evidence=evidence,
        cluster_id=cand.cluster_id,
        score=cand.score,
        frequency=cand.frequency,
        cost_score=cand.cost_score,
        stability=cand.stability,
        recency_days=cand.recency_days,
        session_ids=list(cand.session_ids),
        input_tokens=cand.input_tokens,
        output_tokens=cand.output_tokens,
        cache_read_tokens=cand.cache_read_tokens,
        priced=cand.priced,
    )


def _projects_from_candidates(candidates: list[Candidate]) -> list[ProjectView]:
    """Roll up tokens/usd for projects touched by the shown candidates."""
    by_project: dict[str, ProjectView] = {}
    bases: dict[str, set[str]] = {}
    priced_flags: dict[str, list[bool]] = {}
    for cand in candidates:
        projects = sorted(cand.projects) or ["unknown"]
        for project in projects:
            view = by_project.get(project)
            if view is None:
                view = ProjectView(
                    project=project, tokens=0, usd=0.0, basis="unknown"
                )
                by_project[project] = view
            view.candidates += 1
            view.run_count += cand.distinct_sessions
            view.input_tokens += cand.input_tokens
            view.output_tokens += cand.output_tokens
            view.cache_read_tokens += cand.cache_read_tokens
            view.usd += cand.usd
            view.tokens = (
                view.input_tokens
                + view.output_tokens
                + view.cache_read_tokens
            )
            bases.setdefault(project, set()).add(cand.cost_basis)
            priced_flags.setdefault(project, []).append(cand.priced)
    for project, view in by_project.items():
        view.basis = _rollup_basis(bases.get(project, set()))
        flags = priced_flags.get(project, [])
        view.priced = bool(flags) and all(flags)
    return sorted(
        by_project.values(),
        key=lambda p: (-p.usd, -p.tokens, p.project.lower()),
    )


def projects_from_session_costs(costs: list[ProjectCost]) -> list[ProjectView]:
    """Turn all-session project costs into the report's project rows."""
    views: list[ProjectView] = []
    for cost in costs:
        views.append(
            ProjectView(
                project=cost.project,
                tokens=cost.tokens,
                usd=cost.usd,
                basis=cost.basis,
                run_count=cost.session_count,
                candidates=0,
                input_tokens=cost.input_tokens,
                output_tokens=cost.output_tokens,
                cache_read_tokens=cost.cache_read_tokens,
                priced=cost.priced,
            )
        )
    return sorted(
        views,
        key=lambda p: (-p.usd, -p.tokens, p.project.lower()),
    )


def _not_counted_lines(
    *,
    harness_statuses: dict[str, str],
    skipped: list[tuple[str, str]],
    malformed_lines: int,
    all_candidates: list[Candidate],
    shown_count: int,
    top: int,
    candidates_shown: list[Candidate],
    extra: list[str] | None = None,
    time_truncated: bool = False,
    files_read: int = 0,
    files_total: int = 0,
    session_projects: list[ProjectView] | None = None,
    sessions_without_model: int = 0,
) -> list[str]:
    items: list[str] = []

    for name, status in harness_statuses.items():
        label = _harness_label(name)
        if status == "absent":
            items.append(f"{label} history not found on this machine")
        elif status.startswith("unreadable"):
            reason = status.split(":", 1)[-1].strip() or status
            items.append(f"{label} history unreadable ({reason})")

    # Skip the synthetic time-budget skip — surfaced via time_truncated note.
    real_skipped = [
        (p, r)
        for p, r in skipped
        if p != "(remaining files)" and "time budget hit" not in r
    ]
    symlink_n = sum(1 for _, r in real_skipped if r == SKIP_SYMLINK_OUTSIDE)
    dir_skips = [
        (p, r) for p, r in real_skipped if is_unreadable_dir_reason(r)
    ]
    other_skipped = [
        (p, r)
        for p, r in real_skipped
        if r != SKIP_SYMLINK_OUTSIDE and not is_unreadable_dir_reason(r)
    ]
    if symlink_n:
        items.append(
            f"{symlink_n} file{'s' if symlink_n != 1 else ''} skipped "
            f"({SKIP_SYMLINK_OUTSIDE})"
        )
    if dir_skips:
        n = len(dir_skips)
        noun = "directory" if n == 1 else "directories"
        if n == 1:
            path, reason = dir_skips[0]
            items.append(f"1 {noun} skipped ({reason}): {path}")
        else:
            items.append(f"{n} {noun} skipped ({SKIP_UNREADABLE_DIR})")
            for path, reason in dir_skips:
                items.append(f"{path} ({reason})")
    if other_skipped:
        if len(other_skipped) == 1:
            _path, reason = other_skipped[0]
            items.append(f"1 session file skipped ({reason})")
        else:
            items.append(f"{len(other_skipped)} session files skipped")

    if malformed_lines:
        items.append(
            f"{malformed_lines} malformed line"
            f"{'s' if malformed_lines != 1 else ''} skipped while reading"
        )

    if time_truncated:
        unread = max(0, files_total - files_read)
        items.append(
            f"stopped early after ~20s "
            f"(read {files_read} of {files_total} files"
            f"{f'; {unread} unread' if unread else ''})"
        )

    unknown = [c for c in candidates_shown if c.cost_basis == "unknown"]
    if not unknown and session_projects:
        unknown = [p for p in session_projects if p.basis == "unknown"]
    if unknown:
        items.append(
            f"{len(unknown)} cost figure"
            f"{'s' if len(unknown) != 1 else ''} unknown "
            "(no tokens and no text to estimate from)"
        )

    if sessions_without_model:
        items.append(
            f"{sessions_without_model} session"
            f"{'s' if sessions_without_model != 1 else ''} had no model id "
            "and therefore no cost"
        )

    hidden = max(0, len(all_candidates) - shown_count)
    if hidden:
        items.append(
            f"{hidden} more repeated chore"
            f"{'s' if hidden != 1 else ''} hidden "
            f"(showing top {top}; pass --top {top + hidden} to see them)"
        )

    for note in extra or []:
        if note and note not in items:
            items.append(note)

    if not items:
        items.append("nothing skipped")
    return items


def _empty_repeat_message(days: int, min_runs: int) -> str:
    """Empty-state copy that matches the window and threshold actually used."""
    core = (
        f"No chore repeated across {min_runs}+ separate sessions "
        f"in the last {days} days."
    )
    parts = [core]
    if days >= 30:
        parts.append(
            "Groundhog needs several months of history before "
            "patterns emerge for most people."
        )
    hints: list[str] = []
    # Never suggest a smaller-or-equal window, or a higher-or-equal threshold.
    if 30 > days:
        hints.append("--days 30")
    if 2 < min_runs:
        hints.append("--min-runs 2")
    if hints:
        parts.append(f"Try {' or '.join(hints)}.")
    return " ".join(parts)


def _format_timespan(
    distinct_sessions: int, first_seen: str, last_seen: str
) -> str:
    """Same calendar day → ``N times on DATE``; else an arrow range.

    ``N`` is distinct sessions, not turns inside one conversation.
    """
    if first_seen and last_seen and first_seen == last_seen:
        return f"{distinct_sessions} times on {first_seen}"
    return f"{distinct_sessions} times · {first_seen} → {last_seen}"


def _harness_label(name: str) -> str:
    return _HARNESS_LABELS.get(name, name.replace("_", " ").title())


def _basis_label(basis: str) -> str:
    return _BASIS_LABELS.get(basis, basis)


def _rollup_basis(bases: set[str]) -> str:
    if not bases:
        return "unknown"
    if bases == {"measured"}:
        return "measured"
    if "unknown" in bases and len(bases) == 1:
        return "unknown"
    if "estimated" in bases or "unknown" in bases:
        return "estimated" if "estimated" in bases else "unknown"
    return next(iter(bases))


def _short_date(value: Optional[str]) -> str:
    if not value:
        return "unknown"
    # ISO timestamps → YYYY-MM-DD
    text = value.strip()
    if "T" in text:
        return text.split("T", 1)[0]
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return text


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _quote(text: str, limit: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"
