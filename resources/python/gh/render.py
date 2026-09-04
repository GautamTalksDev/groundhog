"""Render text and JSON reports from ranked candidates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from gh import SCHEMA_VERSION
from gh.rank import Candidate, RankResult

# Friendly harness labels for strangers (never leak snake_case internals).
_HARNESS_LABELS = {
    "claude_code": "Claude Code",
    "codex": "Codex",
}

_BASIS_LABELS = {
    "measured": "from your logs",
    "estimated": "estimated",
    "unknown": "unknown",
}


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
    not_counted = _not_counted_lines(
        harness_statuses=harness_statuses,
        skipped=skipped,
        malformed_lines=malformed_lines,
        all_candidates=all_candidates,
        shown_count=len(shown),
        top=top,
        candidates_shown=shown,
        extra=extra_not_counted or [],
        time_truncated=time_truncated,
        files_read=files_read,
        files_total=files_total,
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
    )


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
        "YOU KEEP REDOING THIS",
    ]

    if report.zero_sessions:
        lines.append(
            f"No session files found in the last {report.days} days."
        )
        if report.locations_checked:
            lines.append("Checked:")
            for loc in report.locations_checked:
                lines.append(f"  {loc}")
    elif not report.candidates:
        lines.append(
            f"Nothing you've repeated {report.min_runs}+ times in this "
            "window. Try --days 30 or --min-runs 2."
        )
    else:
        for cand in report.candidates:
            projects = ", ".join(cand.projects) or "unknown"
            lines.append("")
            lines.append(f"{cand.rank}. {cand.label}")
            lines.append(
                f"   {cand.run_count} times · "
                f"{cand.first_seen} → {cand.last_seen} · {projects}"
            )
            lines.append(
                f"   ~{_fmt_tokens(cand.tokens)} tokens · "
                f"~${cand.usd:.2f} ({_basis_label(cand.basis)})"
            )
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
            lines.append(
                f"  {proj.project.ljust(name_width)}   "
                f"{_fmt_tokens(proj.tokens).rjust(token_width)}   "
                f"${proj.usd:.2f}   {_basis_label(proj.basis)}"
            )
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
                "first_seen": c.first_seen,
                "last_seen": c.last_seen,
                "projects": c.projects,
                "tokens": c.tokens,
                "usd": round(c.usd, 6),
                "basis": c.basis,
                "stability_sentence": c.stability_sentence,
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
                "usd": round(p.usd, 6),
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
    )


def _projects_from_candidates(candidates: list[Candidate]) -> list[ProjectView]:
    """Roll up tokens/usd for projects touched by the shown candidates."""
    by_project: dict[str, ProjectView] = {}
    bases: dict[str, set[str]] = {}
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
            view.run_count += cand.run_count
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
    for project, view in by_project.items():
        view.basis = _rollup_basis(bases.get(project, set()))
    return sorted(
        by_project.values(),
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
    if real_skipped:
        if len(real_skipped) == 1:
            _path, reason = real_skipped[0]
            items.append(f"1 session file skipped ({reason})")
        else:
            items.append(f"{len(real_skipped)} session files skipped")

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

    estimated = [c for c in candidates_shown if c.cost_basis == "estimated"]
    if estimated:
        items.append(
            f"{len(estimated)} cost figure"
            f"{'s' if len(estimated) != 1 else ''} estimated "
            "(token counts were missing from the file)"
        )

    unknown = [c for c in candidates_shown if c.cost_basis == "unknown"]
    if unknown:
        items.append(
            f"{len(unknown)} cost figure"
            f"{'s' if len(unknown) != 1 else ''} unknown "
            "(no tokens and no text to estimate from)"
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
