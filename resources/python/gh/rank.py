"""Score and rank Play candidates from clustered intents."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from gh.cluster import Cluster
from gh.cost import CostBreakdown


@dataclass
class EvidenceItem:
    """One example ask that landed in this candidate's cluster."""

    raw_text: str
    timestamp: Optional[str]
    project: str
    session_id: str


@dataclass
class Candidate:
    """A ranked Play candidate with scored components and evidence."""

    cluster_id: str
    label: str
    score: float
    frequency: float
    cost_score: float
    stability: float
    run_count: int
    distinct_sessions: int
    usd: float
    cost_basis: str
    recency_days: Optional[float]
    projects: set[str]
    evidence: list[EvidenceItem]
    session_ids: list[str]
    first_seen: Optional[str]
    last_seen: Optional[str]
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    priced: bool = False


@dataclass
class ProjectRollup:
    """Per-project token/usd totals across ranked candidates."""

    project: str
    candidates: int = 0
    run_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    usd: float = 0.0


@dataclass
class RankResult:
    """Ranked candidates plus per-project rollup for the report."""

    candidates: list[Candidate] = field(default_factory=list)
    project_rollups: list[ProjectRollup] = field(default_factory=list)


def score_candidates(
    clusters: list[Cluster],
    costs: list[CostBreakdown],
    *,
    now: Optional[datetime] = None,
) -> RankResult:
    """Rank clusters by frequency × cost × stability (with recency decay)."""
    if not clusters:
        return RankResult()
    if len(costs) != len(clusters):
        # Degrade: pad/truncate rather than crash.
        paired = list(zip(clusters, costs))
    else:
        paired = list(zip(clusters, costs))

    now = now or datetime.now(timezone.utc)

    session_counts = [max(1, c.distinct_sessions) for c, _ in paired]
    max_log_sessions = math.log(max(session_counts)) if session_counts else 1.0
    if max_log_sessions <= 0:
        max_log_sessions = 1.0

    cost_magnitudes = [_cost_magnitude(cost) for _, cost in paired]
    max_log_cost = (
        math.log(max(cost_magnitudes) + 1.0) if cost_magnitudes else 1.0
    )
    if max_log_cost <= 0:
        max_log_cost = 1.0

    turn_variances = [_turn_variance(c) for c, _ in paired]
    max_var = max(turn_variances) if turn_variances else 0.0

    candidates: list[Candidate] = []
    for (cluster, cost), variance in zip(paired, turn_variances):
        frequency = math.log(max(1, cluster.distinct_sessions)) / max_log_sessions
        cost_score = math.log(_cost_magnitude(cost) + 1.0) / max_log_cost
        if max_var > 0:
            stability = 1.0 - (variance / max_var)
        else:
            stability = 1.0
        # Clamp to [0, 1].
        frequency = _clamp01(frequency)
        cost_score = _clamp01(cost_score)
        stability = _clamp01(stability)

        score = frequency * cost_score * stability
        recency_days = _recency_days(cluster.last_seen, now)
        if recency_days is not None and recency_days > 7:
            score *= 0.5

        evidence = _evidence(cluster)
        session_ids = sorted(
            {m.session_id for m in cluster.members if m.session_id}
        )
        candidates.append(
            Candidate(
                cluster_id=cluster.id,
                label=cluster.label,
                score=round(score, 6),
                frequency=round(frequency, 6),
                cost_score=round(cost_score, 6),
                stability=round(stability, 6),
                run_count=cluster.run_count,
                distinct_sessions=cluster.distinct_sessions,
                usd=cost.usd,
                cost_basis=cost.basis,
                recency_days=recency_days,
                projects=set(cluster.projects),
                evidence=evidence,
                session_ids=session_ids,
                first_seen=cluster.first_seen,
                last_seen=cluster.last_seen,
                input_tokens=cost.input_tokens,
                output_tokens=cost.output_tokens,
                cache_read_tokens=cost.cache_read_tokens,
                priced=cost.priced,
            )
        )

    candidates.sort(
        key=lambda c: (
            -c.score,
            -c.distinct_sessions,
            -c.run_count,
            -c.usd,
            c.label.lower(),
        )
    )
    rollups = _project_rollups(candidates)
    return RankResult(candidates=candidates, project_rollups=rollups)


def _cost_magnitude(cost: CostBreakdown) -> float:
    """Prefer USD; if unknown/zero with no dollars, fall back to token count."""
    if cost.basis == "unknown" or (cost.usd <= 0 and cost.basis != "measured"):
        tokens = (
            cost.input_tokens + cost.output_tokens + cost.cache_read_tokens
        )
        if tokens > 0:
            return float(tokens)
        return 0.0
    if cost.usd > 0:
        return float(cost.usd)
    tokens = cost.input_tokens + cost.output_tokens + cost.cache_read_tokens
    return float(tokens) if tokens > 0 else 0.0


def _turn_variance(cluster: Cluster) -> float:
    values = [float(m.session_turn_count) for m in cluster.members]
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _recency_days(last_seen: Optional[str], now: datetime) -> Optional[float]:
    if not last_seen:
        return None
    parsed = _parse_ts(last_seen)
    if parsed is None:
        return None
    delta = now - parsed
    return max(0.0, delta.total_seconds() / 86400.0)


def _parse_ts(value: str) -> Optional[datetime]:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _evidence(cluster: Cluster, limit: int = 3) -> list[EvidenceItem]:
    # Prefer diverse timestamps; keep original order by recency desc.
    members = sorted(
        cluster.members,
        key=lambda m: m.timestamp or "",
        reverse=True,
    )
    items: list[EvidenceItem] = []
    seen_text: set[str] = set()
    for member in members:
        key = member.raw_text.strip()
        if key in seen_text:
            continue
        seen_text.add(key)
        items.append(
            EvidenceItem(
                raw_text=member.raw_text,
                timestamp=member.timestamp,
                project=member.project,
                session_id=member.session_id,
            )
        )
        if len(items) >= limit:
            break
    return items


def _project_rollups(candidates: list[Candidate]) -> list[ProjectRollup]:
    by_project: dict[str, ProjectRollup] = {}
    for cand in candidates:
        # Attribute full candidate cost to each project it touches
        # (usually one). Avoid dividing — rollup is "involved in".
        share_projects = sorted(cand.projects) or ["unknown"]
        for project in share_projects:
            roll = by_project.get(project)
            if roll is None:
                roll = ProjectRollup(project=project)
                by_project[project] = roll
            roll.candidates += 1
            roll.run_count += cand.distinct_sessions
            roll.input_tokens += cand.input_tokens
            roll.output_tokens += cand.output_tokens
            roll.cache_read_tokens += cand.cache_read_tokens
            roll.usd += cand.usd
    return sorted(
        by_project.values(),
        key=lambda r: (-r.usd, -r.run_count, r.project.lower()),
    )


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
