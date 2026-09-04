#!/usr/bin/env python3
"""Groundhog — find repeated agent chores in local session history."""
#
# Hard constraints (never violate):
# 1. Python 3 standard library only — no third-party packages, no pip.
# 2. Reads local files only — zero network.
# 3. Writes only to an explicit --out path (otherwise stdout).
# 4. Every missing/unreadable source becomes a labeled unknown, never a crash.
#

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from gh.cluster import Cluster, cluster_intents
from gh.cost import cost_for_cluster, load_prices, project_costs_from_sessions
from gh.discover import (
    DiscoveryResult,
    checked_locations,
    discover_sessions,
    format_discovery_table,
)
from gh.intents import extract_intents
from gh.parse import ParseResult, parse_sessions
from gh.rank import RankResult, score_candidates
from gh.render import (
    Report,
    build_report,
    projects_from_session_costs,
    render_json,
    render_text,
)
from gh.suggest import suggest_scaffold

TIME_BUDGET_SECONDS = 20.0


@dataclass
class PipelineResult:
    """Report plus intermediates needed for --suggest."""

    report: Report
    clusters: list
    rank_result: RankResult
    discovery: DiscoveryResult
    notes: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groundhog",
        description=(
            "Read local Claude Code / Codex session history and tell you "
            "which chores you keep paying to redo."
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Look back this many days (default: 14)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write report to this path (default: stdout)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of top candidates to show (default: 3)",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=3,
        help="Minimum distinct sessions for a candidate (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress to stderr",
    )
    parser.add_argument(
        "--redact",
        dest="redact",
        action="store_true",
        default=True,
        help="Redact secrets and truncate evidence (default: on)",
    )
    parser.add_argument(
        "--no-redact",
        dest="redact",
        action="store_false",
        help="Disable redaction (show full evidence text)",
    )
    parser.add_argument(
        "--dump-intents",
        action="store_true",
        help="Debug: dump extracted intents",
    )
    parser.add_argument(
        "--dump-clusters",
        action="store_true",
        help="Debug: dump intent clusters",
    )
    parser.add_argument(
        "--dump-candidates",
        action="store_true",
        help="Debug: dump ranked candidates",
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Emit a Play scaffold for a ranked candidate instead of the report",
    )
    parser.add_argument(
        "--suggest-rank",
        type=int,
        default=1,
        help="Which ranked candidate to scaffold (default: 1)",
    )
    return parser


def _safe_stage(name: str, fn, notes: list[str], fallback):
    """Run a pipeline stage; on failure record a NOT COUNTED note."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — stranger machines must not crash
        notes.append(f"{name} failed ({type(exc).__name__}: {exc})")
        return fallback


def _empty_discovery() -> DiscoveryResult:
    return DiscoveryResult(
        files=[],
        sources={"claude_code": "absent", "codex": "absent", "cursor": "absent"},
    )


def _dump_intents(intents) -> None:
    print(f"intents: {len(intents)}", file=sys.stderr)
    for i, intent in enumerate(intents, 1):
        tokens = (
            "unknown"
            if intent.session_tokens is None
            else str(intent.session_tokens)
        )
        print(
            f"--- intent {i} — {intent.harness}/{intent.project} "
            f"turns={intent.session_turn_count} tokens={tokens} ---",
            file=sys.stderr,
        )
        print(f"  raw:  {intent.raw_text}", file=sys.stderr)
        print(f"  norm: {intent.normalized}", file=sys.stderr)


def _dump_clusters(clusters, priced=None) -> None:
    print(f"clusters: {len(clusters)}", file=sys.stderr)
    cost_by_id = {c.id: cost for c, cost in (priced or [])}
    for cluster in clusters:
        projects = ", ".join(sorted(cluster.projects)) or "unknown"
        print(
            f"--- {cluster.id} sessions={cluster.distinct_sessions} "
            f"members={cluster.run_count} "
            f"cohesion={cluster.cohesion:.2f} projects=[{projects}] ---",
            file=sys.stderr,
        )
        print(f"  label: {cluster.label}", file=sys.stderr)
        cost = cost_by_id.get(cluster.id)
        if cost is not None:
            print(
                f"  cost:  ${cost.usd:.4f} basis={cost.basis}",
                file=sys.stderr,
            )


def _dump_candidates(rank_result) -> None:
    print(f"candidates: {len(rank_result.candidates)}", file=sys.stderr)
    for i, cand in enumerate(rank_result.candidates, 1):
        print(
            f"--- #{i} score={cand.score:.3f} {cand.label[:70]} ---",
            file=sys.stderr,
        )


def _fallback_report(
    args: argparse.Namespace,
    *,
    notes: list[str],
    discovery: DiscoveryResult | None = None,
    locations: list[str] | None = None,
) -> Report:
    discovery = discovery or _empty_discovery()
    return build_report(
        days=args.days,
        min_runs=args.min_runs,
        top=args.top,
        session_count=0,
        harness_statuses=discovery.sources,
        rank_result=RankResult(),
        skipped=[],
        malformed_lines=0,
        extra_not_counted=notes,
        locations_checked=locations or checked_locations(),
        redact=args.redact,
    )


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    """Full analysis pipeline. Never raises — failures become NOT COUNTED."""
    notes: list[str] = []
    started = time.monotonic()
    deadline = started + TIME_BUDGET_SECONDS
    locations = checked_locations()

    discovery = _safe_stage(
        "discovery",
        lambda: discover_sessions(days=args.days),
        notes,
        _empty_discovery(),
    )

    parsed = _safe_stage(
        "parsing",
        lambda: parse_sessions(discovery.files, deadline=deadline),
        notes,
        ParseResult(),
    )

    intents = _safe_stage(
        "intent extraction",
        lambda: extract_intents(
            parsed.sessions,
            projects=sorted(
                {s.project for s in parsed.sessions if s.project}
            ),
        ),
        notes,
        [],
    )

    clusters = _safe_stage(
        "grouping",
        lambda: cluster_intents(intents, min_runs=args.min_runs),
        notes,
        [],
    )

    prices = _safe_stage("price table", load_prices, notes, {})

    def _cost_and_rank():
        priced = [(c, cost_for_cluster(c, prices)) for c in clusters]
        result = score_candidates(
            [c for c, _ in priced],
            [cost for _, cost in priced],
        )
        return priced, result

    cost_rank = _safe_stage(
        "costing/ranking",
        _cost_and_rank,
        notes,
        ([], RankResult()),
    )
    priced, rank_result = cost_rank

    session_projects = []
    if not getattr(rank_result, "candidates", None) and parsed.sessions:
        session_projects = _safe_stage(
            "session project costs",
            lambda: projects_from_session_costs(
                project_costs_from_sessions(parsed.sessions, prices)
            ),
            notes,
            [],
        )

    if args.verbose:
        try:
            print(format_discovery_table(discovery), file=sys.stderr)
            print(
                f"parse: sessions={len(parsed.sessions)} "
                f"skipped={len(parsed.skipped)} "
                f"malformed_lines={parsed.malformed_lines} "
                f"truncated={parsed.truncated}",
                file=sys.stderr,
            )
            print(f"intents: {len(intents)}", file=sys.stderr)
            print(f"clusters: {len(clusters)}", file=sys.stderr)
            print(
                f"candidates: {len(rank_result.candidates)}",
                file=sys.stderr,
            )
            elapsed = time.monotonic() - started
            print(f"elapsed: {elapsed:.2f}s", file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass

    if args.dump_intents:
        try:
            _dump_intents(intents)
        except Exception:  # noqa: BLE001
            notes.append("intent dump failed")
    if args.dump_clusters:
        try:
            _dump_clusters(clusters, priced)
        except Exception:  # noqa: BLE001
            notes.append("cluster dump failed")
    if args.dump_candidates:
        try:
            _dump_candidates(rank_result)
        except Exception:  # noqa: BLE001
            notes.append("candidate dump failed")

    report = _safe_stage(
        "report build",
        lambda: build_report(
            days=args.days,
            min_runs=args.min_runs,
            top=args.top,
            session_count=len(parsed.sessions),
            harness_statuses=discovery.sources,
            rank_result=rank_result,
            skipped=parsed.skipped,
            malformed_lines=parsed.malformed_lines,
            extra_not_counted=notes,
            locations_checked=locations,
            redact=args.redact,
            time_truncated=parsed.truncated,
            files_read=parsed.files_read,
            files_total=parsed.files_total or len(discovery.files),
            session_projects=session_projects or None,
        ),
        notes,
        _fallback_report(
            args, notes=notes, discovery=discovery, locations=locations
        ),
    )
    return PipelineResult(
        report=report,
        clusters=clusters if isinstance(clusters, list) else [],
        rank_result=rank_result if isinstance(rank_result, RankResult) else RankResult(),
        discovery=discovery if isinstance(discovery, DiscoveryResult) else _empty_discovery(),
        notes=notes,
    )


def _render_suggest(args: argparse.Namespace, pipeline: PipelineResult) -> str:
    rank = max(1, int(args.suggest_rank or 1))
    candidates = pipeline.rank_result.candidates
    if not candidates:
        return (
            "# Suggested Play scaffold\n\n"
            "No ranked candidates in this window — nothing to scaffold.\n\n"
            f"---\n\n"
            "This is a starting point recovered from your transcripts. "
            "Review every step and parameter before using it as a Play.\n"
        )
    if rank > len(candidates):
        return (
            f"# Suggested Play scaffold\n\n"
            f"No candidate at rank #{rank} "
            f"(only {len(candidates)} ranked).\n"
        )
    candidate = candidates[rank - 1]
    cluster = next(
        (c for c in pipeline.clusters if c.id == candidate.cluster_id),
        None,
    )
    if cluster is None:
        # Reconstruct a minimal cluster shell from the candidate evidence.
        from gh.intents import Intent

        members = [
            Intent(
                session_id=ev.session_id,
                harness="unknown",
                project=ev.project,
                timestamp=ev.timestamp,
                raw_text=ev.raw_text,
                normalized="",
                session_turn_count=0,
                session_tokens=None,
            )
            for ev in candidate.evidence
        ]
        cluster = Cluster(
            id=candidate.cluster_id,
            members=members,
            label=candidate.label,
            projects=set(candidate.projects),
            first_seen=candidate.first_seen,
            last_seen=candidate.last_seen,
            run_count=candidate.run_count,
            cohesion=0.0,
        )
    result = suggest_scaffold(
        candidate,
        cluster,
        pipeline.discovery.files,
        rank=rank,
    )
    return result.markdown


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse --help / bad args: preserve that exit code.
        code = exc.code if isinstance(exc.code, int) else 1
        return code

    try:
        pipeline = run_pipeline(args)
    except Exception as exc:  # noqa: BLE001 — last resort
        if args.verbose:
            traceback.print_exc(file=sys.stderr)
        pipeline = PipelineResult(
            report=_fallback_report(
                args,
                notes=[f"pipeline failed ({type(exc).__name__}: {exc})"],
            ),
            clusters=[],
            rank_result=RankResult(),
            discovery=_empty_discovery(),
            notes=[f"pipeline failed ({type(exc).__name__}: {exc})"],
        )

    try:
        if args.suggest:
            output = _render_suggest(args, pipeline)
        elif args.format == "json":
            output = render_json(pipeline.report)
        else:
            output = render_text(pipeline.report)
    except Exception as exc:  # noqa: BLE001
        output = (
            "GROUNDHOG · could not render report\n\n"
            "NOT COUNTED\n"
            f"  · render failed ({type(exc).__name__}: {exc})\n\n"
            "Local only · read your session files · wrote nothing · sent nothing\n"
        )

    if args.out:
        try:
            Path(args.out).write_text(output, encoding="utf-8")
        except OSError as exc:
            print(
                f"groundhog: could not write --out ({exc}); printing to stdout",
                file=sys.stderr,
            )
            sys.stdout.write(output)
    else:
        sys.stdout.write(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
