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
from pathlib import Path

from gh.cluster import cluster_intents
from gh.cost import cost_for_cluster, load_prices
from gh.discover import (
    DiscoveryResult,
    checked_locations,
    discover_sessions,
    format_discovery_table,
)
from gh.intents import extract_intents
from gh.parse import ParseResult, parse_sessions
from gh.rank import RankResult, score_candidates
from gh.render import Report, build_report, render_json, render_text

TIME_BUDGET_SECONDS = 20.0


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
        help="Minimum runs for a candidate (default: 3)",
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
            f"--- {cluster.id} runs={cluster.run_count} "
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


def run_pipeline(args: argparse.Namespace) -> Report:
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
        ),
        notes,
        _fallback_report(
            args, notes=notes, discovery=discovery, locations=locations
        ),
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse --help / bad args: preserve that exit code.
        code = exc.code if isinstance(exc.code, int) else 1
        return code

    try:
        report = run_pipeline(args)
    except Exception as exc:  # noqa: BLE001 — last resort
        if args.verbose:
            traceback.print_exc(file=sys.stderr)
        report = _fallback_report(
            args,
            notes=[f"pipeline failed ({type(exc).__name__}: {exc})"],
        )

    try:
        output = (
            render_json(report)
            if args.format == "json"
            else render_text(report)
        )
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
