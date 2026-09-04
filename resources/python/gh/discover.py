"""Find local Claude Code / Codex / Cursor session history files."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SessionFile:
    """One discovered session transcript file."""

    path: str
    harness: str
    mtime: float
    size_bytes: int
    project: str = ""


# Reason strings rendered in NOT COUNTED. Keep in sync with gh.render.
SKIP_SYMLINK_OUTSIDE = "symlink points outside the history directory"
SKIP_UNREADABLE_DIR = "unreadable directory"


def is_unreadable_dir_reason(reason: str) -> bool:
    return reason == SKIP_UNREADABLE_DIR or reason.startswith(
        SKIP_UNREADABLE_DIR + ":"
    )


@dataclass
class DiscoveryResult:
    """Outcome of scanning local harness session roots."""

    files: list[SessionFile] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    skipped: list[tuple[str, str]] = field(default_factory=list)


# Harness name -> one or more root directories under $HOME.
_HARNESS_ROOTS: dict[str, tuple[str, ...]] = {
    "claude_code": (".claude/projects",),
    "codex": (".codex/sessions", ".codex/history"),
    "cursor": (".cursor/projects",),
}

_CURSOR_PROJECTS_MARKER = "-projects-"


def harness_roots() -> dict[str, tuple[str, ...]]:
    """Relative roots checked under $HOME for each harness."""
    return {k: tuple(v) for k, v in _HARNESS_ROOTS.items()}


def checked_locations(home: Path | None = None) -> list[str]:
    """Absolute paths Groundhog looks at (for empty-state reporting)."""
    base = home if home is not None else Path.home()
    paths: list[str] = []
    for rels in _HARNESS_ROOTS.values():
        for rel in rels:
            paths.append(str(base / rel))
    return paths


def cursor_project_name(dirname: str) -> str:
    """Project label from a ``~/.cursor/projects/<dir>`` directory name.

    Names look like ``home-<user>-projects-<ProjectName>``. The segment after
    the last ``-projects-`` is kept with its original case. Directories that
    do not match fall back to the raw name.
    """
    idx = dirname.rfind(_CURSOR_PROJECTS_MARKER)
    if idx == -1:
        return dirname
    name = dirname[idx + len(_CURSOR_PROJECTS_MARKER) :]
    return name if name else dirname


def discover_sessions(days: int, home: Path | None = None) -> DiscoveryResult:
    """Search each harness root for recent *.jsonl session files.

    Missing dirs, permission errors, and unreadable files become labeled
    source statuses. Never raises. Zero files is a valid result.
    """
    result = DiscoveryResult()
    for harness in _HARNESS_ROOTS:
        partial = discover_harness(harness, days, home=home)
        result.sources[harness] = partial.sources.get(harness, "absent")
        result.files.extend(partial.files)
        result.skipped.extend(partial.skipped)
    result.files.sort(key=lambda f: f.mtime, reverse=True)
    return result


def discover_harness(
    harness: str, days: int, home: Path | None = None
) -> DiscoveryResult:
    """Search one harness's roots. Unknown harness → empty with absent status."""
    result = DiscoveryResult()
    rel_roots = _HARNESS_ROOTS.get(harness)
    if not rel_roots:
        result.sources[harness] = "absent"
        return result

    cutoff = time.time() - max(0, days) * 86400.0
    base = home if home is not None else Path.home()
    root_statuses: list[str] = []
    harness_files: list[SessionFile] = []
    harness_skipped: list[tuple[str, str]] = []

    for rel in rel_roots:
        root = base / rel
        if harness == "cursor":
            status, found, skipped = _scan_cursor_projects(root, cutoff)
        else:
            status, found, skipped = _scan_root(root, harness, cutoff)
        root_statuses.append(status)
        harness_files.extend(found)
        harness_skipped.extend(skipped)

    result.sources[harness] = _combine_statuses(root_statuses)
    result.files = harness_files
    result.skipped = harness_skipped
    result.files.sort(key=lambda f: f.mtime, reverse=True)
    return result


def _scan_cursor_projects(
    projects_root: Path, cutoff: float
) -> tuple[str, list[SessionFile], list[tuple[str, str]]]:
    """Scan ``~/.cursor/projects/*/agent-transcripts/**/*.jsonl``. Never raises."""
    try:
        if not projects_root.exists():
            return "absent", [], []
    except OSError as exc:
        return f"unreadable: {exc}", [], []

    try:
        if not projects_root.is_dir():
            return f"unreadable: not a directory: {projects_root}", [], []
    except OSError as exc:
        return f"unreadable: {exc}", [], []

    try:
        entries = os.listdir(projects_root)
    except OSError as exc:
        return f"unreadable: {exc}", [], []

    files: list[SessionFile] = []
    skipped: list[tuple[str, str]] = []
    try:
        for name in entries:
            project_dir = projects_root / name
            try:
                if not project_dir.is_dir():
                    continue
            except OSError as exc:
                detail = getattr(exc, "strerror", None) or type(exc).__name__
                skipped.append(
                    (str(project_dir), f"{SKIP_UNREADABLE_DIR}: {detail}")
                )
                continue
            transcripts = project_dir / "agent-transcripts"
            _status, found, refused = _scan_root(transcripts, "cursor", cutoff)
            skipped.extend(refused)
            project = cursor_project_name(name)
            for sf in found:
                files.append(
                    SessionFile(
                        path=sf.path,
                        harness=sf.harness,
                        mtime=sf.mtime,
                        size_bytes=sf.size_bytes,
                        project=project,
                    )
                )
    except OSError as exc:
        if not files and not skipped:
            return f"unreadable: {exc}", [], []

    return "found", files, skipped


def _resolved_inside(path: Path, root: Path) -> bool:
    """True if ``path`` resolves to a location inside ``root``. Never raises."""
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
    except OSError:
        return False
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        return False
    return True


def _scan_root(
    root: Path, harness: str, cutoff: float
) -> tuple[str, list[SessionFile], list[tuple[str, str]]]:
    """Scan one root. Returns (status, files, skipped). Never raises."""
    try:
        if not root.exists():
            return "absent", [], []
    except OSError as exc:
        return f"unreadable: {exc}", [], []

    try:
        if not root.is_dir():
            return f"unreadable: not a directory: {root}", [], []
    except OSError as exc:
        return f"unreadable: {exc}", [], []

    # Confirm the directory itself is listable before walking.
    try:
        os.listdir(root)
    except OSError as exc:
        return f"unreadable: {exc}", [], []

    files: list[SessionFile] = []
    skipped: list[tuple[str, str]] = []

    def _on_walk_error(err: OSError) -> None:
        path = getattr(err, "filename", None) or str(err)
        detail = getattr(err, "strerror", None) or type(err).__name__
        skipped.append((str(path), f"{SKIP_UNREADABLE_DIR}: {detail}"))

    try:
        for dirpath, _dirnames, filenames in os.walk(
            root, followlinks=False, onerror=_on_walk_error
        ):
            for name in filenames:
                if not name.endswith(".jsonl"):
                    continue
                path = Path(dirpath) / name
                if not _resolved_inside(path, root):
                    skipped.append((str(path), SKIP_SYMLINK_OUTSIDE))
                    continue
                try:
                    st = path.stat()
                except OSError:
                    # Skip unreadable individual files; root still "found".
                    continue
                if st.st_mtime < cutoff:
                    continue
                files.append(
                    SessionFile(
                        path=str(path),
                        harness=harness,
                        mtime=st.st_mtime,
                        size_bytes=st.st_size,
                    )
                )
    except OSError as exc:
        # Walk started then failed mid-way — keep what we have.
        if not files and not skipped:
            return f"unreadable: {exc}", [], []

    return "found", files, skipped


def _combine_statuses(statuses: list[str]) -> str:
    """Fold per-root statuses into one harness-level label."""
    if any(s == "found" for s in statuses):
        return "found"
    unreadables = [s for s in statuses if s.startswith("unreadable:")]
    if unreadables:
        # Prefer the first concrete reason.
        return unreadables[0]
    return "absent"


def format_discovery_table(result: DiscoveryResult) -> str:
    """Build a stderr-friendly discovery summary table."""
    # Aggregate counts/sizes per harness.
    counts: dict[str, int] = {h: 0 for h in result.sources}
    sizes: dict[str, int] = {h: 0 for h in result.sources}
    for f in result.files:
        counts[f.harness] = counts.get(f.harness, 0) + 1
        sizes[f.harness] = sizes.get(f.harness, 0) + f.size_bytes

    headers = ("harness", "status", "files", "total_size")
    rows: list[tuple[str, str, str, str]] = [headers]
    for harness, status in result.sources.items():
        rows.append(
            (
                harness,
                status,
                str(counts.get(harness, 0)),
                _fmt_size(sizes.get(harness, 0)),
            )
        )

    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    lines = ["discovery:"]
    for i, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row))
        lines.append(line)
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"
