"""Measure context rediscovery: exploration the agent repeats before it edits."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from gh.parse import Session, ToolCall

# Need this many sessions with a first edit before rates are honest.
MIN_RESOLVABLE = 5

_MUTATING_TOOLS = frozenset(
    {
        "strreplace",
        "write",
        "edit",
        "delete",
        "editnotebook",
        "applypatch",
    }
)
_SHELL_TOOLS = frozenset({"shell", "bash", "terminal"})
_READ_TOOLS = frozenset({"read", "readfile"})
_SEARCH_TOOLS = frozenset({"glob", "grep"})

_READONLY_CMDS = frozenset(
    {
        "ls",
        "dir",
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "bat",
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "ag",
        "ack",
        "find",
        "fd",
        "locate",
        "wc",
        "file",
        "stat",
        "pwd",
        "which",
        "type",
        "echo",
        "printf",
        "true",
        "false",
        "test",
        "dirname",
        "basename",
        "realpath",
        "readlink",
        "tree",
        "md5sum",
        "sha256sum",
        "sha1sum",
        "cksum",
        "env",
        "printenv",
        "date",
        "uname",
        "whoami",
        "id",
        "hostname",
        "jq",
        "sed",
        "awk",
        "cut",
        "sort",
        "uniq",
        "tr",
        "diff",
        "cmp",
        "column",
        "nl",
        "od",
        "hexdump",
        "strings",
        "cd",
    }
)
_GIT_READONLY = frozenset(
    {
        "status",
        "log",
        "diff",
        "show",
        "blame",
        "ls-files",
        "ls-tree",
        "rev-parse",
        "describe",
        "branch",
        "tag",
        "remote",
        "config",
        "shortlog",
        "whatchanged",
        "cat-file",
        "name-rev",
        "symbolic-ref",
        "rev-list",
        "range-diff",
    }
)
_PREFIX_SKIP = frozenset(
    {"sudo", "command", "time", "nice", "nohup", "env", "stdbuf"}
)
_SPLIT_CMDS = re.compile(r"\s*(?:&&|\|\||;|\n)\s*")
_SPLIT_PIPE = re.compile(r"\|")
_TOKEN = re.compile(r"\S+")


@dataclass
class FileReread:
    path: str
    sessions: int


@dataclass
class ProjectPrefix:
    project: str
    sessions: int
    median_prefix: int


@dataclass
class RediscoveryReport:
    """Context rediscovery across the scanned window."""

    resolvable_sessions: int = 0
    no_mutation_sessions: int = 0
    sessions_with_tools: int = 0
    harnesses_excluded: list[str] = field(default_factory=list)
    pattern_pct: Optional[float] = None
    median_prefix: Optional[float] = None
    p90_prefix: Optional[float] = None
    explore_pct: Optional[float] = None
    top_files: list[FileReread] = field(default_factory=list)
    per_project: list[ProjectPrefix] = field(default_factory=list)
    sufficient: bool = False
    notes: list[str] = field(default_factory=list)


def is_mutating(call: ToolCall) -> bool:
    """True if this call is the first-edit class (write/edit/mutating shell)."""
    name = (call.name or "").strip().lower()
    if name in _MUTATING_TOOLS:
        return True
    if name in _SHELL_TOOLS:
        return _shell_is_mutating(call.command or "")
    return False


def exploration_prefix(calls: list[ToolCall]) -> Optional[list[ToolCall]]:
    """Calls before the first mutation, or None if the session never edits."""
    for i, call in enumerate(calls):
        if is_mutating(call):
            return list(calls[:i])
    return None


def has_explore_loop(prefix: list[ToolCall]) -> bool:
    """Glob or Grep, then two or more Reads, in that order."""
    seen_search = False
    reads_after = 0
    for call in prefix:
        name = (call.name or "").strip().lower()
        if name in _SEARCH_TOOLS:
            seen_search = True
        elif seen_search and name in _READ_TOOLS:
            reads_after += 1
    return seen_search and reads_after >= 2


def measure_context_rediscovery(sessions: list[Session]) -> RediscoveryReport:
    """Compute rediscovery stats. Never raises; degrades to labeled notes."""
    report = RediscoveryReport()
    if not sessions:
        report.notes.append("no sessions to measure")
        return report

    by_harness: dict[str, list[Session]] = defaultdict(list)
    for session in sessions:
        by_harness[session.harness or "unknown"].append(session)

    included: list[Session] = []
    for harness, group in by_harness.items():
        if any(s.tool_calls for s in group):
            included.extend(group)
        else:
            report.harnesses_excluded.append(harness)

    report.sessions_with_tools = sum(1 for s in included if s.tool_calls)

    prefix_lengths: list[int] = []
    pattern_hits = 0
    before_change_tools = 0
    all_tools = 0
    file_sessions: dict[str, set[str]] = defaultdict(set)
    project_lengths: dict[str, list[int]] = defaultdict(list)

    for session in included:
        calls = list(session.tool_calls or [])
        if not calls:
            continue
        all_tools += len(calls)
        prefix = exploration_prefix(calls)
        if prefix is None:
            report.no_mutation_sessions += 1
            before_change_tools += len(calls)
            continue
        report.resolvable_sessions += 1
        prefix_lengths.append(len(prefix))
        before_change_tools += len(prefix)
        if has_explore_loop(prefix):
            pattern_hits += 1
        sid = session.session_id or id(session)
        for call in prefix:
            if (call.name or "").strip().lower() not in _READ_TOOLS:
                continue
            path = (call.path or "").strip()
            if path:
                file_sessions[path].add(str(sid))
        project = (session.project or "").strip() or "unknown"
        project_lengths[project].append(len(prefix))

    report.sufficient = report.resolvable_sessions >= MIN_RESOLVABLE
    if report.sufficient and prefix_lengths:
        ordered = sorted(prefix_lengths)
        report.median_prefix = _percentile(ordered, 50)
        report.p90_prefix = _percentile(ordered, 90)
        report.pattern_pct = 100.0 * pattern_hits / report.resolvable_sessions
        if all_tools > 0:
            report.explore_pct = 100.0 * before_change_tools / all_tools
        ranked = sorted(
            file_sessions.items(),
            key=lambda kv: (-len(kv[1]), kv[0]),
        )
        report.top_files = [
            FileReread(path=path, sessions=len(sids))
            for path, sids in ranked[:5]
        ]
        projects = []
        for name, lengths in project_lengths.items():
            projects.append(
                ProjectPrefix(
                    project=name,
                    sessions=len(lengths),
                    median_prefix=int(round(_percentile(sorted(lengths), 50))),
                )
            )
        projects.sort(key=lambda p: (-p.sessions, p.project.lower()))
        report.per_project = projects

    if report.harnesses_excluded:
        labels = ", ".join(_harness_label(h) for h in report.harnesses_excluded)
        report.notes.append(
            f"{labels} sessions had no tool-use blocks; "
            "excluded from context rediscovery"
        )
    if report.no_mutation_sessions:
        n = report.no_mutation_sessions
        report.notes.append(
            f"{n} session{'s' if n != 1 else ''} had no mutating call; "
            "not folded into the median"
        )
    return report


def _percentile(ordered: list[int], p: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (p / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(ordered[lo])
    frac = idx - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _shell_is_mutating(command: str) -> bool:
    text = (command or "").strip()
    if not text:
        return True
    for segment in _SPLIT_CMDS.split(text):
        if not segment.strip():
            continue
        for piece in _SPLIT_PIPE.split(segment):
            if _one_command_mutating(piece.strip()):
                return True
    return False


def _one_command_mutating(command: str) -> bool:
    tokens = _TOKEN.findall(command)
    i = 0
    while i < len(tokens) and "=" in tokens[i] and not tokens[i].startswith("-"):
        i += 1
    while i < len(tokens) and tokens[i].rsplit("/", 1)[-1] in _PREFIX_SKIP:
        i += 1
        while i < len(tokens) and "=" in tokens[i] and not tokens[i].startswith("-"):
            i += 1
    if i >= len(tokens):
        return True
    cmd = tokens[i].rsplit("/", 1)[-1]
    if cmd == "sed" and any(
        a == "-i" or a.startswith("-i") for a in tokens[i + 1 :]
    ):
        return True
    if cmd == "git":
        return _git_is_mutating(tokens[i + 1 :])
    return cmd not in _READONLY_CMDS


def _git_is_mutating(args: list[str]) -> bool:
    rest = [a for a in args if not a.startswith("-")]
    if not rest:
        return False
    sub = rest[0]
    if sub in _GIT_READONLY:
        # `git branch -d` / `git tag -d` delete refs.
        if sub in {"branch", "tag"} and any(
            a in ("-d", "-D", "-m", "-M") for a in args
        ):
            return True
        return False
    if sub == "stash":
        action = rest[1] if len(rest) > 1 else "push"
        return action not in {"list", "show"}
    return True


def _harness_label(name: str) -> str:
    return {
        "claude_code": "Claude Code",
        "codex": "Codex",
        "cursor": "Cursor",
    }.get(name, name.replace("_", " ").title())
