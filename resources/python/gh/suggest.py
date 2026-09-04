"""Suggest a rote Play scaffold from a ranked Groundhog candidate.

Recovers task shape and observed tool procedures from cluster evidence.
Does not invent steps that are absent from transcripts.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from gh.cluster import Cluster
from gh.discover import SessionFile
from gh.intents import strip_leading_locative
from gh.rank import Candidate

_MIN_RUNS_TO_GENERALIZE = 3
_MIN_TOOL_STEPS_TOTAL = 3
_MIN_SESSIONS_FOR_PROCEDURE = 2

_RE_PATH = re.compile(r"(?:~/|/|[A-Za-z]:\\)[^\s'\"`]+")
_RE_FILENAME = re.compile(r"^[\w.+-]+\.[A-Za-z0-9]{1,12}$")
_RE_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)
_RE_SHA = re.compile(r"^[0-9a-f]{7,40}$", re.I)
_RE_NUM = re.compile(r"^\d+(?:\.\d+)?$")
_RE_TOKEN = re.compile(r"\S+")

_FOOTER = (
    "This is a starting point recovered from your transcripts. "
    "Review every step and parameter before using it as a Play."
)


@dataclass
class SuggestedParam:
    name: str
    kind: str
    examples: list[str] = field(default_factory=list)


@dataclass
class ToolStep:
    name: str
    summary: str
    # Coarse key for cross-run matching (name + summary shape).
    key: str = ""


@dataclass
class SuggestResult:
    markdown: str
    ok: bool
    notes: list[str] = field(default_factory=list)


def suggest_scaffold(
    candidate: Candidate,
    cluster: Cluster,
    session_files: list[SessionFile],
    *,
    rank: int = 1,
) -> SuggestResult:
    """Build a markdown Play scaffold for one ranked candidate."""
    notes: list[str] = []
    texts = [m.raw_text.strip() for m in cluster.members if m.raw_text.strip()]
    description, params = parameterize_task(texts)

    session_ids = list(candidate.session_ids) or sorted(
        {m.session_id for m in cluster.members if m.session_id}
    )
    paths = _paths_for_sessions(session_ids, session_files)
    per_session_steps = [
        recover_tool_steps(path) for path in paths
    ]
    sessions_with_tools = sum(1 for steps in per_session_steps if steps)
    total_tools = sum(len(steps) for steps in per_session_steps)

    small_sample = candidate.distinct_sessions < _MIN_RUNS_TO_GENERALIZE
    if small_sample:
        notes.append(
            f"Only {candidate.distinct_sessions} session(s) in this cluster — "
            "too small to generalize a Play from."
        )

    procedure_note = ""
    common: list[ToolStep] = []
    varied: list[ToolStep] = []
    if sessions_with_tools < _MIN_SESSIONS_FOR_PROCEDURE or total_tools < _MIN_TOOL_STEPS_TOTAL:
        procedure_note = (
            "Tool-use blocks are absent or too sparse across distinct "
            f"session transcripts ({sessions_with_tools} session(s) with "
            f"tools, {total_tools} tool call(s) total) to recover a shared "
            "procedure. Not inventing steps."
        )
        notes.append(procedure_note)
    else:
        common, varied = common_and_varied_steps(per_session_steps)

    md = render_suggest_markdown(
        candidate=candidate,
        rank=rank,
        description=description,
        params=params,
        common=common,
        varied=varied,
        procedure_note=procedure_note,
        notes=notes,
        members=cluster.members,
    )
    return SuggestResult(markdown=md, ok=True, notes=notes)


def parameterize_task(texts: list[str]) -> tuple[str, list[SuggestedParam]]:
    """Diff member texts into a constant description plus named parameters."""
    cleaned = [_prep_text(t) for t in texts if t and t.strip()]
    if not cleaned:
        return "(no task text)", []
    if len(cleaned) == 1:
        return cleaned[0], []

    token_lists = [_RE_TOKEN.findall(t) for t in cleaned]
    # Align every run to the shortest as a pivot for slotting.
    pivot_i = min(range(len(token_lists)), key=lambda i: len(token_lists[i]))
    pivot = token_lists[pivot_i]

    # For each pivot index, collect the token used by each run at the
    # best-aligned position (exact index when lengths match; else search).
    columns: list[list[str]] = [[] for _ in pivot]
    for tokens in token_lists:
        if len(tokens) == len(pivot):
            for i, tok in enumerate(tokens):
                columns[i].append(tok)
            continue
        # Greedy left-to-right match against pivot.
        j = 0
        for i, piv in enumerate(pivot):
            if j < len(tokens) and tokens[j] == piv:
                columns[i].append(tokens[j])
                j += 1
            elif j < len(tokens):
                columns[i].append(tokens[j])
                j += 1
            else:
                columns[i].append(piv)

    params: list[SuggestedParam] = []
    parts: list[str] = []
    param_n = 0
    i = 0
    while i < len(columns):
        values = columns[i]
        if _is_constant_column(values):
            parts.append(values[0])
            i += 1
            continue
        # Extend variable span while columns keep varying.
        j = i + 1
        while j < len(columns) and not _is_constant_column(columns[j]):
            j += 1
        examples = []
        for run_idx in range(len(cleaned)):
            chunk = " ".join(columns[k][run_idx] for k in range(i, j))
            if chunk and chunk not in examples:
                examples.append(chunk)
        kind = infer_param_kind(examples)
        name = _name_param(parts, kind, param_n)
        param_n += 1
        params.append(SuggestedParam(name=name, kind=kind, examples=examples[:5]))
        parts.append("{" + name + "}")
        i = j

    description = " ".join(parts).strip() or cleaned[0]
    return description, params


def infer_param_kind(examples: list[str]) -> str:
    if not examples:
        return "free text"
    scores = Counter()
    for ex in examples:
        s = ex.strip().strip("\"'`")
        if _RE_PATH.search(s) or s.startswith("~/") or s.startswith("/"):
            scores["path"] += 1
        elif _RE_FILENAME.match(s):
            scores["filename"] += 1
        elif _RE_UUID.match(s) or _RE_SHA.match(s):
            scores["identifier"] += 1
        elif _RE_NUM.match(s) or re.fullmatch(r"cp-?\d+(?:\.\d+)?", s, re.I):
            scores["number"] += 1
        else:
            scores["free text"] += 1
    return scores.most_common(1)[0][0]


def recover_tool_steps(path: str) -> list[ToolStep]:
    """Walk a session JSONL and collect tool_use blocks in order."""
    steps: list[ToolStep] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or "tool_use" not in line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                for block in _content_blocks(obj):
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_use":
                        continue
                    name = block.get("name") or block.get("tool") or "tool"
                    if not isinstance(name, str):
                        name = "tool"
                    summary = summarize_tool_input(block.get("input"))
                    key = f"{name}|{_coarse_summary(summary)}"
                    steps.append(ToolStep(name=name, summary=summary, key=key))
    except OSError:
        return []
    return steps


def summarize_tool_input(inp: Any) -> str:
    """One-line summary of a tool_use input dict."""
    if inp is None:
        return ""
    if isinstance(inp, str):
        return _clip(inp, 80)
    if not isinstance(inp, dict):
        return _clip(str(inp), 80)
    for key in (
        "command",
        "path",
        "target_directory",
        "file_path",
        "pattern",
        "query",
        "glob_pattern",
        "url",
        "prompt",
        "content",
        "old_string",
        "new_string",
    ):
        val = inp.get(key)
        if isinstance(val, str) and val.strip():
            return f"{key}={_clip(val.strip(), 64)}"
    # Fall back to first short string value.
    for key, val in inp.items():
        if isinstance(val, str) and val.strip():
            return f"{key}={_clip(val.strip(), 64)}"
    return ""


def common_and_varied_steps(
    per_session: list[list[ToolStep]],
) -> tuple[list[ToolStep], list[ToolStep]]:
    """Steps in most sessions → common; the rest → varied."""
    nonempty = [steps for steps in per_session if steps]
    n = len(nonempty)
    if n == 0:
        return [], []
    threshold = max(2, (n + 1) // 2)  # strict majority-ish, at least 2

    # Preserve first-seen order from the longest transcript.
    key_order: list[str] = []
    seen_keys: set[str] = set()
    for steps in sorted(nonempty, key=len, reverse=True):
        for step in steps:
            if step.key not in seen_keys:
                seen_keys.add(step.key)
                key_order.append(step.key)

    counts: Counter[str] = Counter()
    exemplars: dict[str, ToolStep] = {}
    for steps in nonempty:
        for key in {s.key for s in steps}:
            counts[key] += 1
        for step in steps:
            exemplars.setdefault(step.key, step)

    common: list[ToolStep] = []
    varied: list[ToolStep] = []
    for key in key_order:
        step = exemplars[key]
        if counts[key] >= threshold:
            common.append(step)
        else:
            varied.append(step)
    # Cap length so the scaffold stays readable.
    return common[:40], varied[:20]


def render_suggest_markdown(
    *,
    candidate: Candidate,
    rank: int,
    description: str,
    params: list[SuggestedParam],
    common: list[ToolStep],
    varied: list[ToolStep],
    procedure_note: str,
    notes: list[str],
    members,
) -> str:
    lines: list[str] = []
    lines.append(f"# Suggested Play scaffold — candidate #{rank}")
    lines.append("")
    lines.append(f"**Label:** {candidate.label}")
    projects = ", ".join(sorted(candidate.projects)) or "unknown"
    lines.append(
        f"**Observed:** {candidate.distinct_sessions} sessions · "
        f"{candidate.first_seen or '?'} → {candidate.last_seen or '?'} · "
        f"{projects}"
    )
    lines.append("")
    lines.append("## Task")
    lines.append("")
    lines.append(description)
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    if not params:
        lines.append(
            "_No varying slots found across runs — the asks were nearly identical._"
        )
    else:
        lines.append("| Name | Type | Example values |")
        lines.append("| --- | --- | --- |")
        for p in params:
            examples = "; ".join(_clip(e, 48) for e in p.examples) or "—"
            lines.append(f"| `{p.name}` | {p.kind} | {examples} |")
    lines.append("")
    lines.append("## Proposed procedure (common across runs)")
    lines.append("")
    if procedure_note:
        lines.append(f"**Not recovered:** {procedure_note}")
    elif not common:
        lines.append(
            "**Not recovered:** No shared tool-use sequence met the "
            "majority threshold across session transcripts."
        )
    else:
        for i, step in enumerate(common, 1):
            detail = f" — {step.summary}" if step.summary else ""
            lines.append(f"{i}. `{step.name}`{detail}")
    lines.append("")
    lines.append("## Varied between runs")
    lines.append("")
    if procedure_note:
        lines.append("_n/a — procedure not recovered._")
    elif not varied:
        lines.append("_No minority tool steps to list._")
    else:
        for step in varied:
            detail = f" — {step.summary}" if step.summary else ""
            lines.append(f"- `{step.name}`{detail}")
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    seen = set()
    for m in members:
        key = (m.session_id, (m.timestamp or "")[:10])
        if key in seen:
            continue
        seen.add(key)
        date = (m.timestamp or "unknown")[:10]
        lines.append(f"- `{m.session_id}` · {date} · {m.project}")
    if notes:
        lines.append("")
        lines.append("## Honesty notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(_FOOTER)
    lines.append("")
    return "\n".join(lines)


def _prep_text(text: str) -> str:
    s = strip_leading_locative((text or "").strip())
    s = " ".join(s.split())
    return s


def _is_constant_column(values: list[str]) -> bool:
    norm = {v for v in values if v is not None}
    return len(norm) <= 1


def _name_param(parts: list[str], kind: str, index: int) -> str:
    prev = ""
    for tok in reversed(parts):
        if tok.startswith("{") and tok.endswith("}"):
            continue
        cleaned = re.sub(r"[^A-Za-z0-9]+", "", tok).lower()
        if cleaned and cleaned not in {
            "the",
            "a",
            "an",
            "to",
            "for",
            "in",
            "of",
            "and",
            "or",
            "with",
            "at",
            "on",
        }:
            prev = cleaned
            break
    if kind == "path" and prev:
        return f"{prev}_path" if prev not in ("path", "file") else "path"
    if kind == "filename" and prev:
        return f"{prev}_file" if prev != "file" else "filename"
    if kind == "number" and prev in {"cp", "checkpoint", "gate"}:
        return "checkpoint"
    if prev and len(prev) >= 3:
        return f"{prev}_{index + 1}" if index else prev
    return f"param_{index + 1}"


def _paths_for_sessions(
    session_ids: list[str], session_files: list[SessionFile]
) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for sid in session_ids:
        if not sid:
            continue
        for sf in session_files:
            p = sf.path
            stem = Path(p).stem
            if sid == stem or sid in p:
                if p not in seen:
                    seen.add(p)
                    paths.append(p)
                break
    return paths


def _content_blocks(obj: dict) -> list[Any]:
    msg = obj.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, list):
            return content
    content = obj.get("content")
    if isinstance(content, list):
        return content
    return []


def _coarse_summary(summary: str) -> str:
    s = summary or ""
    s = _RE_PATH.sub("<path>", s)
    s = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", s, flags=re.I)
    s = re.sub(r"\d+", "<n>", s)
    # Keep tool arg key only for matching breadth.
    if "=" in s:
        return s.split("=", 1)[0]
    return s[:40]


def _clip(text: str, limit: int) -> str:
    t = " ".join((text or "").split())
    if len(t) <= limit:
        return t
    return t[: limit - 1].rstrip() + "…"
