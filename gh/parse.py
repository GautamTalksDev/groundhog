"""Normalize session files into Session and Turn records."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from gh.discover import SessionFile


@dataclass
class Turn:
    """One user or assistant turn within a session."""

    role: str
    text: str
    timestamp: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cache_read_tokens: Optional[int]
    model: Optional[str]


@dataclass
class Session:
    """Normalized session transcript."""

    session_id: str
    harness: str
    project: str
    started_at: Optional[str]
    ended_at: Optional[str]
    turns: list[Turn]
    parse_status: str


@dataclass
class ParseResult:
    """Sessions successfully normalized plus per-file skip reasons."""

    sessions: list[Session] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    malformed_lines: int = 0
    truncated: bool = False
    files_read: int = 0
    files_total: int = 0


def first_present(obj: Any, candidates: Iterable[str]) -> Any:
    """Return the first non-None value found among candidate keys/paths.

    Supports dotted paths (e.g. ``message.content``, ``usage.input_tokens``).
    Missing intermediate keys yield None for that candidate and continue.
    """
    for key in candidates:
        value = _dig(obj, key)
        if value is not None:
            return value
    return None


def _dig(obj: Any, path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def parse_sessions(
    files: list[SessionFile],
    *,
    deadline: Optional[float] = None,
) -> ParseResult:
    """Parse discovered session files into normalized Session records.

    Streams JSONL line by line — never holds all raw lines in memory.
    Reads smallest files first. If ``deadline`` (monotonic seconds) is
    passed and exceeded, stops and sets ``truncated=True``.

    Never raises. Unreadable/empty/broken files land in ``skipped``.
    """
    import time as _time

    result = ParseResult(files_total=len(files))
    # Smallest first so a stranger's huge transcripts don't block everything.
    ordered = sorted(files, key=lambda f: (f.size_bytes, f.path))

    for sf in ordered:
        if deadline is not None and _time.monotonic() >= deadline:
            remaining = len(ordered) - result.files_read
            result.truncated = True
            if remaining > 0:
                result.skipped.append(
                    (
                        "(remaining files)",
                        f"time budget hit; {remaining} file(s) not read",
                    )
                )
            break
        result.files_read += 1
        try:
            session, bad_lines = _parse_file(sf)
        except OSError as exc:
            result.skipped.append((sf.path, f"unreadable: {exc}"))
            continue
        except Exception as exc:  # noqa: BLE001 — degrade, never crash
            result.skipped.append((sf.path, f"parse error: {exc}"))
            continue

        result.malformed_lines += bad_lines
        if session is None:
            result.skipped.append((sf.path, "no usable turns"))
            continue
        result.sessions.append(session)
    return result


def _parse_file(sf: SessionFile) -> tuple[Optional[Session], int]:
    turns: list[Turn] = []
    bad_lines = 0
    session_id: Optional[str] = None
    project: Optional[str] = None
    model_hint: Optional[str] = None

    with open(sf.path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if not isinstance(obj, dict):
                bad_lines += 1
                continue

            if session_id is None:
                sid = first_present(
                    obj,
                    (
                        "sessionId",
                        "session_id",
                        "id",
                        "payload.id",
                        "payload.session_id",
                    ),
                )
                if isinstance(sid, str) and sid:
                    session_id = sid

            if project is None:
                proj = first_present(
                    obj,
                    (
                        "cwd",
                        "project",
                        "project_path",
                        "payload.cwd",
                        "payload.project",
                    ),
                )
                if isinstance(proj, str) and proj:
                    project = proj

            turn = _turn_from_record(obj, default_model=model_hint)
            if turn is None:
                continue
            if turn.model:
                model_hint = turn.model
            turns.append(turn)

    if not turns:
        return None, bad_lines

    timestamps = [t.timestamp for t in turns if t.timestamp]
    started = timestamps[0] if timestamps else None
    ended = timestamps[-1] if timestamps else None

    status = "ok"
    if bad_lines:
        status = f"partial: {bad_lines} malformed line(s)"

    return (
        Session(
            session_id=session_id or _session_id_from_path(sf.path),
            harness=sf.harness,
            project=_normalize_project(project, sf.path),
            started_at=started,
            ended_at=ended,
            turns=turns,
            parse_status=status,
        ),
        bad_lines,
    )


def _turn_from_record(obj: dict, default_model: Optional[str] = None) -> Optional[Turn]:
    role = _extract_role(obj)
    text = _extract_text(obj)
    if role is None or text is None:
        return None
    text = text.strip()
    if not text:
        return None

    timestamp = _coerce_timestamp(
        first_present(
            obj,
            (
                "timestamp",
                "time",
                "created_at",
                "payload.timestamp",
                "payload.created_at",
            ),
        )
    )

    input_tokens = _coerce_int(
        first_present(
            obj,
            (
                "usage.input_tokens",
                "usage.inputTokens",
                "usage.prompt_tokens",
                "usage.promptTokens",
                "message.usage.input_tokens",
                "message.usage.inputTokens",
                "message.usage.prompt_tokens",
                "message.usage.promptTokens",
                "payload.usage.input_tokens",
                "payload.usage.inputTokens",
                "payload.usage.prompt_tokens",
                "payload.usage.promptTokens",
                "input_tokens",
                "inputTokens",
            ),
        )
    )
    output_tokens = _coerce_int(
        first_present(
            obj,
            (
                "usage.output_tokens",
                "usage.outputTokens",
                "usage.completion_tokens",
                "usage.completionTokens",
                "message.usage.output_tokens",
                "message.usage.outputTokens",
                "message.usage.completion_tokens",
                "message.usage.completionTokens",
                "payload.usage.output_tokens",
                "payload.usage.outputTokens",
                "payload.usage.completion_tokens",
                "payload.usage.completionTokens",
                "output_tokens",
                "outputTokens",
            ),
        )
    )
    cache_read_tokens = _coerce_int(
        first_present(
            obj,
            (
                "usage.cache_read_input_tokens",
                "usage.cache_read_tokens",
                "usage.cacheReadInputTokens",
                "usage.cacheReadTokens",
                "message.usage.cache_read_input_tokens",
                "message.usage.cache_read_tokens",
                "message.usage.cacheReadInputTokens",
                "payload.usage.cache_read_input_tokens",
                "payload.usage.cacheReadInputTokens",
                "cache_read_tokens",
                "cacheReadInputTokens",
            ),
        )
    )

    model = first_present(
        obj,
        (
            "model",
            "message.model",
            "payload.model",
            "usage.model",
        ),
    )
    if not isinstance(model, str) or not model:
        model = default_model

    return Turn(
        role=role,
        text=text,
        timestamp=timestamp,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        model=model if isinstance(model, str) else None,
    )


def _extract_role(obj: dict) -> Optional[str]:
    explicit = first_present(
        obj,
        (
            "role",
            "message.role",
            "payload.role",
            "payload.message.role",
        ),
    )
    if isinstance(explicit, str):
        lowered = explicit.lower()
        if lowered in ("user", "human"):
            return "user"
        if lowered in ("assistant", "model", "bot"):
            return "assistant"

    typ = first_present(obj, ("type", "payload.type", "payload.msg_type"))
    if isinstance(typ, str):
        t = typ.lower()
        if t in ("user", "human", "user_message", "event_msg"):
            # Codex event_msg may carry user_message in nested type.
            nested = first_present(obj, ("payload.type",))
            if t == "event_msg" and isinstance(nested, str):
                if nested.lower() in ("user_message", "user"):
                    return "user"
                if nested.lower() in ("agent_message", "assistant_message"):
                    return "assistant"
            if t in ("user", "human", "user_message"):
                return "user"
        if t in ("assistant", "agent", "agent_message", "assistant_message"):
            return "assistant"
        # Claude Code uses type=user / type=assistant at top level.
        if t == "user":
            return "user"
        if t == "assistant":
            return "assistant"

    # Codex response_item with message payload.
    if typ == "response_item":
        nested_type = first_present(obj, ("payload.type",))
        nested_role = first_present(obj, ("payload.role",))
        if nested_type == "message" and isinstance(nested_role, str):
            r = nested_role.lower()
            if r in ("user", "human"):
                return "user"
            if r in ("assistant", "model"):
                return "assistant"

    return None


def _extract_text(obj: dict) -> Optional[str]:
    """Pull message text from string or content-block array shapes."""
    candidates = (
        "text",
        "content",
        "message",
        "message.content",
        "message.text",
        "payload.text",
        "payload.message",
        "payload.content",
        "payload.message.content",
    )
    for key in candidates:
        value = _dig(obj, key)
        extracted = _coerce_text(value)
        if extracted is not None:
            return extracted
    return None


def _coerce_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Nested message object: prefer its content/text.
        nested = first_present(value, ("content", "text", "message"))
        return _coerce_text(nested)
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            # Claude: {type:text, text:...}; Codex: {type:output_text, text:...}
            block_type = block.get("type")
            if block_type in (
                None,
                "text",
                "output_text",
                "input_text",
                "message",
            ):
                piece = first_present(block, ("text", "content", "value"))
                if isinstance(piece, str) and piece:
                    parts.append(piece)
        if parts:
            return "\n".join(parts)
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Heuristic: ms vs seconds.
        ts = float(value)
        if ts > 1e12:
            ts = ts / 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return str(value)
    if isinstance(value, str):
        return value
    return str(value)


def _session_id_from_path(path: str) -> str:
    stem = Path(path).stem
    return stem or path


def _normalize_project(project: Optional[str], path: str) -> str:
    if project:
        # Prefer basename of cwd/project paths for readability.
        cleaned = project.rstrip(os.sep)
        base = os.path.basename(cleaned)
        return base or project
    derived = _project_from_path(path)
    return derived or "unknown"


def _project_from_path(path: str) -> Optional[str]:
    """Derive a project label from known harness path layouts."""
    parts = Path(path).parts
    # Claude: ~/.claude/projects/<encoded-cwd>/<session>.jsonl
    if "projects" in parts:
        try:
            idx = parts.index("projects")
            if idx + 1 < len(parts):
                encoded = parts[idx + 1]
                # Encoded cwd often looks like -Users-name-proj-foo
                label = encoded.strip("-").replace("-", "/")
                base = os.path.basename(label.replace("/", os.sep))
                return base or encoded
        except ValueError:
            pass
    # Codex: ~/.codex/sessions/<...>/<session>.jsonl — use parent dir name
    parent = Path(path).parent.name
    if parent and parent not in ("sessions", "history", "projects"):
        return parent
    return None
