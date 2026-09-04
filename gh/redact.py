"""Redact secret-like patterns and truncate evidence for safe display."""

from __future__ import annotations

import re

# Default evidence length when --redact is on.
EVIDENCE_LIMIT = 120

# Secret-like tokens to scrub from all rendered output.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-+=/]{8,}", re.IGNORECASE),
    # Long base64-looking runs (API keys, blobs) — keep short ones.
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),
)


def redact_text(text: str, *, limit: int | None = EVIDENCE_LIMIT) -> str:
    """Strip secret-like spans, then optionally truncate."""
    cleaned = text or ""
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub("<redacted>", cleaned)
    cleaned = " ".join(cleaned.split())
    if limit is not None and len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def redact_report_strings(text: str) -> str:
    """Scrub secrets from a full rendered report string (no truncation)."""
    cleaned = text or ""
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub("<redacted>", cleaned)
    return cleaned
