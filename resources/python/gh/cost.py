"""Convert token usage to dollars with a labeled price basis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

from gh.cluster import Cluster
from gh.intents import Intent
from gh.parse import Session

# Chars-per-token used only when usage fields are absent from the file.
_CHARS_PER_TOKEN = 4.0

# Certainty order: unknown is least certain.
_BASIS_RANK = {"measured": 0, "estimated": 1, "unknown": 2}


@dataclass
class CostBreakdown:
    """Token totals and USD for one cluster, with an honest basis label."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    usd: float
    basis: str  # measured | estimated | unknown
    price_model: str = "default"  # longest-prefix key used (or default)
    # True only when a model id was read AND token counts were measured.
    # Ranking may still see usd; rendering must not print $ unless priced.
    priced: bool = False


@dataclass
class ProjectCost:
    """Per-project token/usd totals rolled up from parsed sessions."""

    project: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    usd: float = 0.0
    basis: str = "unknown"
    session_count: int = 0
    priced: bool = False
    sessions_without_model: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens


def load_prices(path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """Load prices.json. Missing/unreadable → empty dict (callers use defaults)."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "prices.json"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_model_price(
    model_id: Optional[str], prices: dict[str, Any]
) -> tuple[dict[str, float], str]:
    """Return (rate_dict, matched_key) via longest-prefix match.

    Unknown / missing model → prices['default'] (or a hard-coded fallback),
    with matched_key labeled ``default``.
    """
    fallback = {"input": 3.0, "output": 15.0, "cache_read": 0.30}
    default_raw = prices.get("default") if isinstance(prices, dict) else None
    default = _coerce_rates(default_raw) or fallback

    if not model_id or not isinstance(model_id, str):
        return default, "default"

    needle = model_id.strip().lower()
    best_key = ""
    best_rates: Optional[dict[str, float]] = None
    for key, raw in prices.items():
        if key.startswith("_") or key == "default":
            continue
        if not isinstance(key, str):
            continue
        prefix = key.lower()
        if needle == prefix or needle.startswith(prefix):
            rates = _coerce_rates(raw)
            if rates is None:
                continue
            if len(prefix) > len(best_key):
                best_key = key
                best_rates = rates

    if best_rates is not None:
        return best_rates, best_key
    return default, "default"


def cost_for_cluster(
    cluster: Cluster, prices: dict[str, Any]
) -> CostBreakdown:
    """Sum unique-session costs for a cluster. Never pretends estimates are measured."""
    # One cost per session_id — multiple intents from one session don't double-count.
    seen: set[str] = set()
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    usd = 0.0
    basis = "measured"
    price_models: list[str] = []
    priced_flags: list[bool] = []

    members = list(cluster.members) if cluster else []
    if not members:
        return CostBreakdown(0, 0, 0, 0.0, "unknown", "default", False)

    for member in members:
        sid = member.session_id or id(member)
        if sid in seen:
            continue
        seen.add(str(sid))
        part = _cost_for_intent(member, prices)
        input_tokens += part.input_tokens
        output_tokens += part.output_tokens
        cache_read_tokens += part.cache_read_tokens
        usd += part.usd
        basis = _worse_basis(basis, part.basis)
        price_models.append(part.price_model)
        priced_flags.append(part.priced)

    price_model = (
        "default"
        if any(p == "default" for p in price_models)
        else (price_models[0] if len(set(price_models)) == 1 else "mixed")
    )
    return CostBreakdown(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        usd=round(usd, 6),
        basis=basis,
        price_model=price_model,
        priced=bool(priced_flags) and all(priced_flags),
    )


def project_costs_from_sessions(
    sessions: list[Session], prices: dict[str, Any]
) -> list[ProjectCost]:
    """Roll up measured/estimated cost per project across all sessions."""
    buckets: dict[str, ProjectCost] = {}
    for session in sessions:
        name = (session.project or "").strip() or "unknown"
        part = cost_for_session(session, prices)
        roll = buckets.get(name)
        if roll is None:
            roll = ProjectCost(
                project=name,
                basis=part.basis,
                priced=part.priced,
            )
            buckets[name] = roll
        else:
            roll.basis = _worse_basis(roll.basis, part.basis)
            roll.priced = roll.priced and part.priced
        roll.input_tokens += part.input_tokens
        roll.output_tokens += part.output_tokens
        roll.cache_read_tokens += part.cache_read_tokens
        roll.usd += part.usd
        roll.session_count += 1
        has_model, _has_tokens = session_pricing_status(session)
        if not has_model:
            roll.sessions_without_model += 1
    for roll in buckets.values():
        roll.usd = round(roll.usd, 6)
    return sorted(
        buckets.values(),
        key=lambda p: (-p.usd, -p.tokens, p.project.lower()),
    )


def cost_for_session(session: Session, prices: dict[str, Any]) -> CostBreakdown:
    """Cost one parsed session with the same measured/estimated/unknown rules."""
    usage = _usage_from_session(session)
    intent = Intent(
        session_id=session.session_id,
        harness=session.harness,
        project=session.project or "unknown",
        timestamp=session.started_at,
        raw_text="",
        normalized="",
        session_turn_count=len(session.turns),
        session_tokens=usage["session_tokens"],
        session_input_tokens=usage["input_tokens"],
        session_output_tokens=usage["output_tokens"],
        session_cache_read_tokens=usage["cache_read_tokens"],
        session_model=usage["model"],
        session_text_chars=usage["text_chars"],
    )
    return _cost_for_intent(intent, prices)


def session_pricing_status(session: Session) -> tuple[bool, bool]:
    """Return (has_model_id, has_measured_tokens) from the file itself."""
    usage = _usage_from_session(session)
    has_model = bool(usage.get("model") and str(usage["model"]).strip())
    has_tokens = usage.get("session_tokens") is not None
    return has_model, has_tokens


def count_sessions_without_model(sessions: list[Session]) -> int:
    """Sessions where no model id was present in the file."""
    return sum(1 for s in sessions if not session_pricing_status(s)[0])


def count_sessions_with_tokens(sessions: list[Session]) -> int:
    """Sessions where token counts were actually read from the file."""
    return sum(1 for s in sessions if session_pricing_status(s)[1])


def date_range_for_sessions(sessions: list[Session]) -> str:
    """YYYY-MM-DD or ``start → end`` across parsed sessions; ``none`` if empty."""
    dates: list[str] = []
    for session in sessions:
        for raw in (session.started_at, session.ended_at):
            if not raw:
                continue
            text = str(raw).strip()
            if "T" in text:
                text = text.split("T", 1)[0]
            elif len(text) >= 10 and text[4] == "-" and text[7] == "-":
                text = text[:10]
            if text:
                dates.append(text)
    if not dates:
        return "none"
    first, last = min(dates), max(dates)
    return first if first == last else f"{first} → {last}"


def cost_breakdown_dict(cost: CostBreakdown) -> dict[str, Any]:
    """JSON-ready dict; basis always present beside the dollar figure."""
    return asdict(cost)


def _cost_for_intent(intent: Intent, prices: dict[str, Any]) -> CostBreakdown:
    rates, price_key = resolve_model_price(intent.session_model, prices)

    measured_in = intent.session_input_tokens
    measured_out = intent.session_output_tokens
    measured_cache = intent.session_cache_read_tokens

    has_any_measured = any(
        v is not None for v in (measured_in, measured_out, measured_cache)
    )

    if has_any_measured:
        inp = int(measured_in or 0)
        out = int(measured_out or 0)
        cache = int(measured_cache or 0)
        basis = "measured"
    else:
        chars = intent.session_text_chars or 0
        if chars > 0:
            # No file usage: estimate from turn text. All mass as input —
            # never labeled measured.
            est = max(1, int(round(chars / _CHARS_PER_TOKEN)))
            inp, out, cache = est, 0, 0
            basis = "estimated"
        else:
            return CostBreakdown(0, 0, 0, 0.0, "unknown", price_key, False)

    usd = _usd(inp, out, cache, rates)
    has_model = bool(
        intent.session_model and str(intent.session_model).strip()
    )
    priced = has_model and has_any_measured
    return CostBreakdown(
        inp, out, cache, round(usd, 6), basis, price_key, priced
    )


def _usd(
    inp: int, out: int, cache: int, rates: dict[str, float]
) -> float:
    return (
        (inp / 1_000_000.0) * rates.get("input", 0.0)
        + (out / 1_000_000.0) * rates.get("output", 0.0)
        + (cache / 1_000_000.0) * rates.get("cache_read", 0.0)
    )


def _coerce_rates(raw: Any) -> Optional[dict[str, float]]:
    if not isinstance(raw, dict):
        return None
    try:
        return {
            "input": float(raw.get("input", 0.0)),
            "output": float(raw.get("output", 0.0)),
            "cache_read": float(raw.get("cache_read", 0.0)),
        }
    except (TypeError, ValueError):
        return None


def _worse_basis(a: str, b: str) -> str:
    return a if _BASIS_RANK.get(a, 9) >= _BASIS_RANK.get(b, 9) else b


def _usage_from_session(session: Session) -> dict:
    """Aggregate per-session token fields; None means absent (not zero)."""
    input_total = 0
    output_total = 0
    cache_total = 0
    saw_input = False
    saw_output = False
    saw_cache = False
    model: Optional[str] = None
    text_chars = 0

    for turn in session.turns:
        text_chars += len(turn.text or "")
        if turn.model and not model:
            model = turn.model
        if turn.input_tokens is not None:
            input_total += turn.input_tokens
            saw_input = True
        if turn.output_tokens is not None:
            output_total += turn.output_tokens
            saw_output = True
        if turn.cache_read_tokens is not None:
            cache_total += turn.cache_read_tokens
            saw_cache = True

    session_tokens: Optional[int]
    if saw_input or saw_output or saw_cache:
        session_tokens = (
            (input_total if saw_input else 0)
            + (output_total if saw_output else 0)
            + (cache_total if saw_cache else 0)
        )
    else:
        session_tokens = None

    return {
        "session_tokens": session_tokens,
        "input_tokens": input_total if saw_input else None,
        "output_tokens": output_total if saw_output else None,
        "cache_read_tokens": cache_total if saw_cache else None,
        "model": model,
        "text_chars": text_chars,
    }
