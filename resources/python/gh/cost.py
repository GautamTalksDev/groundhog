"""Convert token usage to dollars with a labeled price basis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

from gh.cluster import Cluster
from gh.intents import Intent

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

    members = list(cluster.members) if cluster else []
    if not members:
        return CostBreakdown(0, 0, 0, 0.0, "unknown", "default")

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
    )


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
            return CostBreakdown(0, 0, 0, 0.0, "unknown", price_key)

    usd = _usd(inp, out, cache, rates)
    return CostBreakdown(inp, out, cache, round(usd, 6), basis, price_key)


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
