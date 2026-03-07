"""Canonical name normalization helpers for OpenRA actor types."""

from __future__ import annotations

from typing import Any

# OpenRA runtime aliases occasionally differ from static game-data IDs.
# Normalize everything to canonical IDs before validation and diagnostics.
ALIASES_TO_CANONICAL: dict[str, str] = {
    "weaf": "weap",
    "tenf": "tent",
    "domf": "dome",
    "fixf": "fix",
    "syrf": "proc",
}


def normalize_name(name: str | None) -> str:
    """Normalize one actor type name to canonical form."""
    if not name:
        return ""
    lowered = str(name).strip().lower()
    return ALIASES_TO_CANONICAL.get(lowered, lowered)


def normalize_prereq_token(token: str) -> str:
    """Normalize one prerequisite token, preserving OR alternatives."""
    if "|" not in token:
        return normalize_name(token)
    return "|".join(normalize_name(part) for part in token.split("|"))


def normalize_prereq_tokens(tokens: list[str] | None) -> list[str]:
    """Normalize prerequisite token list."""
    if not tokens:
        return []
    return [normalize_prereq_token(t) for t in tokens]


def get_production_item(entry: dict[str, Any]) -> str:
    """Read production queue item across item/type schema variants."""
    raw = entry.get("item", "")
    if not raw:
        raw = entry.get("type", "")
    return normalize_name(raw)


def normalize_obs_types(obs: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of observation with normalized actor type fields."""
    if not obs:
        return {} if obs is None else dict(obs)

    out: dict[str, Any] = dict(obs)

    out["available_production"] = [
        normalize_name(v) for v in obs.get("available_production", [])
    ]

    def _norm_actors(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in items:
            copied = dict(item)
            copied["type"] = normalize_name(copied.get("type", ""))
            normalized.append(copied)
        return normalized

    out["units"] = _norm_actors(obs.get("units", []))
    out["buildings"] = _norm_actors(obs.get("buildings", []))
    out["visible_enemies"] = _norm_actors(obs.get("visible_enemies", []))
    out["visible_enemy_buildings"] = _norm_actors(obs.get("visible_enemy_buildings", []))

    production_norm: list[dict[str, Any]] = []
    for p in obs.get("production", []):
        copied = dict(p)
        normalized_item = get_production_item(copied)
        copied["item"] = normalized_item
        copied["type"] = normalized_item  # compatibility for older tests/callers
        production_norm.append(copied)
    out["production"] = production_norm

    return out
