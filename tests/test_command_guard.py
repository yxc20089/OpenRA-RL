"""Tests for canonical normalization and action-level command guard."""

from openra_env.command_guard import CommandGuard
from openra_env.normalization import (
    get_production_item,
    normalize_name,
    normalize_obs_types,
    normalize_prereq_tokens,
)


def _base_obs(tick: int = 100) -> dict:
    return {
        "tick": tick,
        "available_production": ["powr", "e1"],
        "units": [{"actor_id": 1, "type": "e1"}],
        "buildings": [{"actor_id": 10, "type": "fact"}],
        "visible_enemies": [],
        "visible_enemy_buildings": [],
        "production": [],
    }


def test_normalize_aliases_and_prereqs():
    assert normalize_name("WEAF") == "weap"
    assert normalize_name("syrf") == "proc"
    assert normalize_prereq_tokens(["weaf|fixf", "domf"]) == ["weap|fix", "dome"]


def test_normalize_obs_types_unifies_production_item_and_type():
    obs = {
        "tick": 1,
        "available_production": ["weaf"],
        "units": [],
        "buildings": [{"type": "tenf"}],
        "visible_enemies": [],
        "visible_enemy_buildings": [],
        "production": [{"queue_type": "Building", "type": "domf", "progress": 0.4}],
    }
    norm = normalize_obs_types(obs)
    assert norm["available_production"] == ["weap"]
    assert norm["buildings"][0]["type"] == "tent"
    assert norm["production"][0]["item"] == "dome"
    assert norm["production"][0]["type"] == "dome"
    assert get_production_item(norm["production"][0]) == "dome"


def test_build_structure_duplicate_blocked_when_already_in_queue():
    guard = CommandGuard()
    obs = _base_obs()
    obs["production"] = [{"queue_type": "Building", "item": "powr", "progress": 0.3}]
    decision = guard.evaluate("build_structure", {"building_type": "powr"}, obs)
    assert decision.status == "block"
    assert decision.reason == "already_in_queue"


def test_place_building_deferred_when_not_ready():
    guard = CommandGuard()
    obs = _base_obs()
    obs["production"] = [{"queue_type": "Building", "item": "powr", "progress": 0.5}]
    decision = guard.evaluate("place_building", {"building_type": "powr"}, obs)
    assert decision.status == "defer"
    assert decision.reason == "not_ready_to_place"
    assert decision.next_action_hint == "advance_until_ready"


def test_cancel_build_loop_deferred_for_building():
    guard = CommandGuard()

    build_obs = _base_obs(tick=100)
    first_build = guard.evaluate("build_structure", {"building_type": "powr"}, build_obs)
    assert first_build.allowed

    cancel_obs = _base_obs(tick=101)
    cancel_obs["production"] = [{"queue_type": "Building", "item": "powr", "progress": 0.1}]
    cancel_decision = guard.evaluate("cancel_production", {"item_type": "powr"}, cancel_obs)
    assert cancel_decision.allowed

    second_build = guard.evaluate("build_structure", {"building_type": "powr"}, _base_obs(tick=102))
    assert second_build.status == "defer"
    assert second_build.reason == "cancel_build_loop"


def test_unit_control_same_tick_exact_duplicate_is_blocked():
    guard = CommandGuard()
    obs = _base_obs(tick=500)
    params = {"resolved_unit_ids": [1, 2], "target_x": 20, "target_y": 20, "queued": False}

    first = guard.evaluate("move_units", params, obs)
    second = guard.evaluate("move_units", params, obs)

    assert first.allowed
    assert second.status == "block"
    assert second.reason == "duplicate_control_same_tick"


def test_build_unit_count_10_remains_allowed():
    guard = CommandGuard()
    obs = _base_obs(tick=900)
    decision = guard.evaluate("build_unit", {"unit_type": "e1", "count": 10}, obs)
    assert decision.allowed
