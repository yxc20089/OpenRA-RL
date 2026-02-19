"""Tests for pre-game planning phase: opponent intel module and planning MCP tools."""

import time

import pytest

from openra_env.opponent_intel import (
    AI_PROFILES,
    get_opponent_profile,
    get_opponent_summary,
)


# ─── Opponent Intel Module Tests ─────────────────────────────────────────────


class TestAIProfiles:
    def test_all_difficulties_present(self):
        assert "easy" in AI_PROFILES
        assert "normal" in AI_PROFILES
        assert "hard" in AI_PROFILES

    def test_profiles_have_required_fields(self):
        required = {
            "difficulty",
            "display_name",
            "aggressiveness",
            "expansion_tendency",
            "unit_diversity",
            "build_order_quality",
            "estimated_win_rate_vs_new_player",
            "typical_first_attack_tick",
            "behavioral_traits",
            "recommended_counters",
            "typical_army_composition",
            "recent_match_history",
        }
        for difficulty, profile in AI_PROFILES.items():
            missing = required - set(profile.keys())
            assert not missing, f"Profile '{difficulty}' missing fields: {missing}"

    def test_win_rates_are_valid(self):
        for difficulty, profile in AI_PROFILES.items():
            rate = profile["estimated_win_rate_vs_new_player"]
            assert 0.0 <= rate <= 1.0, f"Profile '{difficulty}' has invalid win rate: {rate}"

    def test_attack_ticks_are_positive(self):
        for difficulty, profile in AI_PROFILES.items():
            assert profile["typical_first_attack_tick"] > 0

    def test_army_composition_sums_to_one(self):
        for difficulty, profile in AI_PROFILES.items():
            total = sum(profile["typical_army_composition"].values())
            assert abs(total - 1.0) < 0.01, f"Profile '{difficulty}' army composition sums to {total}"

    def test_match_history_has_results(self):
        for difficulty, profile in AI_PROFILES.items():
            history = profile["recent_match_history"]
            assert len(history) >= 3, f"Profile '{difficulty}' has too few matches"
            for match in history:
                assert match["result"] in ("win", "loss")
                assert match["duration_ticks"] > 0
                assert match["score"] > 0

    def test_normal_ai_is_aggressive(self):
        """Normal AI should be aggressive per user requirements."""
        profile = AI_PROFILES["normal"]
        assert profile["aggressiveness"] == "high"
        assert profile["expansion_tendency"] == "high"

    def test_difficulty_ordering(self):
        """Harder difficulties should have higher win rates and earlier attacks."""
        easy = AI_PROFILES["easy"]
        normal = AI_PROFILES["normal"]
        hard = AI_PROFILES["hard"]
        assert easy["estimated_win_rate_vs_new_player"] < normal["estimated_win_rate_vs_new_player"]
        assert normal["estimated_win_rate_vs_new_player"] < hard["estimated_win_rate_vs_new_player"]
        assert easy["typical_first_attack_tick"] > normal["typical_first_attack_tick"]
        assert normal["typical_first_attack_tick"] > hard["typical_first_attack_tick"]


class TestGetOpponentProfile:
    def test_get_by_difficulty(self):
        for key in ("easy", "normal", "hard"):
            profile = get_opponent_profile(key)
            assert profile is not None
            assert profile["difficulty"].lower() == key

    def test_strips_bot_prefix(self):
        profile = get_opponent_profile("bot_normal")
        assert profile is not None
        assert profile["difficulty"] == "Normal"

    def test_case_insensitive(self):
        assert get_opponent_profile("NORMAL") is not None
        assert get_opponent_profile("Bot_Hard") is not None

    def test_unknown_returns_none(self):
        assert get_opponent_profile("impossible") is None
        assert get_opponent_profile("") is None


class TestGetOpponentSummary:
    def test_summary_contains_key_sections(self):
        summary = get_opponent_summary("normal")
        assert "Opponent Scouting Report" in summary
        assert "Aggressiveness" in summary
        assert "Behavioral traits" in summary
        assert "Recommended counters" in summary
        assert "Win rate" in summary

    def test_summary_for_each_difficulty(self):
        for key in ("easy", "normal", "hard"):
            summary = get_opponent_summary(key)
            assert len(summary) > 100, f"Summary for '{key}' seems too short"

    def test_unknown_returns_error_message(self):
        summary = get_opponent_summary("nonexistent")
        assert "Unknown" in summary

    def test_normal_summary_mentions_aggression(self):
        summary = get_opponent_summary("normal")
        assert "aggressive" in summary.lower()

    def test_normal_summary_mentions_expansion(self):
        summary = get_opponent_summary("normal")
        assert "second base" in summary.lower() or "expand" in summary.lower()


# ─── Planning MCP Tool Tests ────────────────────────────────────────────────


class TestPlanningTools:
    """Test planning phase MCP tools on a bare OpenRAEnvironment."""

    @pytest.fixture
    def env_with_obs(self):
        """Create env with planning support and a cached observation."""
        from fastmcp import FastMCP
        from openra_env.server.openra_environment import OpenRAEnvironment

        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        mcp = FastMCP("openra-test")

        # Set up planning state
        env._planning_enabled = True
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_active = False
        env._planning_start_time = 0.0
        env._planning_turns_used = 0
        env._planning_strategy = ""

        # Set up required env state
        env._player_faction = "russia"
        env._enemy_faction = "england"
        env._unit_groups = {}
        env._pending_placements = {}
        env._attempted_placements = {}
        env._placement_results = []
        env._PLACEABLE_QUEUE_TYPES = {"Building", "Defense"}

        class FakeConfig:
            bot_type = "normal"

        env._config = FakeConfig()

        from openra_env.models import OpenRAState
        env._state = OpenRAState()

        # Cached observation
        env._last_obs = {
            "tick": 0,
            "done": False,
            "result": "",
            "economy": {
                "cash": 10000,
                "ore": 0,
                "power_provided": 0,
                "power_drained": 0,
                "resource_capacity": 5000,
                "harvester_count": 0,
            },
            "military": {
                "units_killed": 0,
                "units_lost": 0,
                "buildings_killed": 0,
                "buildings_lost": 0,
                "army_value": 0,
                "active_unit_count": 1,
            },
            "units": [
                {
                    "actor_id": 100,
                    "type": "mcv",
                    "pos_x": 32768,
                    "pos_y": 32768,
                    "cell_x": 32,
                    "cell_y": 32,
                    "hp_percent": 1.0,
                    "is_idle": True,
                    "current_activity": "",
                    "owner": "Multi0",
                    "can_attack": False,
                    "facing": 0,
                    "experience_level": 0,
                    "stance": 1,
                    "speed": 56,
                    "attack_range": 0,
                    "passenger_count": 0,
                    "ammo": -1,
                    "is_building": False,
                },
            ],
            "buildings": [],
            "production": [],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "map_info": {"width": 64, "height": 64, "map_name": "singles"},
            "available_production": [],
            "spatial_map": "",
            "spatial_channels": 0,
        }

        # _refresh_obs is a no-op for testing (observation already cached)
        env._refresh_obs = lambda: None

        env._register_tools(mcp)
        return env, mcp

    def _get_tool(self, mcp, name):
        """Get a tool function from the MCP tool manager."""
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool = mcp._tool_manager._tools.get(name)
            if tool and hasattr(tool, "fn"):
                return tool.fn
        return None

    def test_planning_tools_registered(self, env_with_obs):
        _, mcp = env_with_obs
        tool_names = set()
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_names = set(mcp._tool_manager._tools.keys())

        assert "get_opponent_intel" in tool_names
        assert "start_planning_phase" in tool_names
        assert "end_planning_phase" in tool_names
        assert "get_planning_status" in tool_names

    def test_tool_count_increased(self, env_with_obs):
        _, mcp = env_with_obs
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            count = len(mcp._tool_manager._tools)
            # Previous: 40 tools. Added: 4 planning + 3 bulk = 47
            assert count == 47, f"Expected 47 tools, got {count}"

    def test_get_opponent_intel(self, env_with_obs):
        env, mcp = env_with_obs
        fn = self._get_tool(mcp, "get_opponent_intel")
        assert fn is not None
        result = fn()
        assert result["difficulty"] == "Normal"
        assert result["aggressiveness"] == "high"
        assert result["your_faction"] == "russia"
        assert result["enemy_faction"] == "england"

    def test_start_planning_phase(self, env_with_obs):
        env, mcp = env_with_obs
        fn = self._get_tool(mcp, "start_planning_phase")
        result = fn()
        assert result["planning_active"] is True
        assert result["max_turns"] == 10
        assert "map" in result
        assert "base_position" in result
        assert "enemy_estimated_position" in result
        assert result["your_faction"] == "russia"
        assert result["your_side"] == "soviet"
        assert result["enemy_faction"] == "england"
        assert "tech_tree" in result
        assert "opponent_intel" in result
        assert "opponent_summary" in result
        assert "instructions" in result
        assert len(result["starting_units"]) == 1
        assert result["starting_units"][0]["type"] == "mcv"
        assert env._planning_active is True

    def test_start_planning_when_disabled(self, env_with_obs):
        env, mcp = env_with_obs
        env._planning_enabled = False
        fn = self._get_tool(mcp, "start_planning_phase")
        result = fn()
        assert result["planning_enabled"] is False
        assert "message" in result
        assert env._planning_active is False

    def test_double_start_planning(self, env_with_obs):
        env, mcp = env_with_obs
        fn = self._get_tool(mcp, "start_planning_phase")
        fn()  # First start
        result = fn()  # Second start
        assert "error" in result
        assert "already active" in result["error"].lower()

    def test_end_planning_phase(self, env_with_obs):
        env, mcp = env_with_obs
        start_fn = self._get_tool(mcp, "start_planning_phase")
        end_fn = self._get_tool(mcp, "end_planning_phase")

        start_fn()
        result = end_fn(strategy="Rush with tanks, build 2 refineries early")

        assert result["planning_complete"] is True
        assert result["strategy_recorded"] is True
        assert "tanks" in result["strategy"]
        assert result["planning_duration_seconds"] >= 0
        assert env._planning_active is False
        assert env._planning_strategy == "Rush with tanks, build 2 refineries early"
        assert env._state.planning_strategy == "Rush with tanks, build 2 refineries early"

    def test_end_planning_without_start(self, env_with_obs):
        env, mcp = env_with_obs
        fn = self._get_tool(mcp, "end_planning_phase")
        result = fn(strategy="some strategy")
        assert "error" in result

    def test_end_planning_empty_strategy(self, env_with_obs):
        env, mcp = env_with_obs
        start_fn = self._get_tool(mcp, "start_planning_phase")
        end_fn = self._get_tool(mcp, "end_planning_phase")

        start_fn()
        result = end_fn()

        assert result["planning_complete"] is True
        assert result["strategy_recorded"] is False
        assert result["strategy"] == ""

    def test_get_planning_status_before_start(self, env_with_obs):
        env, mcp = env_with_obs
        fn = self._get_tool(mcp, "get_planning_status")
        result = fn()
        assert result["planning_active"] is False
        assert "strategy" in result

    def test_get_planning_status_during_planning(self, env_with_obs):
        env, mcp = env_with_obs
        start_fn = self._get_tool(mcp, "start_planning_phase")
        status_fn = self._get_tool(mcp, "get_planning_status")

        start_fn()
        result = status_fn()

        assert result["planning_active"] is True
        assert result["turns_used"] == 0
        assert result["turns_remaining"] == 10
        assert result["time_elapsed_seconds"] >= 0
        assert result["time_remaining_seconds"] > 0

    def test_get_planning_status_when_disabled(self, env_with_obs):
        env, mcp = env_with_obs
        env._planning_enabled = False
        fn = self._get_tool(mcp, "get_planning_status")
        result = fn()
        assert result["planning_enabled"] is False

    def test_game_state_includes_planning_indicator(self, env_with_obs):
        env, mcp = env_with_obs
        start_fn = self._get_tool(mcp, "start_planning_phase")
        get_state_fn = self._get_tool(mcp, "get_game_state")

        start_fn()
        result = get_state_fn()

        assert result.get("planning_active") is True
        assert result.get("planning_turns_remaining") == 10

    def test_game_state_includes_strategy_after_planning(self, env_with_obs):
        env, mcp = env_with_obs
        start_fn = self._get_tool(mcp, "start_planning_phase")
        end_fn = self._get_tool(mcp, "end_planning_phase")
        get_state_fn = self._get_tool(mcp, "get_game_state")

        start_fn()
        end_fn(strategy="Build tanks and attack early")
        result = get_state_fn()

        assert result.get("planning_active") is None or result.get("planning_active") is False
        assert result.get("planning_strategy") == "Build tanks and attack early"

    def test_planning_base_position_from_mcv(self, env_with_obs):
        env, mcp = env_with_obs
        fn = self._get_tool(mcp, "start_planning_phase")
        result = fn()
        # MCV is at cell (32, 32)
        assert result["base_position"]["x"] == 32
        assert result["base_position"]["y"] == 32

    def test_planning_enemy_position_estimate(self, env_with_obs):
        env, mcp = env_with_obs
        fn = self._get_tool(mcp, "start_planning_phase")
        result = fn()
        # Map is 64x64, base at (32, 32), enemy should be opposite
        assert result["enemy_estimated_position"]["x"] == 32  # 64 - 32
        assert result["enemy_estimated_position"]["y"] == 32  # 64 - 32

    def test_start_planning_includes_key_units(self, env_with_obs):
        env, mcp = env_with_obs
        fn = self._get_tool(mcp, "start_planning_phase")
        result = fn()
        assert "key_units" in result
        assert len(result["key_units"]) > 0
        # Key units should have full stats (cost, hp, etc.)
        for utype, udata in result["key_units"].items():
            assert "cost" in udata
            assert "hp" in udata
            assert "name" in udata

    def test_start_planning_includes_key_buildings(self, env_with_obs):
        env, mcp = env_with_obs
        fn = self._get_tool(mcp, "start_planning_phase")
        result = fn()
        assert "key_buildings" in result
        assert len(result["key_buildings"]) > 0
        for btype, bdata in result["key_buildings"].items():
            assert "cost" in bdata
            assert "name" in bdata

    def test_start_planning_instructions_mention_new_tools(self, env_with_obs):
        env, mcp = env_with_obs
        fn = self._get_tool(mcp, "start_planning_phase")
        result = fn()
        instructions = result["instructions"]
        assert "get_faction_briefing" in instructions
        assert "get_map_analysis" in instructions
        assert "batch_lookup" in instructions


# ─── Bulk Knowledge Tool Tests ─────────────────────────────────────────────


class TestBulkKnowledgeTools:
    """Test bulk knowledge tools: get_faction_briefing, get_map_analysis, batch_lookup."""

    @pytest.fixture
    def env_with_obs(self):
        """Create env with planning support and a cached observation."""
        from fastmcp import FastMCP
        from openra_env.server.openra_environment import OpenRAEnvironment

        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        mcp = FastMCP("openra-test")

        env._planning_enabled = True
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_active = False
        env._planning_start_time = 0.0
        env._planning_turns_used = 0
        env._planning_strategy = ""

        env._player_faction = "russia"
        env._enemy_faction = "england"
        env._unit_groups = {}
        env._pending_placements = {}
        env._attempted_placements = {}
        env._placement_results = []
        env._PLACEABLE_QUEUE_TYPES = {"Building", "Defense"}

        class FakeConfig:
            bot_type = "normal"

        env._config = FakeConfig()

        from openra_env.models import OpenRAState
        env._state = OpenRAState()

        env._last_obs = {
            "tick": 0,
            "done": False,
            "result": "",
            "economy": {
                "cash": 10000, "ore": 0, "power_provided": 0,
                "power_drained": 0, "resource_capacity": 5000, "harvester_count": 0,
            },
            "military": {
                "units_killed": 0, "units_lost": 0,
                "buildings_killed": 0, "buildings_lost": 0,
                "army_value": 0, "active_unit_count": 1,
            },
            "units": [
                {
                    "actor_id": 100, "type": "mcv",
                    "pos_x": 32768, "pos_y": 32768,
                    "cell_x": 32, "cell_y": 32,
                    "hp_percent": 1.0, "is_idle": True,
                    "current_activity": "", "owner": "Multi0",
                    "can_attack": False, "facing": 0,
                    "experience_level": 0, "stance": 1,
                    "speed": 56, "attack_range": 0,
                    "passenger_count": 0, "ammo": -1, "is_building": False,
                },
            ],
            "buildings": [],
            "production": [],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "map_info": {"width": 64, "height": 64, "map_name": "singles"},
            "available_production": [],
            "spatial_map": "",
            "spatial_channels": 0,
        }

        env._refresh_obs = lambda: None
        env._register_tools(mcp)
        return env, mcp

    def _get_tool(self, mcp, name):
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool = mcp._tool_manager._tools.get(name)
            if tool and hasattr(tool, "fn"):
                return tool.fn
        return None

    def test_bulk_tools_registered(self, env_with_obs):
        _, mcp = env_with_obs
        tool_names = set()
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_names = set(mcp._tool_manager._tools.keys())
        assert "get_faction_briefing" in tool_names
        assert "get_map_analysis" in tool_names
        assert "batch_lookup" in tool_names

    def test_tool_count_with_bulk_tools(self, env_with_obs):
        _, mcp = env_with_obs
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            count = len(mcp._tool_manager._tools)
            # Previous: 44 tools. Added: 3 bulk tools = 47
            assert count == 47, f"Expected 47 tools, got {count}"

    # ── get_faction_briefing ──

    def test_faction_briefing_returns_units(self, env_with_obs):
        _, mcp = env_with_obs
        fn = self._get_tool(mcp, "get_faction_briefing")
        result = fn()
        assert result["faction"] == "russia"
        assert result["side"] == "soviet"
        assert "units" in result
        assert len(result["units"]) > 10  # Soviet has ~20 units
        assert "e1" in result["units"]  # Rifle Infantry is available to both sides
        assert "3tnk" in result["units"]  # Heavy Tank is soviet

    def test_faction_briefing_returns_buildings(self, env_with_obs):
        _, mcp = env_with_obs
        fn = self._get_tool(mcp, "get_faction_briefing")
        result = fn()
        assert "buildings" in result
        assert len(result["buildings"]) > 10
        assert "powr" in result["buildings"]
        assert "barr" in result["buildings"]
        assert "weap" in result["buildings"]

    def test_faction_briefing_returns_tech_tree(self, env_with_obs):
        _, mcp = env_with_obs
        fn = self._get_tool(mcp, "get_faction_briefing")
        result = fn()
        assert "tech_tree" in result
        assert len(result["tech_tree"]) > 5
        assert result["tech_tree"][0] == "powr"

    def test_faction_briefing_units_have_full_stats(self, env_with_obs):
        _, mcp = env_with_obs
        fn = self._get_tool(mcp, "get_faction_briefing")
        result = fn()
        e1 = result["units"]["e1"]
        assert e1["name"] == "Rifle Infantry"
        assert e1["cost"] == 100
        assert e1["hp"] == 5000
        assert "speed" in e1
        assert "description" in e1

    def test_faction_briefing_excludes_wrong_side(self, env_with_obs):
        _, mcp = env_with_obs
        fn = self._get_tool(mcp, "get_faction_briefing")
        result = fn()
        # Allied-only units should NOT be in soviet briefing
        assert "1tnk" not in result["units"]  # Light Tank is allied-only
        assert "tent" not in result["buildings"]  # Allied barracks

    def test_faction_briefing_allied(self, env_with_obs):
        env, mcp = env_with_obs
        env._player_faction = "england"
        fn = self._get_tool(mcp, "get_faction_briefing")
        result = fn()
        assert result["side"] == "allied"
        assert "1tnk" in result["units"]
        assert "tent" in result["buildings"]
        assert "3tnk" not in result["units"]

    # ── get_map_analysis ──

    def test_map_analysis_no_spatial(self, env_with_obs):
        _, mcp = env_with_obs
        fn = self._get_tool(mcp, "get_map_analysis")
        result = fn()
        assert result["map_name"] == "singles"
        assert result["width"] == 64
        assert result["height"] == 64
        assert "base_position" in result
        assert "enemy_estimated_position" in result
        assert "note" in result  # No spatial data available

    def test_map_analysis_with_spatial(self, env_with_obs):
        """Test map analysis with a small synthetic spatial map."""
        import base64
        import struct

        env, mcp = env_with_obs
        w, h, channels = 4, 4, 9  # Small test map
        data = [0.0] * (w * h * channels)

        # Set up terrain: all passable, some resources
        for y in range(h):
            for x in range(w):
                base_idx = (y * w + x) * channels
                data[base_idx + 0] = 1.0  # terrain
                data[base_idx + 3] = 1.0  # passable

        # Add resources at (1,1) and (2,2)
        data[(1 * w + 1) * channels + 2] = 5.0
        data[(2 * w + 2) * channels + 2] = 3.0

        # Make (3,3) water (impassable)
        data[(3 * w + 3) * channels + 3] = 0.0

        raw_bytes = struct.pack(f"{len(data)}f", *data)
        env._last_obs["spatial_map"] = base64.b64encode(raw_bytes).decode("ascii")
        env._last_obs["spatial_channels"] = channels
        env._last_obs["map_info"] = {"width": w, "height": h, "map_name": "test_map"}

        fn = self._get_tool(mcp, "get_map_analysis")
        result = fn()

        assert result["map_name"] == "test_map"
        assert result["width"] == w
        assert result["height"] == h
        assert "passable_ratio" in result
        assert "has_water" in result
        assert "map_type" in result
        assert "resource_patches" in result
        assert "quadrant_summary" in result
        assert "strategic_notes" in result
        assert result["passable_ratio"] > 0.5

    # ── batch_lookup ──

    def test_batch_lookup_units(self, env_with_obs):
        _, mcp = env_with_obs
        fn = self._get_tool(mcp, "batch_lookup")
        result = fn(queries=[
            {"type": "unit", "name": "e1"},
            {"type": "unit", "name": "3tnk"},
        ])
        assert result["count"] == 2
        assert result["results"][0]["name"] == "Rifle Infantry"
        assert result["results"][1]["name"] == "Heavy Tank"

    def test_batch_lookup_buildings(self, env_with_obs):
        _, mcp = env_with_obs
        fn = self._get_tool(mcp, "batch_lookup")
        result = fn(queries=[
            {"type": "building", "name": "powr"},
            {"type": "building", "name": "weap"},
        ])
        assert result["count"] == 2
        assert result["results"][0]["name"] == "Power Plant"
        assert result["results"][1]["name"] == "War Factory"

    def test_batch_lookup_mixed(self, env_with_obs):
        _, mcp = env_with_obs
        fn = self._get_tool(mcp, "batch_lookup")
        result = fn(queries=[
            {"type": "unit", "name": "e1"},
            {"type": "building", "name": "powr"},
            {"type": "faction", "name": "russia"},
            {"type": "tech_tree", "name": "soviet"},
        ])
        assert result["count"] == 4
        assert result["results"][0]["name"] == "Rifle Infantry"
        assert result["results"][1]["name"] == "Power Plant"
        assert result["results"][2]["display_name"] == "Russia"
        assert "soviet" in result["results"][3]

    def test_batch_lookup_unknown_item(self, env_with_obs):
        _, mcp = env_with_obs
        fn = self._get_tool(mcp, "batch_lookup")
        result = fn(queries=[
            {"type": "unit", "name": "nonexistent"},
            {"type": "unit", "name": "e1"},
        ])
        assert result["count"] == 2
        assert "error" in result["results"][0]
        assert result["results"][1]["name"] == "Rifle Infantry"

    def test_batch_lookup_unknown_type(self, env_with_obs):
        _, mcp = env_with_obs
        fn = self._get_tool(mcp, "batch_lookup")
        result = fn(queries=[{"type": "invalid", "name": "x"}])
        assert "error" in result["results"][0]

    def test_batch_lookup_empty(self, env_with_obs):
        _, mcp = env_with_obs
        fn = self._get_tool(mcp, "batch_lookup")
        result = fn(queries=[])
        assert result["count"] == 0
        assert result["results"] == []
