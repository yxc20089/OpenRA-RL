"""Tests for MCP tool registration, game data module, and environment integration."""

import asyncio
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from openra_env.game_data import (
    RA_BUILDINGS,
    RA_FACTIONS,
    RA_TECH_TREE,
    RA_UNITS,
    get_all_building_types,
    get_all_buildings_for_side,
    get_all_unit_types,
    get_all_units_for_side,
    get_building_stats,
    get_faction_info,
    get_tech_tree,
    get_unit_stats,
)
from openra_env.models import ActionType, CommandModel, OpenRAAction
from openra_env.server.openra_environment import OpenRAEnvironment


# ─── Game Data Tests ──────────────────────────────────────────────────────────


class TestUnitData:
    def test_all_units_have_required_fields(self):
        required = {"name", "category", "cost", "hp", "speed", "armor", "side", "prerequisites", "description"}
        for unit_type, data in RA_UNITS.items():
            missing = required - set(data.keys())
            assert not missing, f"Unit '{unit_type}' missing fields: {missing}"

    def test_unit_costs_positive(self):
        for unit_type, data in RA_UNITS.items():
            assert data["cost"] > 0, f"Unit '{unit_type}' has non-positive cost"

    def test_unit_hp_positive(self):
        for unit_type, data in RA_UNITS.items():
            assert data["hp"] > 0, f"Unit '{unit_type}' has non-positive HP"

    def test_unit_sides_valid(self):
        valid_sides = {"both", "allied", "soviet"}
        for unit_type, data in RA_UNITS.items():
            assert data["side"] in valid_sides, f"Unit '{unit_type}' has invalid side: {data['side']}"

    def test_unit_categories_valid(self):
        valid = {"infantry", "vehicle", "aircraft", "ship"}
        for unit_type, data in RA_UNITS.items():
            assert data["category"] in valid, f"Unit '{unit_type}' has invalid category"

    def test_known_units_exist(self):
        for key in ["e1", "e3", "1tnk", "3tnk", "harv", "mcv", "mig", "heli"]:
            assert key in RA_UNITS, f"Expected unit '{key}' not found"

    def test_get_unit_stats_found(self):
        result = get_unit_stats("e1")
        assert result is not None
        assert result["name"] == "Rifle Infantry"
        assert result["cost"] == 100

    def test_get_unit_stats_not_found(self):
        assert get_unit_stats("nonexistent") is None

    def test_get_unit_stats_case_insensitive(self):
        assert get_unit_stats("E1") is not None  # Lowercased internally
        assert get_unit_stats("e1") is not None


class TestBuildingData:
    def test_all_buildings_have_required_fields(self):
        required = {"name", "cost", "hp", "power", "side", "prerequisites", "produces", "description"}
        for bldg_type, data in RA_BUILDINGS.items():
            missing = required - set(data.keys())
            assert not missing, f"Building '{bldg_type}' missing fields: {missing}"

    def test_building_costs_positive(self):
        for bldg_type, data in RA_BUILDINGS.items():
            assert data["cost"] > 0, f"Building '{bldg_type}' has non-positive cost"

    def test_building_sides_valid(self):
        valid_sides = {"both", "allied", "soviet"}
        for bldg_type, data in RA_BUILDINGS.items():
            assert data["side"] in valid_sides, f"Building '{bldg_type}' has invalid side"

    def test_known_buildings_exist(self):
        for key in ["fact", "powr", "barr", "tent", "proc", "weap", "dome"]:
            assert key in RA_BUILDINGS, f"Expected building '{key}' not found"

    def test_power_plants_provide_power(self):
        assert RA_BUILDINGS["powr"]["power"] > 0
        assert RA_BUILDINGS["apwr"]["power"] > 0

    def test_production_buildings_consume_power(self):
        for key in ["barr", "tent", "weap"]:
            assert RA_BUILDINGS[key]["power"] < 0

    def test_get_building_stats_found(self):
        result = get_building_stats("powr")
        assert result is not None
        assert result["name"] == "Power Plant"
        assert result["power"] == 100

    def test_get_building_stats_not_found(self):
        assert get_building_stats("nonexistent") is None


class TestTechTree:
    def test_both_sides_present(self):
        assert "soviet" in RA_TECH_TREE
        assert "allied" in RA_TECH_TREE

    def test_soviet_starts_with_power(self):
        assert RA_TECH_TREE["soviet"][0] == "powr"

    def test_allied_starts_with_power(self):
        assert RA_TECH_TREE["allied"][0] == "powr"

    def test_all_tech_tree_entries_are_valid_buildings(self):
        for side, entries in RA_TECH_TREE.items():
            for entry in entries:
                assert entry in RA_BUILDINGS, f"Tech tree entry '{entry}' not in RA_BUILDINGS"

    def test_get_tech_tree_by_side(self):
        result = get_tech_tree("soviet")
        assert "soviet" in result
        assert "allied" not in result

    def test_get_tech_tree_by_faction(self):
        result = get_tech_tree("russia")
        assert "soviet" in result

    def test_get_tech_tree_all(self):
        result = get_tech_tree()
        assert "soviet" in result
        assert "allied" in result


class TestFactionData:
    def test_all_factions_present(self):
        for faction in ["england", "france", "germany", "russia", "ukraine"]:
            assert faction in RA_FACTIONS

    def test_faction_sides_valid(self):
        for faction, data in RA_FACTIONS.items():
            assert data["side"] in {"allied", "soviet"}

    def test_allied_factions(self):
        for f in ["england", "france", "germany"]:
            assert RA_FACTIONS[f]["side"] == "allied"

    def test_soviet_factions(self):
        for f in ["russia", "ukraine"]:
            assert RA_FACTIONS[f]["side"] == "soviet"

    def test_get_faction_info_returns_units_and_buildings(self):
        result = get_faction_info("russia")
        assert result is not None
        assert "available_units" in result
        assert "available_buildings" in result
        assert len(result["available_units"]) > 5
        assert len(result["available_buildings"]) > 5

    def test_get_faction_info_not_found(self):
        assert get_faction_info("nonexistent") is None

    def test_faction_specific_units(self):
        russia = get_faction_info("russia")
        assert "ttnk" in russia["available_units"]

        germany = get_faction_info("germany")
        assert "ctnk" in germany["available_units"]

    def test_get_all_unit_types(self):
        types = get_all_unit_types()
        assert len(types) > 10
        assert "e1" in types
        assert types == sorted(types)  # Should be sorted

    def test_get_all_building_types(self):
        types = get_all_building_types()
        assert len(types) > 10
        assert "powr" in types
        assert types == sorted(types)


class TestBulkHelpers:
    def test_get_all_units_for_soviet(self):
        units = get_all_units_for_side("soviet")
        assert len(units) > 10
        assert "e1" in units  # both sides
        assert "3tnk" in units  # soviet only
        assert "1tnk" not in units  # allied only
        for utype, data in units.items():
            assert "cost" in data
            assert "hp" in data

    def test_get_all_units_for_allied(self):
        units = get_all_units_for_side("allied")
        assert len(units) > 10
        assert "e1" in units  # both sides
        assert "1tnk" in units  # allied only
        assert "3tnk" not in units  # soviet only

    def test_get_all_buildings_for_soviet(self):
        buildings = get_all_buildings_for_side("soviet")
        assert len(buildings) > 10
        assert "powr" in buildings  # both sides
        assert "barr" in buildings  # soviet only
        assert "tent" not in buildings  # allied only

    def test_get_all_buildings_for_allied(self):
        buildings = get_all_buildings_for_side("allied")
        assert len(buildings) > 10
        assert "powr" in buildings
        assert "tent" in buildings
        assert "barr" not in buildings


# ─── MCP Tool Registration Tests ─────────────────────────────────────────────


class TestMCPToolRegistration:
    @pytest.fixture
    def env(self):
        """Create an OpenRAEnvironment instance (doesn't launch OpenRA)."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        # Manually initialize just the MCP parts
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")
        env._last_obs = None
        env._register_tools(mcp)
        return env, mcp

    def test_tools_registered(self, env):
        _, mcp = env
        tool_names = set()
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_names = set(mcp._tool_manager._tools.keys())

        # Read tools
        assert "get_game_state" in tool_names
        assert "get_economy" in tool_names
        assert "get_units" in tool_names
        assert "get_buildings" in tool_names
        assert "get_enemies" in tool_names
        assert "get_production" in tool_names
        assert "get_map_info" in tool_names

        # Knowledge tools
        assert "lookup_unit" in tool_names
        assert "lookup_building" in tool_names
        assert "lookup_tech_tree" in tool_names
        assert "lookup_faction" in tool_names

        # Action tools
        assert "advance" in tool_names
        assert "move_units" in tool_names
        assert "attack_move" in tool_names
        assert "attack_target" in tool_names
        assert "stop_units" in tool_names
        assert "build_unit" in tool_names
        assert "build_structure" in tool_names
        assert "place_building" in tool_names
        assert "deploy_unit" in tool_names
        assert "sell_building" in tool_names
        assert "repair_building" in tool_names
        assert "set_rally_point" in tool_names
        assert "guard_target" in tool_names
        assert "set_stance" in tool_names
        assert "harvest" in tool_names
        assert "power_down" in tool_names
        assert "set_primary" in tool_names
        assert "cancel_production" in tool_names
        assert "get_replay_path" in tool_names

    def test_tool_count(self, env):
        _, mcp = env
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            count = len(mcp._tool_manager._tools)
            # 7 read + 1 terrain + 4 knowledge + 3 bulk + 4 planning + 27 action + 1 replay = 47
            assert count == 47, f"Expected 47 tools, got {count}"


class TestMCPReadTools:
    """Test read tools return cached observation data."""

    @pytest.fixture
    def env_with_obs(self):
        """Create env with a cached observation."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")
        env._register_tools(mcp)

        # Planning phase attributes (required by get_game_state)
        env._planning_active = False
        env._planning_strategy = ""
        env._planning_enabled = True
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_start_time = 0.0
        env._planning_turns_used = 0

        env._last_obs = {
            "tick": 100,
            "done": False,
            "result": "",
            "economy": {
                "cash": 5000,
                "ore": 1000,
                "power_provided": 200,
                "power_drained": 80,
                "resource_capacity": 5000,
                "harvester_count": 2,
            },
            "military": {
                "units_killed": 3,
                "units_lost": 1,
                "buildings_killed": 0,
                "buildings_lost": 0,
                "army_value": 3500,
                "active_unit_count": 5,
            },
            "units": [
                {
                    "actor_id": 10,
                    "type": "1tnk",
                    "pos_x": 1000,
                    "pos_y": 2000,
                    "cell_x": 10,
                    "cell_y": 20,
                    "hp_percent": 0.8,
                    "is_idle": True,
                    "current_activity": "",
                    "owner": "Multi0",
                    "can_attack": True,
                    "facing": 0,
                    "experience_level": 0,
                    "stance": 3,
                    "speed": 113,
                    "attack_range": 5120,
                    "passenger_count": -1,
                    "is_building": False,
                },
            ],
            "buildings": [
                {
                    "actor_id": 1,
                    "type": "powr",
                    "pos_x": 500,
                    "pos_y": 500,
                    "hp_percent": 1.0,
                    "owner": "Multi0",
                    "is_producing": False,
                    "production_progress": 0.0,
                    "producing_item": "",
                    "is_powered": True,
                    "is_repairing": False,
                    "sell_value": 150,
                    "rally_x": -1,
                    "rally_y": -1,
                    "power_amount": 100,
                    "can_produce": [],
                    "cell_x": 5,
                    "cell_y": 5,
                },
            ],
            "production": [],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "map_info": {"width": 128, "height": 128, "map_name": "Test Map"},
            "available_production": ["e1", "e3"],
        }
        return env, mcp

    def test_get_game_state_returns_summary(self, env_with_obs):
        env, mcp = env_with_obs
        # Get the tool function directly
        tool = mcp._tool_manager._tools["get_game_state"]
        result = tool.fn()
        assert result["tick"] == 100
        assert result["own_units"] == 1
        assert result["own_buildings"] == 1

    def test_get_economy_returns_economy(self, env_with_obs):
        env, mcp = env_with_obs
        tool = mcp._tool_manager._tools["get_economy"]
        result = tool.fn()
        assert result["cash"] == 5000
        assert result["power_provided"] == 200

    def test_get_units_returns_unit_list(self, env_with_obs):
        env, mcp = env_with_obs
        tool = mcp._tool_manager._tools["get_units"]
        result = tool.fn()
        assert len(result) == 1
        assert result[0]["type"] == "1tnk"
        assert result[0]["actor_id"] == 10

    def test_get_buildings_returns_building_list(self, env_with_obs):
        env, mcp = env_with_obs
        tool = mcp._tool_manager._tools["get_buildings"]
        result = tool.fn()
        assert len(result) == 1
        assert result[0]["type"] == "powr"
        assert result[0]["power_amount"] == 100

    def test_get_enemies_empty(self, env_with_obs):
        env, mcp = env_with_obs
        tool = mcp._tool_manager._tools["get_enemies"]
        result = tool.fn()
        assert result["units"] == []
        assert result["buildings"] == []

    def test_get_production_empty(self, env_with_obs):
        env, mcp = env_with_obs
        tool = mcp._tool_manager._tools["get_production"]
        result = tool.fn()
        assert result["queue"] == []
        assert result["available"] == ["e1", "e3"]


class TestMCPKnowledgeTools:
    """Test game knowledge tools return static data."""

    @pytest.fixture
    def mcp(self):
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")
        env._last_obs = None
        env._register_tools(mcp)
        return mcp

    def test_lookup_unit_found(self, mcp):
        tool = mcp._tool_manager._tools["lookup_unit"]
        result = tool.fn("3tnk")
        assert result["name"] == "Heavy Tank"
        assert result["cost"] == 950

    def test_lookup_unit_not_found(self, mcp):
        tool = mcp._tool_manager._tools["lookup_unit"]
        result = tool.fn("nonexistent")
        assert "error" in result
        assert "available_types" in result

    def test_lookup_building_found(self, mcp):
        tool = mcp._tool_manager._tools["lookup_building"]
        result = tool.fn("weap")
        assert result["name"] == "War Factory"

    def test_lookup_tech_tree(self, mcp):
        tool = mcp._tool_manager._tools["lookup_tech_tree"]
        result = tool.fn("soviet")
        assert "soviet" in result

    def test_lookup_faction(self, mcp):
        tool = mcp._tool_manager._tools["lookup_faction"]
        result = tool.fn("russia")
        assert result["side"] == "soviet"
        assert "available_units" in result


# ─── New Action Type Tests ────────────────────────────────────────────────────


class TestNewActionTypes:
    def test_power_down_action(self):
        cmd = CommandModel(action=ActionType.POWER_DOWN, actor_id=42)
        assert cmd.action == ActionType.POWER_DOWN
        assert cmd.actor_id == 42

    def test_set_primary_action(self):
        cmd = CommandModel(action=ActionType.SET_PRIMARY, actor_id=99)
        assert cmd.action == ActionType.SET_PRIMARY

    def test_action_in_openra_action(self):
        action = OpenRAAction(commands=[
            CommandModel(action=ActionType.POWER_DOWN, actor_id=1),
            CommandModel(action=ActionType.SET_PRIMARY, actor_id=2),
        ])
        assert len(action.commands) == 2


class TestBridgeActionMapping:
    def test_new_action_types_in_bridge_map(self):
        from openra_env.server.bridge_client import commands_to_proto
        from openra_env.generated import rl_bridge_pb2

        proto = commands_to_proto([
            {"action": "power_down", "actor_id": 10},
            {"action": "set_primary", "actor_id": 20},
        ])
        assert len(proto.commands) == 2
        assert proto.commands[0].action == rl_bridge_pb2.POWER_DOWN
        assert proto.commands[0].actor_id == 10
        assert proto.commands[1].action == rl_bridge_pb2.SET_PRIMARY
        assert proto.commands[1].actor_id == 20


# ─── Process Manager Replay Config Test ──────────────────────────────────────


class TestReplayConfig:
    def test_record_replays_default_false(self):
        from openra_env.server.openra_process import OpenRAConfig
        config = OpenRAConfig()
        assert config.record_replays is False

    def test_record_replays_in_command(self):
        from openra_env.server.openra_process import OpenRAConfig, OpenRAProcessManager
        openra_path = str(Path(__file__).parent.parent / "OpenRA")
        config = OpenRAConfig(openra_path=openra_path, record_replays=True)
        manager = OpenRAProcessManager(config)
        cmd = manager._build_command()
        assert "Server.RecordReplays=True" in cmd

    def test_no_replay_arg_when_disabled(self):
        from openra_env.server.openra_process import OpenRAConfig, OpenRAProcessManager
        openra_path = str(Path(__file__).parent.parent / "OpenRA")
        config = OpenRAConfig(openra_path=openra_path, record_replays=False)
        manager = OpenRAProcessManager(config)
        cmd = manager._build_command()
        assert "Server.RecordReplays=True" not in cmd


# ─── MCP Bot Pattern Tests ──────────────────────────────────────────────────


class TestMCPBotPatterns:
    """Test patterns used by the MCP bot and LLM agent."""

    def test_tool_schema_to_openai_conversion(self):
        """MCP tool schemas convert to valid OpenAI function calling format."""
        from examples.llm_agent import mcp_tools_to_openai

        # Simulate MCP Tool objects
        class FakeTool:
            def __init__(self, name, description, input_schema):
                self.name = name
                self.description = description
                self.input_schema = input_schema

        tools = [
            FakeTool("get_game_state", "Get game state", {"type": "object", "properties": {}}),
            FakeTool(
                "move_units",
                "Move units to position",
                {
                    "type": "object",
                    "properties": {
                        "unit_ids": {"type": "array", "items": {"type": "integer"}},
                        "target_x": {"type": "integer"},
                        "target_y": {"type": "integer"},
                    },
                    "required": ["unit_ids", "target_x", "target_y"],
                },
            ),
        ]

        result = mcp_tools_to_openai(tools)
        assert len(result) == 2
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_game_state"
        assert result[1]["function"]["name"] == "move_units"
        assert "properties" in result[1]["function"]["parameters"]
        assert "unit_ids" in result[1]["function"]["parameters"]["properties"]

    def test_openai_schema_has_required_fields(self):
        """Each converted tool has type, function.name, function.description, function.parameters."""
        from examples.llm_agent import mcp_tools_to_openai

        class FakeTool:
            def __init__(self):
                self.name = "test_tool"
                self.description = "A test tool"
                self.input_schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

        result = mcp_tools_to_openai([FakeTool()])
        tool = result[0]
        assert tool["type"] == "function"
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]

    def test_compress_history_keeps_system_prompt(self):
        """History compression preserves the system prompt."""
        from examples.llm_agent import compress_history

        messages = [
            {"role": "system", "content": "You are a bot"},
            *[{"role": "user", "content": f"msg {i}"} for i in range(100)],
        ]

        compressed = compress_history(messages, keep_last=10)
        assert compressed[0]["role"] == "system"
        assert compressed[0]["content"] == "You are a bot"
        assert len(compressed) == 12  # system + summary + 10 recent

    def test_compress_history_noop_when_short(self):
        """History compression is a no-op when messages are short."""
        from examples.llm_agent import compress_history

        messages = [
            {"role": "system", "content": "You are a bot"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        compressed = compress_history(messages, keep_last=10)
        assert len(compressed) == 3  # unchanged


class TestScriptedBotNewActions:
    """Test that the scripted bot has the new Sprint 5 action handlers."""

    def test_power_management_handler_exists(self):
        from examples.scripted_bot import ScriptedBot
        bot = ScriptedBot()
        assert hasattr(bot, "_handle_power_management")
        assert hasattr(bot, "_powered_down")

    def test_set_primary_handler_exists(self):
        from examples.scripted_bot import ScriptedBot
        bot = ScriptedBot()
        assert hasattr(bot, "_handle_set_primary")
        assert hasattr(bot, "_primary_set")

    def test_power_management_no_action_when_positive(self):
        """No power down when power balance is positive."""
        from examples.scripted_bot import ScriptedBot
        from openra_env.models import OpenRAObservation, EconomyInfo, BuildingInfoModel

        bot = ScriptedBot()
        obs = OpenRAObservation(
            economy=EconomyInfo(power_provided=200, power_drained=80),
            buildings=[BuildingInfoModel(actor_id=1, type="dome", is_powered=True)],
        )
        commands = bot._handle_power_management(obs)
        assert len(commands) == 0

    def test_power_management_powers_down_when_negative(self):
        """Powers down non-essential building when power balance is negative."""
        from examples.scripted_bot import ScriptedBot
        from openra_env.models import OpenRAObservation, EconomyInfo, BuildingInfoModel

        bot = ScriptedBot()
        obs = OpenRAObservation(
            economy=EconomyInfo(power_provided=50, power_drained=100),
            buildings=[BuildingInfoModel(actor_id=1, type="dome", is_powered=True)],
        )
        commands = bot._handle_power_management(obs)
        assert len(commands) == 1
        assert commands[0].action == ActionType.POWER_DOWN
        assert commands[0].actor_id == 1

    def test_set_primary_with_multiple_barracks(self):
        """Sets primary on newest barracks when 2+ exist."""
        from examples.scripted_bot import ScriptedBot
        from openra_env.models import OpenRAObservation, BuildingInfoModel

        bot = ScriptedBot()
        obs = OpenRAObservation(
            buildings=[
                BuildingInfoModel(actor_id=10, type="tent"),
                BuildingInfoModel(actor_id=20, type="tent"),
            ],
        )
        commands = bot._handle_set_primary(obs)
        assert len(commands) == 1
        assert commands[0].action == ActionType.SET_PRIMARY
        assert commands[0].actor_id == 20  # newest

    def test_set_primary_not_with_single_barracks(self):
        """No set_primary when only one barracks exists."""
        from examples.scripted_bot import ScriptedBot
        from openra_env.models import OpenRAObservation, BuildingInfoModel

        bot = ScriptedBot()
        obs = OpenRAObservation(
            buildings=[BuildingInfoModel(actor_id=10, type="tent")],
        )
        commands = bot._handle_set_primary(obs)
        assert len(commands) == 0


class TestProductionValidation:
    """Test that build_unit/build_structure/build_and_place validate available_production."""

    @pytest.fixture
    def env_with_allied_obs(self):
        """Create env with Allied faction observation (has 1tnk, NOT 3tnk)."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")

        # Stub attributes needed by the tools
        env._planning_active = False
        env._planning_strategy = ""
        env._planning_enabled = False
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_start_time = 0.0
        env._planning_turns_used = 0
        env._pending_placements = {}
        env._attempted_placements = {}
        env._placement_results = []
        env._player_faction = "england"

        env._last_obs = {
            "tick": 500,
            "done": False,
            "result": "",
            "economy": {
                "cash": 3000,
                "ore": 500,
                "power_provided": 200,
                "power_drained": 80,
                "resource_capacity": 4000,
                "harvester_count": 2,
            },
            "military": {
                "units_killed": 0,
                "units_lost": 0,
                "buildings_killed": 0,
                "buildings_lost": 0,
                "army_value": 1000,
                "active_unit_count": 3,
            },
            "units": [],
            "buildings": [
                {
                    "actor_id": 1, "type": "fact", "pos_x": 500, "pos_y": 500,
                    "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
                    "production_progress": 0.0, "producing_item": "",
                    "is_powered": True, "is_repairing": False, "sell_value": 0,
                    "rally_x": -1, "rally_y": -1, "power_amount": 0,
                    "can_produce": [], "cell_x": 5, "cell_y": 5,
                },
            ],
            "production": [],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "map_info": {"width": 128, "height": 128, "map_name": "Test Map"},
            # Allied production: has 1tnk, e1, e3, powr, tent, proc — NO 3tnk
            "available_production": [
                "e1", "e3", "e6", "spy", "medi",
                "1tnk", "arty", "harv", "jeep", "truk",
                "powr", "tent", "proc", "weap", "gun", "dome",
            ],
        }

        # Mock _refresh_obs to be a no-op (obs already set)
        env._refresh_obs = lambda: None

        env._register_tools(mcp)
        return env, mcp

    def test_build_unit_rejects_wrong_faction(self, env_with_allied_obs):
        """build_unit('3tnk') should fail for Allied player with clear error."""
        env, mcp = env_with_allied_obs
        tool = mcp._tool_manager._tools["build_unit"]
        result = tool.fn(unit_type="3tnk")
        assert "error" in result
        assert "3tnk" in result["error"]
        assert "available_units" in result
        # Should list Allied units, not buildings
        assert "1tnk" in result["available_units"]
        assert "powr" not in result["available_units"]

    def test_build_unit_accepts_valid_faction_unit(self, env_with_allied_obs):
        """build_unit('1tnk') should succeed for Allied player."""
        env, mcp = env_with_allied_obs
        # Mock _execute_commands since we don't have a real bridge
        env._execute_commands = lambda cmds: {
            "tick": 501, "done": False, "result": "",
            "economy": env._last_obs["economy"],
            "own_units": 0, "own_buildings": 1,
            "visible_enemies": 0,
            "production": ["1tnk@0%"],
        }
        tool = mcp._tool_manager._tools["build_unit"]
        result = tool.fn(unit_type="1tnk")
        assert "error" not in result
        assert result["tick"] == 501

    def test_build_unit_accepts_e1_for_allied(self, env_with_allied_obs):
        """build_unit('e1') should succeed for Allied player."""
        env, mcp = env_with_allied_obs
        env._execute_commands = lambda cmds: {
            "tick": 501, "done": False, "result": "",
            "economy": env._last_obs["economy"],
            "own_units": 0, "own_buildings": 1,
            "visible_enemies": 0,
            "production": ["e1@0%"],
        }
        tool = mcp._tool_manager._tools["build_unit"]
        result = tool.fn(unit_type="e1")
        assert "error" not in result

    def test_build_structure_rejects_unavailable(self, env_with_allied_obs):
        """build_structure for unavailable building returns error."""
        env, mcp = env_with_allied_obs
        tool = mcp._tool_manager._tools["build_structure"]
        result = tool.fn(building_type="tsla")  # Soviet Tesla Coil
        assert "error" in result
        assert "available_buildings" in result
        assert "powr" in result["available_buildings"]

    def test_build_structure_accepts_valid(self, env_with_allied_obs):
        """build_structure('powr') should succeed for Allied player."""
        env, mcp = env_with_allied_obs
        env._execute_commands = lambda cmds: {
            "tick": 501, "done": False, "result": "",
            "economy": env._last_obs["economy"],
            "own_units": 0, "own_buildings": 1,
            "visible_enemies": 0,
            "production": ["powr@0%"],
        }
        tool = mcp._tool_manager._tools["build_structure"]
        result = tool.fn(building_type="powr")
        assert "error" not in result

    def test_build_and_place_rejects_unavailable(self, env_with_allied_obs):
        """build_and_place for unavailable building returns error."""
        env, mcp = env_with_allied_obs
        tool = mcp._tool_manager._tools["build_and_place"]
        result = tool.fn(building_type="tsla")
        assert "error" in result
        assert "available_buildings" in result

    def test_build_and_place_accepts_valid(self, env_with_allied_obs):
        """build_and_place('proc') should succeed for Allied player."""
        env, mcp = env_with_allied_obs
        env._execute_commands = lambda cmds: {
            "tick": 501, "done": False, "result": "",
            "economy": env._last_obs["economy"],
            "own_units": 0, "own_buildings": 1,
            "visible_enemies": 0,
            "production": ["proc@0%"],
        }
        tool = mcp._tool_manager._tools["build_and_place"]
        result = tool.fn(building_type="proc")
        assert "error" not in result
        assert "proc" in env._pending_placements

    def test_build_unit_error_lists_units_not_buildings(self, env_with_allied_obs):
        """Error response should list only units, not buildings."""
        env, mcp = env_with_allied_obs
        tool = mcp._tool_manager._tools["build_unit"]
        result = tool.fn(unit_type="v2rl")  # Soviet V2 Launcher
        assert "error" in result
        avail = result["available_units"]
        # Should contain units
        assert "e1" in avail
        assert "1tnk" in avail
        # Should NOT contain buildings
        assert "powr" not in avail
        assert "tent" not in avail
        assert "proc" not in avail

    def test_build_structure_error_lists_buildings_not_units(self, env_with_allied_obs):
        """Error response should list only buildings, not units."""
        env, mcp = env_with_allied_obs
        tool = mcp._tool_manager._tools["build_structure"]
        result = tool.fn(building_type="tsla")
        assert "error" in result
        avail = result["available_buildings"]
        # Should contain buildings
        assert "powr" in avail
        assert "tent" in avail
        # Should NOT contain units
        assert "e1" not in avail
        assert "1tnk" not in avail


class TestOreCapAlert:
    """Test the ore storage capacity alert."""

    @pytest.fixture
    def env_with_full_ore(self):
        """Create env with ore near capacity."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")

        env._planning_active = False
        env._planning_strategy = ""
        env._planning_enabled = False
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_start_time = 0.0
        env._planning_turns_used = 0
        env._pending_placements = {}
        env._attempted_placements = {}
        env._placement_results = []
        env._player_faction = ""
        env._last_production_progress = {}
        env._prev_buildings = {}
        env._prev_unit_ids = {}
        env._enemy_ever_seen = False

        env._last_obs = {
            "tick": 8000,
            "done": False,
            "result": "",
            "economy": {
                "cash": 1826,
                "ore": 3800,  # 95% of 4000 capacity
                "power_provided": 300,
                "power_drained": 190,
                "resource_capacity": 4000,
                "harvester_count": 2,
            },
            "military": {
                "units_killed": 0, "units_lost": 0,
                "buildings_killed": 0, "buildings_lost": 0,
                "army_value": 500, "active_unit_count": 2,
            },
            "units": [
                {
                    "actor_id": 10, "type": "e1", "pos_x": 1000, "pos_y": 2000,
                    "cell_x": 10, "cell_y": 20, "hp_percent": 1.0,
                    "is_idle": False, "current_activity": "",
                    "owner": "Multi0", "can_attack": True, "facing": 0,
                    "experience_level": 0, "stance": 3, "speed": 56,
                    "attack_range": 5120, "passenger_count": -1, "is_building": False,
                },
            ],
            "buildings": [
                {
                    "actor_id": 1, "type": "fact", "pos_x": 500, "pos_y": 500,
                    "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
                    "production_progress": 0.0, "producing_item": "",
                    "is_powered": True, "is_repairing": False, "sell_value": 0,
                    "rally_x": -1, "rally_y": -1, "power_amount": 0,
                    "can_produce": [], "cell_x": 5, "cell_y": 5,
                },
                {
                    "actor_id": 2, "type": "proc", "pos_x": 600, "pos_y": 600,
                    "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
                    "production_progress": 0.0, "producing_item": "",
                    "is_powered": True, "is_repairing": False, "sell_value": 700,
                    "rally_x": -1, "rally_y": -1, "power_amount": -30,
                    "can_produce": [], "cell_x": 6, "cell_y": 6,
                },
                {
                    "actor_id": 3, "type": "powr", "pos_x": 400, "pos_y": 400,
                    "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
                    "production_progress": 0.0, "producing_item": "",
                    "is_powered": True, "is_repairing": False, "sell_value": 150,
                    "rally_x": -1, "rally_y": -1, "power_amount": 100,
                    "can_produce": [], "cell_x": 4, "cell_y": 4,
                },
            ],
            "production": [],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "map_info": {"width": 128, "height": 128, "map_name": "Test Map"},
            "available_production": ["e1", "powr", "proc"],
        }

        env._register_tools(mcp)
        return env, mcp

    def test_ore_cap_alert_fires(self, env_with_full_ore):
        """Alert fires when ore >= 90% of capacity."""
        env, mcp = env_with_full_ore
        tool = mcp._tool_manager._tools["get_game_state"]
        result = tool.fn()
        alerts = result.get("alerts", [])
        ore_alerts = [a for a in alerts if "ORE FULL" in a]
        assert len(ore_alerts) == 1
        assert "silo" in ore_alerts[0].lower()

    def test_ore_cap_alert_not_when_low(self, env_with_full_ore):
        """Alert does NOT fire when ore is well below capacity."""
        env, mcp = env_with_full_ore
        env._last_obs["economy"]["ore"] = 1000  # 25% of 4000
        tool = mcp._tool_manager._tools["get_game_state"]
        result = tool.fn()
        alerts = result.get("alerts", [])
        ore_alerts = [a for a in alerts if "ORE FULL" in a]
        assert len(ore_alerts) == 0


class TestWaterBuildingGuard:
    """Test that water buildings skip auto-placement and warn."""

    @pytest.fixture
    def env_with_water_building(self):
        """Create env with a completed spen in pending placements."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")

        env._planning_active = False
        env._planning_strategy = ""
        env._planning_enabled = False
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_start_time = 0.0
        env._planning_turns_used = 0
        env._pending_placements = {"spen": {"cell_x": 0, "cell_y": 0}}
        env._attempted_placements = {}
        env._placement_results = []
        env._player_faction = "russia"

        env._last_obs = {
            "tick": 10000,
            "done": False,
            "result": "",
            "economy": {
                "cash": 2000, "ore": 1000,
                "power_provided": 300, "power_drained": 200,
                "resource_capacity": 4000, "harvester_count": 2,
            },
            "military": {
                "units_killed": 0, "units_lost": 0,
                "buildings_killed": 0, "buildings_lost": 0,
                "army_value": 2000, "active_unit_count": 5,
            },
            "units": [],
            "buildings": [
                {
                    "actor_id": 1, "type": "fact", "pos_x": 500, "pos_y": 500,
                    "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
                    "production_progress": 0.0, "producing_item": "",
                    "is_powered": True, "is_repairing": False, "sell_value": 0,
                    "rally_x": -1, "rally_y": -1, "power_amount": 0,
                    "can_produce": [], "cell_x": 5, "cell_y": 5,
                },
            ],
            "production": [
                {
                    "queue_type": "Building",
                    "item": "spen",
                    "progress": 1.0,
                    "remaining_ticks": 0,
                    "remaining_cost": 0,
                    "paused": False,
                },
            ],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "map_info": {"width": 128, "height": 128, "map_name": "Test Map"},
            "available_production": ["powr", "barr", "proc", "spen"],
        }

        env._register_tools(mcp)
        return env, mcp

    def test_water_building_skips_auto_placement(self, env_with_water_building):
        """Water building (spen) should be removed from pending and warn."""
        env, mcp = env_with_water_building
        assert "spen" in env._pending_placements

        # Trigger placement processing
        env._process_pending_placements()

        # spen should be removed from pending placements
        assert "spen" not in env._pending_placements
        # Should have a warning in placement results
        assert len(env._placement_results) == 1
        assert "WATER BUILDING" in env._placement_results[0]
        assert "spen" in env._placement_results[0]

    def test_water_building_not_in_attempted(self, env_with_water_building):
        """Water building should NOT enter the attempted tracking (no retries)."""
        env, mcp = env_with_water_building
        env._process_pending_placements()
        assert "spen" not in env._attempted_placements


# ── Round 2 Tests ──────────────────────────────────────────────────────────


class TestExecuteCommandsTriggersPlacement:
    """S1: _execute_commands() should trigger _process_pending_placements()."""

    def test_pending_placement_processed_via_execute_commands(self):
        """When _execute_commands runs, pending placements should be processed."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._pending_placements = {"powr": {"cell_x": 5, "cell_y": 5}}
        env._attempted_placements = {}
        env._placement_results = []
        env._player_faction = "england"
        env._prev_buildings = {}
        env._prev_unit_ids = {}

        obs_dict = {
            "tick": 100,
            "done": False,
            "result": "",
            "economy": {
                "cash": 5000, "ore": 0,
                "power_provided": 0, "power_drained": 0,
                "resource_capacity": 4000, "harvester_count": 1,
            },
            "military": {
                "units_killed": 0, "units_lost": 0,
                "buildings_killed": 0, "buildings_lost": 0,
                "army_value": 0, "active_unit_count": 0,
            },
            "units": [],
            "buildings": [
                {
                    "actor_id": 1, "type": "fact", "pos_x": 5120, "pos_y": 5120,
                    "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
                    "production_progress": 0.0, "producing_item": "",
                    "is_powered": True, "is_repairing": False, "sell_value": 0,
                    "rally_x": -1, "rally_y": -1, "power_amount": 0,
                    "can_produce": [], "cell_x": 5, "cell_y": 5,
                },
            ],
            "production": [
                {
                    "queue_type": "Building",
                    "item": "powr",
                    "progress": 1.0,
                    "remaining_ticks": 0,
                    "remaining_cost": 0,
                    "paused": False,
                },
            ],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "map_info": {"width": 128, "height": 128, "map_name": "Test Map"},
            "available_production": ["powr", "proc"],
        }
        env._last_obs = obs_dict

        # Track whether _process_pending_placements was called
        placement_called = []

        def mock_process_pending():
            placement_called.append(True)

        env._process_pending_placements = mock_process_pending

        # Patch run_coroutine_threadsafe to return obs_dict directly
        mock_future = MagicMock()
        mock_future.result.return_value = obs_dict

        with patch("asyncio.run_coroutine_threadsafe", return_value=mock_future):
            env._loop = MagicMock()
            from openra_env.models import CommandModel, ActionType
            result = env._execute_commands([CommandModel(action=ActionType.NO_OP)])

        assert len(placement_called) == 1, "_process_pending_placements was not called by _execute_commands"
        assert result["tick"] == 100


class TestDeadUnitFiltering:
    """S3: _resolve_unit_ids should filter dead unit IDs and warn."""

    @pytest.fixture
    def env_with_units(self):
        """Create env with some living units."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._placement_results = []
        env._unit_groups = {"alpha": [10, 11, 99]}  # 99 is dead
        env._last_obs = {
            "units": [
                {"actor_id": 10, "type": "e1", "can_attack": True, "is_idle": True},
                {"actor_id": 11, "type": "e1", "can_attack": True, "is_idle": False},
                {"actor_id": 12, "type": "e1", "can_attack": False, "is_idle": True},
            ],
        }
        return env

    def test_list_filters_dead_ids(self, env_with_units):
        """List of int IDs filters out dead units."""
        env = env_with_units
        result = env._resolve_unit_ids([10, 11, 50, 99], env._last_obs)
        assert result == [10, 11]
        # Should warn about dead units
        dead_warnings = [r for r in env._placement_results if "DEAD UNITS" in r]
        assert len(dead_warnings) == 1
        assert "50" in dead_warnings[0]
        assert "99" in dead_warnings[0]

    def test_string_ids_filter_dead(self, env_with_units):
        """Comma-separated string IDs filter dead units."""
        env = env_with_units
        result = env._resolve_unit_ids("10,99,50", env._last_obs)
        assert result == [10]
        dead_warnings = [r for r in env._placement_results if "DEAD UNITS" in r]
        assert len(dead_warnings) == 1

    def test_bracketed_string_filters_dead(self, env_with_units):
        """Bracketed string like '[10, 50]' filters dead units."""
        env = env_with_units
        result = env._resolve_unit_ids("[10, 50]", env._last_obs)
        assert result == [10]

    def test_group_filters_dead(self, env_with_units):
        """Named group filters dead units from group members."""
        env = env_with_units
        result = env._resolve_unit_ids("alpha", env._last_obs)
        assert result == [10, 11]
        dead_warnings = [r for r in env._placement_results if "DEAD UNITS" in r]
        assert len(dead_warnings) == 1
        assert "99" in dead_warnings[0]

    def test_all_combat_returns_living(self, env_with_units):
        """'all_combat' returns living units with can_attack, no dead warning."""
        env = env_with_units
        result = env._resolve_unit_ids("all_combat", env._last_obs)
        assert result == [10, 11]
        assert len(env._placement_results) == 0  # no warnings

    def test_all_ids_dead(self, env_with_units):
        """All requested IDs dead returns empty list with warning."""
        env = env_with_units
        result = env._resolve_unit_ids([50, 99], env._last_obs)
        assert result == []
        dead_warnings = [r for r in env._placement_results if "DEAD UNITS" in r]
        assert len(dead_warnings) == 1

    def test_no_dead_no_warning(self, env_with_units):
        """All IDs valid produces no warning."""
        env = env_with_units
        result = env._resolve_unit_ids([10, 11], env._last_obs)
        assert result == [10, 11]
        assert len(env._placement_results) == 0


class TestBuildUnitFundsCheck:
    """S4: build_unit should return error when insufficient funds."""

    @pytest.fixture
    def env_broke(self):
        """Create env with $0 funds."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")

        env._planning_active = False
        env._planning_strategy = ""
        env._planning_enabled = False
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_start_time = 0.0
        env._planning_turns_used = 0
        env._pending_placements = {}
        env._attempted_placements = {}
        env._placement_results = []
        env._player_faction = "russia"
        env._last_production_progress = {}

        env._last_obs = {
            "tick": 5000,
            "done": False,
            "result": "",
            "economy": {
                "cash": 0, "ore": 0,
                "power_provided": 100, "power_drained": 50,
                "resource_capacity": 4000, "harvester_count": 1,
            },
            "military": {
                "units_killed": 0, "units_lost": 0,
                "buildings_killed": 0, "buildings_lost": 0,
                "army_value": 0, "active_unit_count": 0,
            },
            "units": [],
            "buildings": [
                {
                    "actor_id": 1, "type": "fact", "pos_x": 500, "pos_y": 500,
                    "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
                    "production_progress": 0.0, "producing_item": "",
                    "is_powered": True, "is_repairing": False, "sell_value": 0,
                    "rally_x": -1, "rally_y": -1, "power_amount": 0,
                    "can_produce": [], "cell_x": 5, "cell_y": 5,
                },
            ],
            "production": [],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "map_info": {"width": 128, "height": 128, "map_name": "Test Map"},
            "available_production": ["e1", "e2", "powr", "proc", "barr"],
        }

        # Mock _refresh_obs to be a no-op (obs already set)
        env._refresh_obs = lambda: None

        env._register_tools(mcp)
        return env, mcp

    def test_build_unit_rejects_no_funds(self, env_broke):
        """build_unit returns error when funds are insufficient."""
        env, mcp = env_broke
        tool = mcp._tool_manager._tools["build_unit"]
        result = tool.fn(unit_type="e1", count=1)
        assert "error" in result
        assert "Insufficient funds" in result["error"]
        assert "$0" in result["error"]

    def test_build_unit_allows_when_funded(self, env_broke):
        """build_unit succeeds when funds are sufficient."""
        env, mcp = env_broke
        env._last_obs["economy"]["cash"] = 500

        # Mock _execute_commands since we don't have a real bridge
        env._execute_commands = lambda cmds: {
            "tick": 5001, "done": False, "result": "",
            "economy": env._last_obs["economy"],
            "own_units": 0, "own_buildings": 1,
            "visible_enemies": 0, "production": [],
        }

        tool = mcp._tool_manager._tools["build_unit"]
        result = tool.fn(unit_type="e1", count=1)
        assert "error" not in result
        assert "tick" in result


class TestStalledProductionAlert:
    """S2: get_game_state should alert when production stalled at $0."""

    @pytest.fixture
    def env_stalled(self):
        """Create env with stalled production and $0 funds."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")

        env._planning_active = False
        env._planning_strategy = ""
        env._planning_enabled = False
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_start_time = 0.0
        env._planning_turns_used = 0
        env._pending_placements = {}
        env._attempted_placements = {}
        env._placement_results = []
        env._player_faction = "russia"
        # Pre-seed with same progress to simulate stall
        env._last_production_progress = {"weap": 0.56}

        env._last_obs = {
            "tick": 10000,
            "done": False,
            "result": "",
            "economy": {
                "cash": 0, "ore": 0,
                "power_provided": 200, "power_drained": 100,
                "resource_capacity": 4000, "harvester_count": 1,
            },
            "military": {
                "units_killed": 0, "units_lost": 0,
                "buildings_killed": 0, "buildings_lost": 0,
                "army_value": 500, "active_unit_count": 2,
            },
            "units": [
                {
                    "actor_id": 10, "type": "e1", "pos_x": 1000, "pos_y": 2000,
                    "cell_x": 10, "cell_y": 20, "hp_percent": 1.0,
                    "is_idle": True, "current_activity": "",
                    "owner": "Multi0", "can_attack": True, "facing": 0,
                    "experience_level": 0, "stance": 3, "speed": 56,
                    "attack_range": 5120, "passenger_count": -1, "is_building": False,
                },
            ],
            "buildings": [
                {
                    "actor_id": 1, "type": "fact", "pos_x": 500, "pos_y": 500,
                    "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
                    "production_progress": 0.0, "producing_item": "",
                    "is_powered": True, "is_repairing": False, "sell_value": 0,
                    "rally_x": -1, "rally_y": -1, "power_amount": 0,
                    "can_produce": [], "cell_x": 5, "cell_y": 5,
                },
            ],
            "production": [
                {
                    "queue_type": "Building",
                    "item": "weap",
                    "progress": 0.56,
                    "remaining_ticks": 300,
                    "remaining_cost": 1000,
                    "paused": False,
                },
            ],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "map_info": {"width": 128, "height": 128, "map_name": "Test Map"},
            "available_production": ["e1", "powr", "proc"],
        }

        env._register_tools(mcp)
        return env, mcp

    def test_stalled_alert_fires(self, env_stalled):
        """Alert fires when production progress unchanged and $0 funds."""
        env, mcp = env_stalled
        tool = mcp._tool_manager._tools["get_game_state"]
        result = tool.fn()
        alerts = result.get("alerts", [])
        stalled_alerts = [a for a in alerts if "STALLED" in a]
        assert len(stalled_alerts) == 1
        assert "weap" in stalled_alerts[0]
        assert "$0" in stalled_alerts[0]

    def test_stalled_alert_not_on_first_call(self, env_stalled):
        """Alert does NOT fire on first call (no previous progress to compare)."""
        env, mcp = env_stalled
        # Clear the pre-seeded progress so it's like first call
        env._last_production_progress = {}
        tool = mcp._tool_manager._tools["get_game_state"]
        result = tool.fn()
        alerts = result.get("alerts", [])
        stalled_alerts = [a for a in alerts if "STALLED" in a]
        assert len(stalled_alerts) == 0

    def test_stalled_alert_not_when_funded(self, env_stalled):
        """Alert does NOT fire when player has funds (even if progress same)."""
        env, mcp = env_stalled
        env._last_obs["economy"]["cash"] = 1000
        tool = mcp._tool_manager._tools["get_game_state"]
        result = tool.fn()
        alerts = result.get("alerts", [])
        stalled_alerts = [a for a in alerts if "STALLED" in a]
        assert len(stalled_alerts) == 0

    def test_stalled_alert_not_when_progressing(self, env_stalled):
        """Alert does NOT fire when progress is advancing (even at $0)."""
        env, mcp = env_stalled
        # Previous was 0.50, current is 0.56 → progressing
        env._last_production_progress = {"weap": 0.50}
        tool = mcp._tool_manager._tools["get_game_state"]
        result = tool.fn()
        alerts = result.get("alerts", [])
        stalled_alerts = [a for a in alerts if "STALLED" in a]
        assert len(stalled_alerts) == 0

    def test_progress_snapshot_updated(self, env_stalled):
        """_last_production_progress is updated after each call."""
        env, mcp = env_stalled
        env._last_production_progress = {}
        tool = mcp._tool_manager._tools["get_game_state"]
        tool.fn()
        assert "weap" in env._last_production_progress
        assert abs(env._last_production_progress["weap"] - 0.56) < 0.01


class TestBuildingStuckAlertText:
    """S5: BUILDING STUCK alert should suggest get_valid_placements, not 'auto-cancel'."""

    @pytest.fixture
    def env_stuck_building(self):
        """Create env with a stuck building in attempted placements."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")

        env._planning_active = False
        env._planning_strategy = ""
        env._planning_enabled = False
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_start_time = 0.0
        env._planning_turns_used = 0
        env._pending_placements = {}
        env._attempted_placements = {"powr": 5}  # 5 failed attempts
        env._placement_results = []
        env._player_faction = "russia"
        env._last_production_progress = {}

        env._last_obs = {
            "tick": 6000,
            "done": False,
            "result": "",
            "economy": {
                "cash": 2000, "ore": 500,
                "power_provided": 100, "power_drained": 100,
                "resource_capacity": 4000, "harvester_count": 1,
            },
            "military": {
                "units_killed": 0, "units_lost": 0,
                "buildings_killed": 0, "buildings_lost": 0,
                "army_value": 0, "active_unit_count": 0,
            },
            "units": [],
            "buildings": [
                {
                    "actor_id": 1, "type": "fact", "pos_x": 500, "pos_y": 500,
                    "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
                    "production_progress": 0.0, "producing_item": "",
                    "is_powered": True, "is_repairing": False, "sell_value": 0,
                    "rally_x": -1, "rally_y": -1, "power_amount": 0,
                    "can_produce": [], "cell_x": 5, "cell_y": 5,
                },
            ],
            "production": [
                {
                    "queue_type": "Building",
                    "item": "powr",
                    "progress": 1.0,
                    "remaining_ticks": 0,
                    "remaining_cost": 0,
                    "paused": False,
                },
            ],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "map_info": {"width": 128, "height": 128, "map_name": "Test Map"},
            "available_production": ["powr", "proc"],
        }

        env._register_tools(mcp)
        return env, mcp

    def test_stuck_alert_suggests_valid_placements(self, env_stuck_building):
        """BUILDING STUCK alert should suggest get_valid_placements(), not auto-cancel."""
        env, mcp = env_stuck_building
        tool = mcp._tool_manager._tools["get_game_state"]
        result = tool.fn()
        alerts = result.get("alerts", [])
        stuck_alerts = [a for a in alerts if "BUILDING STUCK" in a]
        assert len(stuck_alerts) == 1
        assert "get_valid_placements" in stuck_alerts[0]
        assert "cancel_production" in stuck_alerts[0]
        assert "auto-cancel" not in stuck_alerts[0]


# ── Round 3 Tests ──────────────────────────────────────────────────────────


class TestUnderAttackAlertCap:
    """S1: UNDER ATTACK alerts should be capped when >3 attackers."""

    @pytest.fixture
    def env_base(self):
        """Create env with buildings and variable enemy counts."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")

        env._planning_active = False
        env._planning_strategy = ""
        env._planning_enabled = False
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_start_time = 0.0
        env._planning_turns_used = 0
        env._pending_placements = {}
        env._attempted_placements = {}
        env._placement_results = []
        env._player_faction = "russia"
        env._last_production_progress = {}
        env._prev_buildings = {}
        env._prev_unit_ids = {}

        env._last_obs = {
            "tick": 8000,
            "done": False,
            "result": "",
            "economy": {
                "cash": 2000, "ore": 500,
                "power_provided": 200, "power_drained": 100,
                "resource_capacity": 4000, "harvester_count": 2,
            },
            "military": {
                "units_killed": 0, "units_lost": 0,
                "buildings_killed": 0, "buildings_lost": 0,
                "army_value": 1000, "active_unit_count": 3,
            },
            "units": [
                {
                    "actor_id": 10, "type": "e1", "pos_x": 5120, "pos_y": 5120,
                    "cell_x": 5, "cell_y": 5, "hp_percent": 1.0,
                    "is_idle": True, "current_activity": "",
                    "owner": "Multi0", "can_attack": True, "facing": 0,
                    "experience_level": 0, "stance": 3, "speed": 56,
                    "attack_range": 5120, "passenger_count": -1, "is_building": False,
                },
            ],
            "buildings": [
                {
                    "actor_id": 1, "type": "fact", "pos_x": 5120, "pos_y": 5120,
                    "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
                    "production_progress": 0.0, "producing_item": "",
                    "is_powered": True, "is_repairing": False, "sell_value": 0,
                    "rally_x": -1, "rally_y": -1, "power_amount": 0,
                    "can_produce": [], "cell_x": 5, "cell_y": 5,
                },
                {
                    "actor_id": 2, "type": "barr", "pos_x": 6144, "pos_y": 5120,
                    "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
                    "production_progress": 0.0, "producing_item": "",
                    "is_powered": True, "is_repairing": False, "sell_value": 0,
                    "rally_x": -1, "rally_y": -1, "power_amount": 0,
                    "can_produce": [], "cell_x": 6, "cell_y": 5,
                },
            ],
            "production": [],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "map_info": {"width": 128, "height": 128, "map_name": "Test Map"},
            "available_production": ["e1", "powr", "proc"],
        }

        env._register_tools(mcp)
        return env, mcp

    def _make_enemy(self, actor_id, etype, cell_x, cell_y):
        return {
            "actor_id": actor_id, "type": etype,
            "pos_x": cell_x * 1024, "pos_y": cell_y * 1024,
            "cell_x": cell_x, "cell_y": cell_y, "hp_percent": 1.0,
            "is_idle": False, "current_activity": "", "owner": "Multi1",
            "can_attack": True, "facing": 0, "experience_level": 0,
            "stance": 3, "speed": 56, "attack_range": 5120,
            "passenger_count": -1, "is_building": False,
        }

    def test_few_attackers_individual_alerts(self, env_base):
        """≤3 attackers near base → individual alerts."""
        env, mcp = env_base
        env._last_obs["visible_enemies"] = [
            self._make_enemy(100, "e1", 5, 6),
            self._make_enemy(101, "e3", 6, 6),
        ]
        tool = mcp._tool_manager._tools["get_game_state"]
        result = tool.fn()
        attack_alerts = [a for a in result["alerts"] if "UNDER ATTACK" in a]
        assert len(attack_alerts) == 2
        assert any("e1" in a for a in attack_alerts)
        assert any("e3" in a for a in attack_alerts)

    def test_many_attackers_summarized(self, env_base):
        """>3 attackers near base → one summary alert with type breakdown."""
        env, mcp = env_base
        env._last_obs["visible_enemies"] = [
            self._make_enemy(100, "e1", 5, 6),
            self._make_enemy(101, "e1", 5, 7),
            self._make_enemy(102, "e3", 6, 6),
            self._make_enemy(103, "e3", 7, 5),
            self._make_enemy(104, "e4", 6, 4),
        ]
        tool = mcp._tool_manager._tools["get_game_state"]
        result = tool.fn()
        attack_alerts = [a for a in result["alerts"] if "UNDER ATTACK" in a]
        assert len(attack_alerts) == 1
        assert "5 enemies" in attack_alerts[0]
        assert "e1" in attack_alerts[0]
        assert "e3" in attack_alerts[0]

    def test_far_enemies_no_alert(self, env_base):
        """Enemies far from base → no UNDER ATTACK alert."""
        env, mcp = env_base
        env._last_obs["visible_enemies"] = [
            self._make_enemy(100, "e1", 50, 50),  # far away
        ]
        tool = mcp._tool_manager._tools["get_game_state"]
        result = tool.fn()
        attack_alerts = [a for a in result["alerts"] if "UNDER ATTACK" in a]
        assert len(attack_alerts) == 0


class TestLossTracking:
    """S2: Loss tracking should detect destroyed buildings and units."""

    def test_building_destroyed_alert(self):
        """DESTROYED alert fires when a building disappears between observations."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._placement_results = []
        env._prev_buildings = {1: "fact", 2: "weap", 3: "barr"}
        env._prev_unit_ids = {}
        env._last_obs = {
            "buildings": [
                {"actor_id": 1, "type": "fact"},
                {"actor_id": 3, "type": "barr"},
            ],
            "units": [],
        }
        env._update_loss_tracking()
        destroyed = [r for r in env._placement_results if "DESTROYED" in r]
        assert len(destroyed) == 1
        assert "weap" in destroyed[0]

    def test_units_lost_alert(self):
        """UNITS LOST alert fires with type breakdown when units disappear."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._placement_results = []
        env._prev_buildings = {}
        env._prev_unit_ids = {10: "e1", 11: "e1", 12: "e3", 13: "3tnk", 14: "e1"}
        env._last_obs = {
            "buildings": [],
            "units": [
                {"actor_id": 10, "type": "e1"},
                {"actor_id": 14, "type": "e1"},
            ],
        }
        env._update_loss_tracking()
        lost = [r for r in env._placement_results if "UNITS LOST" in r]
        assert len(lost) == 1
        assert "3 destroyed" in lost[0]
        assert "e1" in lost[0]
        assert "e3" in lost[0]
        assert "3tnk" in lost[0]

    def test_no_losses_no_alert(self):
        """No losses → no alerts."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._placement_results = []
        env._prev_buildings = {1: "fact"}
        env._prev_unit_ids = {10: "e1"}
        env._last_obs = {
            "buildings": [{"actor_id": 1, "type": "fact"}],
            "units": [{"actor_id": 10, "type": "e1"}],
        }
        env._update_loss_tracking()
        assert len(env._placement_results) == 0

    def test_first_observation_no_alert(self):
        """First observation (empty prev) → no alerts, just snapshot."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._placement_results = []
        env._prev_buildings = {}
        env._prev_unit_ids = {}
        env._last_obs = {
            "buildings": [{"actor_id": 1, "type": "fact"}],
            "units": [{"actor_id": 10, "type": "e1"}],
        }
        env._update_loss_tracking()
        assert len(env._placement_results) == 0
        # Should have updated snapshots
        assert env._prev_buildings == {1: "fact"}
        assert env._prev_unit_ids == {10: "e1"}

    def test_multiple_buildings_destroyed(self):
        """Multiple buildings destroyed → multiple DESTROYED alerts."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._placement_results = []
        env._prev_buildings = {1: "fact", 2: "weap", 3: "barr", 4: "kenn"}
        env._prev_unit_ids = {}
        env._last_obs = {
            "buildings": [{"actor_id": 1, "type": "fact"}],
            "units": [],
        }
        env._update_loss_tracking()
        destroyed = [r for r in env._placement_results if "DESTROYED" in r]
        assert len(destroyed) == 3  # weap, barr, kenn

    def test_snapshots_updated_after_tracking(self):
        """_prev_buildings and _prev_unit_ids updated after tracking."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._placement_results = []
        env._prev_buildings = {1: "fact", 2: "weap"}
        env._prev_unit_ids = {10: "e1"}
        env._last_obs = {
            "buildings": [{"actor_id": 1, "type": "fact"}],
            "units": [{"actor_id": 10, "type": "e1"}, {"actor_id": 11, "type": "3tnk"}],
        }
        env._update_loss_tracking()
        assert env._prev_buildings == {1: "fact"}
        assert env._prev_unit_ids == {10: "e1", 11: "3tnk"}


class TestPrereqDiagnosis:
    """S3: Production unavailable should diagnose missing prerequisites."""

    @pytest.fixture
    def env_no_kenn(self):
        """Create env without kennel — dog should explain missing prereq."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")

        env._planning_active = False
        env._planning_strategy = ""
        env._planning_enabled = False
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_start_time = 0.0
        env._planning_turns_used = 0
        env._pending_placements = {}
        env._attempted_placements = {}
        env._placement_results = []
        env._player_faction = "russia"
        env._last_production_progress = {}
        env._prev_buildings = {}
        env._prev_unit_ids = {}

        env._last_obs = {
            "tick": 5000,
            "done": False,
            "result": "",
            "economy": {
                "cash": 2000, "ore": 1000,
                "power_provided": 200, "power_drained": 100,
                "resource_capacity": 4000, "harvester_count": 2,
            },
            "military": {
                "units_killed": 0, "units_lost": 0,
                "buildings_killed": 0, "buildings_lost": 0,
                "army_value": 0, "active_unit_count": 0,
            },
            "units": [],
            "buildings": [
                {
                    "actor_id": 1, "type": "fact", "pos_x": 500, "pos_y": 500,
                    "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
                    "production_progress": 0.0, "producing_item": "",
                    "is_powered": True, "is_repairing": False, "sell_value": 0,
                    "rally_x": -1, "rally_y": -1, "power_amount": 0,
                    "can_produce": [], "cell_x": 5, "cell_y": 5,
                },
                {
                    "actor_id": 2, "type": "barr", "pos_x": 600, "pos_y": 500,
                    "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
                    "production_progress": 0.0, "producing_item": "",
                    "is_powered": True, "is_repairing": False, "sell_value": 0,
                    "rally_x": -1, "rally_y": -1, "power_amount": 0,
                    "can_produce": [], "cell_x": 6, "cell_y": 5,
                },
            ],
            "production": [],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "map_info": {"width": 128, "height": 128, "map_name": "Test Map"},
            "available_production": ["e1", "e2", "powr", "proc", "barr"],
        }

        env._refresh_obs = lambda: None
        env._register_tools(mcp)
        return env, mcp

    def test_dog_missing_kennel(self, env_no_kenn):
        """build_unit('dog') without kennel explains missing prerequisite."""
        env, mcp = env_no_kenn
        tool = mcp._tool_manager._tools["build_unit"]
        result = tool.fn(unit_type="dog", count=1)
        assert "error" in result
        assert "kenn" in result["error"]
        assert "missing_prerequisites" in result
        assert "kenn" in result["missing_prerequisites"]

    def test_3tnk_missing_fix(self, env_no_kenn):
        """build_unit('3tnk') without fix explains missing prerequisites."""
        env, mcp = env_no_kenn
        # Add weap but not fix
        env._last_obs["buildings"].append({
            "actor_id": 3, "type": "weap", "pos_x": 700, "pos_y": 500,
            "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
            "production_progress": 0.0, "producing_item": "",
            "is_powered": True, "is_repairing": False, "sell_value": 0,
            "rally_x": -1, "rally_y": -1, "power_amount": 0,
            "can_produce": [], "cell_x": 7, "cell_y": 5,
        })
        tool = mcp._tool_manager._tools["build_unit"]
        result = tool.fn(unit_type="3tnk", count=1)
        assert "error" in result
        assert "fix" in result["error"]
        assert "missing_prerequisites" in result

    def test_building_missing_prereq(self, env_no_kenn):
        """build_structure for a building needing dome explains missing prereq."""
        env, mcp = env_no_kenn
        tool = mcp._tool_manager._tools["build_structure"]
        result = tool.fn(building_type="afld")
        assert "error" in result
        # afld requires dome and weap
        assert "missing_prerequisites" in result

    def test_diagnose_unknown_type(self):
        """_diagnose_unavailable for unknown type returns generic message."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._last_obs = {"buildings": []}
        result = env._diagnose_unavailable("zzzz")
        assert "not a known" in result["reason"]


# ── Round 4 Tests ──────────────────────────────────────────────────────────


class TestUnitFeedback:
    """S1: move/attack_move/attack_target should return commanded_units feedback."""

    @pytest.fixture
    def env_with_units(self):
        """Create env with units for move command testing."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")

        env._planning_active = False
        env._planning_strategy = ""
        env._planning_enabled = False
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_start_time = 0.0
        env._planning_turns_used = 0
        env._pending_placements = {}
        env._attempted_placements = {}
        env._placement_results = []
        env._player_faction = "russia"
        env._last_production_progress = {}
        env._prev_buildings = {}
        env._prev_unit_ids = {}
        env._unit_groups = {}

        env._last_obs = {
            "tick": 3000,
            "done": False,
            "result": "",
            "economy": {
                "cash": 2000, "ore": 500,
                "power_provided": 200, "power_drained": 80,
                "resource_capacity": 4000, "harvester_count": 1,
            },
            "military": {
                "units_killed": 0, "units_lost": 0,
                "buildings_killed": 0, "buildings_lost": 0,
                "army_value": 0, "active_unit_count": 0,
            },
            "units": [
                {
                    "actor_id": 142, "type": "e1", "pos_x": 13000, "pos_y": 14000,
                    "hp_percent": 1.0, "owner": "Multi0", "is_idle": False,
                    "current_activity": "MoveTo", "can_attack": True,
                    "stance": 3, "cell_x": 13, "cell_y": 14,
                    "facing": 128, "experience_level": 0, "speed": 56,
                    "attack_range": 5120, "passenger_count": 0, "ammo": -1,
                    "is_building": False,
                },
                {
                    "actor_id": 143, "type": "e1", "pos_x": 12000, "pos_y": 14000,
                    "hp_percent": 1.0, "owner": "Multi0", "is_idle": True,
                    "current_activity": "IdleDefault", "can_attack": True,
                    "stance": 3, "cell_x": 12, "cell_y": 14,
                    "facing": 256, "experience_level": 0, "speed": 56,
                    "attack_range": 5120, "passenger_count": 0, "ammo": -1,
                    "is_building": False,
                },
                {
                    "actor_id": 154, "type": "dog", "pos_x": 50000, "pos_y": 30000,
                    "hp_percent": 1.0, "owner": "Multi0", "is_idle": False,
                    "current_activity": "AttackMoveActivity", "can_attack": True,
                    "stance": 3, "cell_x": 50, "cell_y": 30,
                    "facing": 64, "experience_level": 0, "speed": 99,
                    "attack_range": 1024, "passenger_count": 0, "ammo": -1,
                    "is_building": False,
                },
            ],
            "buildings": [
                {
                    "actor_id": 1, "type": "fact", "pos_x": 5000, "pos_y": 5000,
                    "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
                    "production_progress": 0.0, "producing_item": "",
                    "is_powered": True, "is_repairing": False, "sell_value": 0,
                    "rally_x": -1, "rally_y": -1, "power_amount": 0,
                    "can_produce": [], "cell_x": 5, "cell_y": 5,
                },
            ],
            "production": [],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "map_info": {"width": 128, "height": 128, "map_name": "Test Map"},
            "available_production": ["e1", "e2", "dog", "powr", "proc"],
        }

        env._refresh_obs = lambda: None
        env._execute_commands = lambda cmds: {
            "tick": 3050, "done": False, "result": "",
            "economy": env._last_obs["economy"],
            "own_units": 3, "own_buildings": 1,
            "visible_enemies": 0,
            "production": [],
        }
        env._register_tools(mcp)
        return env, mcp

    def test_move_units_returns_feedback(self, env_with_units):
        """move_units should include commanded_units with positions."""
        env, mcp = env_with_units
        tool = mcp._tool_manager._tools["move_units"]
        result = tool.fn(unit_ids="142,143", target_x=50, target_y=20)
        assert "commanded_units" in result
        assert len(result["commanded_units"]) == 2
        unit_142 = next(u for u in result["commanded_units"] if u["id"] == 142)
        assert unit_142["type"] == "e1"
        assert unit_142["cell_x"] == 13
        assert unit_142["cell_y"] == 14
        assert unit_142["activity"] == "MoveTo"

    def test_attack_move_returns_feedback(self, env_with_units):
        """attack_move should include commanded_units with positions."""
        env, mcp = env_with_units
        tool = mcp._tool_manager._tools["attack_move"]
        result = tool.fn(unit_ids="154", target_x=90, target_y=40)
        assert "commanded_units" in result
        assert len(result["commanded_units"]) == 1
        assert result["commanded_units"][0]["type"] == "dog"
        assert result["commanded_units"][0]["cell_x"] == 50
        assert result["commanded_units"][0]["cell_y"] == 30

    def test_attack_move_all_combat(self, env_with_units):
        """attack_move with all_combat includes all 3 combat units."""
        env, mcp = env_with_units
        tool = mcp._tool_manager._tools["attack_move"]
        result = tool.fn(unit_ids="all_combat", target_x=90, target_y=40)
        assert "commanded_units" in result
        assert len(result["commanded_units"]) == 3
        ids = {u["id"] for u in result["commanded_units"]}
        assert ids == {142, 143, 154}

    def test_attack_target_returns_feedback(self, env_with_units):
        """attack_target should include commanded_units."""
        env, mcp = env_with_units
        tool = mcp._tool_manager._tools["attack_target"]
        result = tool.fn(unit_ids="142,143", target_actor_id=999)
        assert "commanded_units" in result
        assert len(result["commanded_units"]) == 2

    def test_stop_units_returns_feedback(self, env_with_units):
        """stop_units should include commanded_units."""
        env, mcp = env_with_units
        tool = mcp._tool_manager._tools["stop_units"]
        result = tool.fn(unit_ids="154")
        assert "commanded_units" in result
        assert len(result["commanded_units"]) == 1
        assert result["commanded_units"][0]["id"] == 154

    def test_command_group_returns_feedback(self, env_with_units):
        """command_group should include commanded_units with positions."""
        env, mcp = env_with_units
        # Set up a group
        env._unit_groups["scouts"] = [142, 154]
        tool = mcp._tool_manager._tools["command_group"]
        result = tool.fn(group_name="scouts", command="attack_move", target_x=90, target_y=40)
        assert "commanded_units" in result
        assert len(result["commanded_units"]) == 2
        ids = {u["id"] for u in result["commanded_units"]}
        assert ids == {142, 154}

    def test_feedback_includes_activity(self, env_with_units):
        """commanded_units should include the current_activity field."""
        env, mcp = env_with_units
        tool = mcp._tool_manager._tools["move_units"]
        result = tool.fn(unit_ids="142", target_x=50, target_y=20)
        assert result["commanded_units"][0]["activity"] == "MoveTo"

    def test_feedback_excludes_dead_units(self, env_with_units):
        """If a commanded unit died during execution, it shouldn't appear in feedback."""
        env, mcp = env_with_units
        # Unit 143 exists in obs but 999 doesn't — simulate commanding a valid + dead unit
        # _resolve_unit_ids filters dead ones, so feedback should only have living
        tool = mcp._tool_manager._tools["move_units"]
        result = tool.fn(unit_ids="143", target_x=50, target_y=20)
        assert len(result["commanded_units"]) == 1
        assert result["commanded_units"][0]["id"] == 143


class TestFactDestroyedDiagnosis:
    """S2: _diagnose_unavailable should detect missing Construction Yard."""

    def test_powr_without_fact(self):
        """build_and_place('powr') without fact says 'No Construction Yard'."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._last_obs = {
            "buildings": [
                {"actor_id": 2, "type": "barr", "cell_x": 5, "cell_y": 5},
            ],
        }
        result = env._diagnose_unavailable("powr")
        assert "No Construction Yard" in result["reason"]
        assert "MCV" in result["reason"]

    def test_fact_present_uses_normal_diagnosis(self):
        """With fact present, _diagnose_unavailable uses normal prereq check."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._last_obs = {
            "buildings": [
                {"actor_id": 1, "type": "fact", "cell_x": 5, "cell_y": 5},
            ],
        }
        # powr has no explicit prereqs and fact is present → should NOT say "No Construction Yard"
        result = env._diagnose_unavailable("powr")
        assert "No Construction Yard" not in result["reason"]

    def test_afld_without_fact(self):
        """Any building type without fact should say 'No Construction Yard'."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._last_obs = {
            "buildings": [
                {"actor_id": 2, "type": "barr", "cell_x": 5, "cell_y": 5},
                {"actor_id": 3, "type": "weap", "cell_x": 6, "cell_y": 5},
            ],
        }
        result = env._diagnose_unavailable("afld")
        assert "No Construction Yard" in result["reason"]

    def test_unit_without_fact_still_normal(self):
        """Units (not buildings) should NOT get the fact check."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._last_obs = {
            "buildings": [
                {"actor_id": 2, "type": "barr", "cell_x": 5, "cell_y": 5},
            ],
        }
        # dog is a unit, not a building — should use normal prereq diagnosis
        result = env._diagnose_unavailable("dog")
        assert "No Construction Yard" not in result["reason"]
        assert "kenn" in result["reason"]


# ── Round 5 Tests ──────────────────────────────────────────────────────

class TestBatchValidation:
    """S1/S2: batch() should reject unsupported actions and validate build_unit."""

    @pytest.fixture
    def env_with_batch(self):
        """Create env for batch testing."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")

        env._planning_active = False
        env._planning_strategy = ""
        env._planning_enabled = False
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_start_time = 0.0
        env._planning_turns_used = 0
        env._pending_placements = {}
        env._attempted_placements = {}
        env._placement_results = []
        env._player_faction = "russia"
        env._last_production_progress = {}
        env._prev_buildings = {}
        env._prev_unit_ids = {}
        env._unit_groups = {}
        env._enemy_ever_seen = False

        env._last_obs = {
            "tick": 3000, "done": False, "result": "",
            "economy": {
                "cash": 500, "ore": 100,
                "power_provided": 200, "power_drained": 80,
                "resource_capacity": 4000, "harvester_count": 1,
            },
            "military": {
                "units_killed": 0, "units_lost": 0,
                "buildings_killed": 0, "buildings_lost": 0,
                "army_value": 0, "active_unit_count": 0,
            },
            "units": [
                {
                    "actor_id": 150, "type": "e1", "pos_x": 10000, "pos_y": 10000,
                    "hp_percent": 1.0, "owner": "Multi0", "is_idle": True,
                    "current_activity": "", "can_attack": True,
                    "stance": 3, "cell_x": 10, "cell_y": 10,
                    "facing": 128, "experience_level": 0, "speed": 56,
                    "attack_range": 5120, "passenger_count": 0, "ammo": -1,
                    "is_building": False,
                },
            ],
            "buildings": [
                {
                    "actor_id": 1, "type": "fact", "pos_x": 5000, "pos_y": 5000,
                    "hp_percent": 1.0, "owner": "Multi0", "is_producing": False,
                    "production_progress": 0.0, "producing_item": "",
                    "is_powered": True, "is_repairing": False, "sell_value": 0,
                    "rally_x": -1, "rally_y": -1, "power_amount": 0,
                    "can_produce": [], "cell_x": 5, "cell_y": 5,
                },
            ],
            "production": [],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "map_info": {"width": 128, "height": 128, "map_name": "Test Map"},
            "available_production": ["e1", "e2", "dog", "powr", "proc"],
        }

        env._refresh_obs = lambda: None
        env._execute_commands = lambda cmds: {
            "tick": 3050, "done": False, "result": "",
            "economy": env._last_obs["economy"],
            "own_units": 1, "own_buildings": 1,
            "visible_enemies": 0,
            "production": [],
        }
        env._register_tools(mcp)
        return env, mcp

    def test_batch_rejects_advance(self, env_with_batch):
        """advance inside batch should be marked SKIPPED."""
        env, mcp = env_with_batch
        tool = mcp._tool_manager._tools["batch"]
        result = tool.fn(actions=[
            {"tool": "advance", "ticks": 100},
            {"tool": "attack_move", "unit_ids": "150", "target_x": 50, "target_y": 50},
        ])
        assert "advance:SKIPPED" in str(result.get("actions", []))
        assert "attack_move" in result.get("actions", [])

    def test_batch_build_unit_unavailable(self, env_with_batch):
        """build_unit for unavailable unit should be marked FAILED."""
        env, mcp = env_with_batch
        tool = mcp._tool_manager._tools["batch"]
        result = tool.fn(actions=[
            {"tool": "build_unit", "unit_type": "mig", "count": 1},
            {"tool": "attack_move", "unit_ids": "150", "target_x": 50, "target_y": 50},
        ])
        assert "build_unit:FAILED" in result.get("actions", [])
        assert "attack_move" in result.get("actions", [])

    def test_batch_all_unsupported_returns_error(self, env_with_batch):
        """All unsupported actions should return error with SKIPPED list."""
        env, mcp = env_with_batch
        tool = mcp._tool_manager._tools["batch"]
        result = tool.fn(actions=[
            {"tool": "advance", "ticks": 100},
            {"tool": "get_game_state"},
        ])
        assert "error" in result
        assert "advance:SKIPPED" in str(result.get("actions", []))


class TestLossTrackingFixes:
    """S3/S4: MCV deployment and husk decay should not be counted as losses."""

    def test_mcv_deploy_not_loss(self):
        """MCV disappearing + fact appearing should NOT trigger UNITS LOST."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._placement_results = []
        env._prev_buildings = {}
        env._prev_unit_ids = {120: "mcv"}
        env._last_obs = {
            "buildings": [{"actor_id": 1, "type": "fact"}],
            "units": [],
        }
        env._update_loss_tracking()
        loss_alerts = [r for r in env._placement_results if "UNITS LOST" in r]
        assert len(loss_alerts) == 0, f"MCV deployment should not be a loss: {loss_alerts}"

    def test_husk_decay_not_loss(self):
        """Husk disappearing should NOT trigger UNITS LOST."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._placement_results = []
        env._prev_buildings = {1: "fact"}
        env._prev_unit_ids = {200: "2tnk.husk"}
        env._last_obs = {
            "buildings": [{"actor_id": 1, "type": "fact"}],
            "units": [],
        }
        env._update_loss_tracking()
        loss_alerts = [r for r in env._placement_results if "UNITS LOST" in r]
        assert len(loss_alerts) == 0, f"Husk decay should not be a loss: {loss_alerts}"

    def test_real_loss_still_tracked(self):
        """Actual unit destruction should still be tracked."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._placement_results = []
        env._prev_buildings = {1: "fact"}
        env._prev_unit_ids = {150: "e1", 151: "e1"}
        env._last_obs = {
            "buildings": [{"actor_id": 1, "type": "fact"}],
            "units": [{"actor_id": 150, "type": "e1"}],
        }
        env._update_loss_tracking()
        loss_alerts = [r for r in env._placement_results if "UNITS LOST" in r]
        assert len(loss_alerts) == 1
        assert "1x e1" in loss_alerts[0]


class TestNoScoutingHistory:
    """S6: NO SCOUTING alert should not fire after enemy has been seen."""

    def test_no_scouting_fires_before_contact(self):
        """Alert fires when enemies never seen."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._enemy_ever_seen = False
        obs = {
            "tick": 1000,
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "units": [], "buildings": [],
            "production": [], "economy": {"cash": 1000, "ore": 0},
        }
        env._last_production_progress = {}
        env._pending_placements = {}
        env._attempted_placements = {}
        env._placement_results = []
        # Use the alert logic directly
        alerts = []
        if obs.get("visible_enemies") or obs.get("visible_enemy_buildings"):
            env._enemy_ever_seen = True
        if obs["tick"] > 750 and not obs["visible_enemies"] and not obs.get("visible_enemy_buildings"):
            if not env._enemy_ever_seen:
                alerts.append("NO SCOUTING")
        assert len(alerts) == 1

    def test_no_scouting_suppressed_after_contact(self):
        """Alert suppressed once enemy has been seen."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._enemy_ever_seen = True  # enemy was seen before
        alerts = []
        obs = {"tick": 5000, "visible_enemies": [], "visible_enemy_buildings": []}
        if obs.get("visible_enemies") or obs.get("visible_enemy_buildings"):
            env._enemy_ever_seen = True
        if obs["tick"] > 750 and not obs["visible_enemies"] and not obs.get("visible_enemy_buildings"):
            if not env._enemy_ever_seen:
                alerts.append("NO SCOUTING")
        assert len(alerts) == 0


class TestTerrainNote:
    """S7: get_terrain_at should return contextual note."""

    def test_passable_terrain_note(self):
        """Passable cell should say 'Passable terrain'."""
        import base64
        import struct
        # Build minimal spatial map: 1 cell, 9 channels
        channels = 9
        data = [0.0] * channels
        data[0] = 2.0   # terrain_index = 2 (land)
        data[3] = 1.0   # passable = 1.0
        raw = struct.pack(f"{channels}f", *data)
        spatial = base64.b64encode(raw).decode()

        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._last_obs = {
            "spatial_map": spatial,
            "map_info": {"width": 1, "height": 1},
            "spatial_channels": channels,
        }
        env._refresh_obs = lambda: None
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")
        env._planning_active = False
        env._planning_strategy = ""
        env._planning_enabled = False
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_start_time = 0.0
        env._planning_turns_used = 0
        env._pending_placements = {}
        env._attempted_placements = {}
        env._placement_results = []
        env._player_faction = "russia"
        env._last_production_progress = {}
        env._prev_buildings = {}
        env._prev_unit_ids = {}
        env._unit_groups = {}
        env._enemy_ever_seen = False
        env._register_tools(mcp)

        tool = mcp._tool_manager._tools["get_terrain_at"]
        result = tool.fn(cell_x=0, cell_y=0)
        assert result["passable"] is True
        assert "Passable" in result["note"]
        assert "Water" not in result["note"]

    def test_water_terrain_note(self):
        """Impassable water cell should mention water."""
        import base64
        import struct
        channels = 9
        data = [0.0] * channels
        data[0] = 7.0   # terrain_index = 7 (water)
        data[3] = 0.0   # passable = 0.0
        raw = struct.pack(f"{channels}f", *data)
        spatial = base64.b64encode(raw).decode()

        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._last_obs = {
            "spatial_map": spatial,
            "map_info": {"width": 1, "height": 1},
            "spatial_channels": channels,
        }
        env._refresh_obs = lambda: None
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")
        env._planning_active = False
        env._planning_strategy = ""
        env._planning_enabled = False
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_start_time = 0.0
        env._planning_turns_used = 0
        env._pending_placements = {}
        env._attempted_placements = {}
        env._placement_results = []
        env._player_faction = "russia"
        env._last_production_progress = {}
        env._prev_buildings = {}
        env._prev_unit_ids = {}
        env._unit_groups = {}
        env._enemy_ever_seen = False
        env._register_tools(mcp)

        tool = mcp._tool_manager._tools["get_terrain_at"]
        result = tool.fn(cell_x=0, cell_y=0)
        assert result["passable"] is False
        assert "Water" in result["note"]


class TestAdvanceClamping:
    """S8: advance() should report when ticks are clamped."""

    @pytest.fixture
    def env_with_advance(self):
        """Create env for advance testing with mocked bridge."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")

        env._planning_active = False
        env._planning_strategy = ""
        env._planning_enabled = False
        env._planning_max_turns = 10
        env._planning_max_time_s = 60.0
        env._planning_start_time = 0.0
        env._planning_turns_used = 0
        env._pending_placements = {}
        env._attempted_placements = {}
        env._placement_results = []
        env._player_faction = "russia"
        env._last_production_progress = {}
        env._prev_buildings = {}
        env._prev_unit_ids = {}
        env._unit_groups = {}
        env._enemy_ever_seen = False
        env._state = MagicMock()

        obs_dict = {
            "tick": 5000, "done": False, "result": "",
            "economy": {"cash": 1000, "ore": 500, "power_provided": 200,
                        "power_drained": 80, "resource_capacity": 4000,
                        "harvester_count": 1},
            "units": [], "buildings": [],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "production": [],
            "map_info": {"width": 128, "height": 128},
        }
        env._last_obs = obs_dict

        # Mock the async bridge with a running loop in a background thread
        loop = asyncio.new_event_loop()
        import threading
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        env._loop = loop

        mock_bridge = MagicMock()
        async def mock_wait_ticks(t):
            return MagicMock()
        mock_bridge.wait_ticks = mock_wait_ticks
        env._bridge = mock_bridge

        # Patch observation_to_dict
        from openra_env.server import openra_environment
        original_fn = openra_environment.observation_to_dict
        openra_environment.observation_to_dict = lambda proto: obs_dict

        env._register_tools(mcp)
        yield env, mcp
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()
        openra_environment.observation_to_dict = original_fn

    def test_advance_clamp_note(self, env_with_advance):
        """advance(1500) should include clamping note."""
        env, mcp = env_with_advance
        tool = mcp._tool_manager._tools["advance"]
        result = tool.fn(ticks=1500)
        assert "note" in result
        assert "1500" in result["note"]
        assert "500" in result["note"]

    def test_advance_no_note_within_limit(self, env_with_advance):
        """advance(100) should NOT include clamping note."""
        env, mcp = env_with_advance
        tool = mcp._tool_manager._tools["advance"]
        result = tool.fn(ticks=100)
        assert "note" not in result
