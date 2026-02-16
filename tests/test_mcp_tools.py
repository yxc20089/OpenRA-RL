"""Tests for MCP tool registration, game data module, and environment integration."""

from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from openra_env.game_data import (
    RA_BUILDINGS,
    RA_FACTIONS,
    RA_TECH_TREE,
    RA_UNITS,
    get_all_building_types,
    get_all_unit_types,
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
            # 7 read + 1 terrain + 4 knowledge + 27 action + 1 replay = 40
            assert count == 40, f"Expected 40 tools, got {count}"


class TestMCPReadTools:
    """Test read tools return cached observation data."""

    @pytest.fixture
    def env_with_obs(self):
        """Create env with a cached observation."""
        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        from fastmcp import FastMCP
        mcp = FastMCP("openra-test")
        env._register_tools(mcp)

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
