"""Tests for bridge client helper functions.

Tests observation_to_dict and commands_to_proto conversion functions
using mock protobuf objects.
"""

import pytest

from openra_env.server.bridge_client import commands_to_proto, observation_to_dict
from openra_env.generated import rl_bridge_pb2


class TestCommandsToProto:
    def test_no_op(self):
        result = commands_to_proto([{"action": "no_op"}])
        assert len(result.commands) == 1
        assert result.commands[0].action == rl_bridge_pb2.NO_OP

    def test_move_command(self):
        result = commands_to_proto([
            {"action": "move", "actor_id": 42, "target_x": 100, "target_y": 200}
        ])
        cmd = result.commands[0]
        assert cmd.action == rl_bridge_pb2.MOVE
        assert cmd.actor_id == 42
        assert cmd.target_x == 100
        assert cmd.target_y == 200

    def test_attack_command(self):
        result = commands_to_proto([
            {"action": "attack", "actor_id": 10, "target_actor_id": 99}
        ])
        cmd = result.commands[0]
        assert cmd.action == rl_bridge_pb2.ATTACK
        assert cmd.actor_id == 10
        assert cmd.target_actor_id == 99

    def test_build_command(self):
        result = commands_to_proto([
            {"action": "build", "item_type": "powr"}
        ])
        cmd = result.commands[0]
        assert cmd.action == rl_bridge_pb2.BUILD
        assert cmd.item_type == "powr"

    def test_queued_flag(self):
        result = commands_to_proto([
            {"action": "move", "actor_id": 1, "target_x": 10, "target_y": 20, "queued": True}
        ])
        assert result.commands[0].queued is True

    def test_multiple_commands(self):
        result = commands_to_proto([
            {"action": "move", "actor_id": 1, "target_x": 10, "target_y": 20},
            {"action": "attack", "actor_id": 2, "target_actor_id": 50},
            {"action": "build", "item_type": "barr"},
        ])
        assert len(result.commands) == 3
        assert result.commands[0].action == rl_bridge_pb2.MOVE
        assert result.commands[1].action == rl_bridge_pb2.ATTACK
        assert result.commands[2].action == rl_bridge_pb2.BUILD

    def test_unknown_action_defaults_to_noop(self):
        result = commands_to_proto([{"action": "invalid_action"}])
        assert result.commands[0].action == rl_bridge_pb2.NO_OP

    def test_missing_action_defaults_to_noop(self):
        result = commands_to_proto([{}])
        assert result.commands[0].action == rl_bridge_pb2.NO_OP

    def test_all_action_types(self):
        action_types = [
            ("no_op", rl_bridge_pb2.NO_OP),
            ("move", rl_bridge_pb2.MOVE),
            ("attack_move", rl_bridge_pb2.ATTACK_MOVE),
            ("attack", rl_bridge_pb2.ATTACK),
            ("stop", rl_bridge_pb2.STOP),
            ("harvest", rl_bridge_pb2.HARVEST),
            ("build", rl_bridge_pb2.BUILD),
            ("train", rl_bridge_pb2.TRAIN),
            ("deploy", rl_bridge_pb2.DEPLOY),
            ("sell", rl_bridge_pb2.SELL),
            ("repair", rl_bridge_pb2.REPAIR),
            ("place_building", rl_bridge_pb2.PLACE_BUILDING),
            ("cancel_production", rl_bridge_pb2.CANCEL_PRODUCTION),
            ("set_rally_point", rl_bridge_pb2.SET_RALLY_POINT),
        ]
        for action_str, expected_enum in action_types:
            result = commands_to_proto([{"action": action_str}])
            assert result.commands[0].action == expected_enum, f"Failed for {action_str}"

    def test_empty_list(self):
        result = commands_to_proto([])
        assert len(result.commands) == 0

    def test_default_values_for_missing_fields(self):
        result = commands_to_proto([{"action": "move"}])
        cmd = result.commands[0]
        assert cmd.actor_id == 0
        assert cmd.target_actor_id == 0
        assert cmd.target_x == 0
        assert cmd.target_y == 0
        assert cmd.item_type == ""
        assert cmd.queued is False


class TestObservationToDict:
    def _make_observation(self, **kwargs):
        """Create a protobuf GameObservation with given fields."""
        obs = rl_bridge_pb2.GameObservation()
        obs.tick = kwargs.get("tick", 0)
        obs.done = kwargs.get("done", False)
        obs.result = kwargs.get("result", "")
        obs.reward = kwargs.get("reward", 0.0)

        if "economy" in kwargs:
            eco = kwargs["economy"]
            obs.economy.cash = eco.get("cash", 0)
            obs.economy.ore = eco.get("ore", 0)
            obs.economy.power_provided = eco.get("power_provided", 0)
            obs.economy.power_drained = eco.get("power_drained", 0)
            obs.economy.resource_capacity = eco.get("resource_capacity", 0)
            obs.economy.harvester_count = eco.get("harvester_count", 0)

        if "military" in kwargs:
            mil = kwargs["military"]
            obs.military.units_killed = mil.get("units_killed", 0)
            obs.military.units_lost = mil.get("units_lost", 0)
            obs.military.buildings_killed = mil.get("buildings_killed", 0)
            obs.military.buildings_lost = mil.get("buildings_lost", 0)
            obs.military.army_value = mil.get("army_value", 0)
            obs.military.active_unit_count = mil.get("active_unit_count", 0)

        if "map_info" in kwargs:
            mi = kwargs["map_info"]
            obs.map_info.width = mi.get("width", 0)
            obs.map_info.height = mi.get("height", 0)
            obs.map_info.map_name = mi.get("map_name", "")

        for u in kwargs.get("units", []):
            unit = obs.units.add()
            unit.actor_id = u.get("actor_id", 0)
            unit.type = u.get("type", "")
            unit.pos_x = u.get("pos_x", 0)
            unit.pos_y = u.get("pos_y", 0)
            unit.cell_x = u.get("cell_x", 0)
            unit.cell_y = u.get("cell_y", 0)
            unit.hp_percent = u.get("hp_percent", 1.0)
            unit.is_idle = u.get("is_idle", True)
            unit.current_activity = u.get("current_activity", "")
            unit.owner = u.get("owner", "")
            unit.can_attack = u.get("can_attack", False)

        for b in kwargs.get("buildings", []):
            bldg = obs.buildings.add()
            bldg.actor_id = b.get("actor_id", 0)
            bldg.type = b.get("type", "")
            bldg.pos_x = b.get("pos_x", 0)
            bldg.pos_y = b.get("pos_y", 0)
            bldg.hp_percent = b.get("hp_percent", 1.0)
            bldg.owner = b.get("owner", "")
            bldg.is_producing = b.get("is_producing", False)
            bldg.production_progress = b.get("production_progress", 0.0)
            bldg.producing_item = b.get("producing_item", "")
            bldg.is_powered = b.get("is_powered", True)

        for p in kwargs.get("production", []):
            prod = obs.production.add()
            prod.queue_type = p.get("queue_type", "")
            prod.item = p.get("item", "")
            prod.progress = p.get("progress", 0.0)
            prod.remaining_ticks = p.get("remaining_ticks", 0)
            prod.remaining_cost = p.get("remaining_cost", 0)
            prod.paused = p.get("paused", False)

        for ap in kwargs.get("available_production", []):
            obs.available_production.append(ap)

        return obs

    def test_basic_fields(self):
        obs = self._make_observation(tick=42, done=True, result="win", reward=1.5)
        d = observation_to_dict(obs)
        assert d["tick"] == 42
        assert d["done"] is True
        assert d["result"] == "win"
        assert d["reward"] == 1.5

    def test_economy(self):
        obs = self._make_observation(
            economy={"cash": 5000, "power_provided": 100, "power_drained": 60, "harvester_count": 2}
        )
        d = observation_to_dict(obs)
        assert d["economy"]["cash"] == 5000
        assert d["economy"]["power_provided"] == 100
        assert d["economy"]["power_drained"] == 60
        assert d["economy"]["harvester_count"] == 2

    def test_military(self):
        obs = self._make_observation(
            military={"units_killed": 3, "units_lost": 1, "army_value": 5000}
        )
        d = observation_to_dict(obs)
        assert d["military"]["units_killed"] == 3
        assert d["military"]["units_lost"] == 1
        assert d["military"]["army_value"] == 5000

    def test_units(self):
        obs = self._make_observation(
            units=[
                {"actor_id": 1, "type": "e1", "pos_x": 100, "pos_y": 200, "hp_percent": 0.75, "can_attack": True},
                {"actor_id": 2, "type": "1tnk", "is_idle": False, "current_activity": "Move"},
            ]
        )
        d = observation_to_dict(obs)
        assert len(d["units"]) == 2
        assert d["units"][0]["actor_id"] == 1
        assert d["units"][0]["type"] == "e1"
        assert d["units"][0]["hp_percent"] == pytest.approx(0.75)
        assert d["units"][0]["can_attack"] is True
        assert d["units"][1]["is_idle"] is False
        assert d["units"][1]["current_activity"] == "Move"

    def test_buildings(self):
        obs = self._make_observation(
            buildings=[
                {"actor_id": 10, "type": "powr", "is_powered": True},
                {"actor_id": 20, "type": "barr", "is_producing": True, "producing_item": "e1"},
            ]
        )
        d = observation_to_dict(obs)
        assert len(d["buildings"]) == 2
        assert d["buildings"][0]["type"] == "powr"
        assert d["buildings"][1]["is_producing"] is True
        assert d["buildings"][1]["producing_item"] == "e1"

    def test_production(self):
        obs = self._make_observation(
            production=[{"queue_type": "Infantry", "item": "e1", "progress": 0.5, "remaining_ticks": 100}]
        )
        d = observation_to_dict(obs)
        assert len(d["production"]) == 1
        assert d["production"][0]["queue_type"] == "Infantry"
        assert d["production"][0]["progress"] == pytest.approx(0.5)

    def test_visible_enemies(self):
        obs = self._make_observation()
        enemy = obs.visible_enemies.add()
        enemy.actor_id = 99
        enemy.type = "2tnk"
        enemy.owner = "Enemy"
        d = observation_to_dict(obs)
        assert len(d["visible_enemies"]) == 1
        assert d["visible_enemies"][0]["actor_id"] == 99
        assert d["visible_enemies"][0]["owner"] == "Enemy"

    def test_map_info(self):
        obs = self._make_observation(map_info={"width": 128, "height": 128, "map_name": "Test Map"})
        d = observation_to_dict(obs)
        assert d["map_info"]["width"] == 128
        assert d["map_info"]["height"] == 128
        assert d["map_info"]["map_name"] == "Test Map"

    def test_available_production(self):
        obs = self._make_observation(available_production=["e1", "e3", "1tnk"])
        d = observation_to_dict(obs)
        assert d["available_production"] == ["e1", "e3", "1tnk"]

    def test_empty_observation(self):
        obs = self._make_observation()
        d = observation_to_dict(obs)
        assert d["tick"] == 0
        assert d["units"] == []
        assert d["buildings"] == []
        assert d["production"] == []
        assert d["visible_enemies"] == []
        assert d["done"] is False
        assert d["result"] == ""
