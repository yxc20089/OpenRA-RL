"""Tests for OpenRA-RL Pydantic models."""

import pytest

from openra_env.models import (
    ActionType,
    BuildingInfoModel,
    CommandModel,
    EconomyInfo,
    MapInfoModel,
    MilitaryInfo,
    OpenRAAction,
    OpenRAObservation,
    OpenRAState,
    ProductionInfoModel,
    UnitInfoModel,
)


class TestActionType:
    def test_enum_values(self):
        assert ActionType.NO_OP == "no_op"
        assert ActionType.MOVE == "move"
        assert ActionType.ATTACK == "attack"
        assert ActionType.BUILD == "build"
        assert ActionType.TRAIN == "train"

    def test_enum_from_string(self):
        assert ActionType("move") == ActionType.MOVE
        assert ActionType("no_op") == ActionType.NO_OP

    def test_all_action_types_exist(self):
        expected = {
            "no_op", "move", "attack_move", "attack", "stop",
            "harvest", "build", "train", "deploy", "sell",
            "repair", "place_building", "cancel_production", "set_rally_point",
            "guard", "set_stance", "enter_transport", "unload",
            "power_down", "set_primary",
        }
        actual = {a.value for a in ActionType}
        assert actual == expected


class TestCommandModel:
    def test_minimal_command(self):
        cmd = CommandModel(action=ActionType.NO_OP)
        assert cmd.action == ActionType.NO_OP
        assert cmd.actor_id == 0
        assert cmd.target_x == 0
        assert cmd.queued is False

    def test_move_command(self):
        cmd = CommandModel(
            action=ActionType.MOVE,
            actor_id=42,
            target_x=100,
            target_y=200,
        )
        assert cmd.action == ActionType.MOVE
        assert cmd.actor_id == 42
        assert cmd.target_x == 100
        assert cmd.target_y == 200

    def test_attack_command(self):
        cmd = CommandModel(
            action=ActionType.ATTACK,
            actor_id=10,
            target_actor_id=99,
        )
        assert cmd.target_actor_id == 99

    def test_build_command(self):
        cmd = CommandModel(
            action=ActionType.BUILD,
            item_type="powr",
        )
        assert cmd.item_type == "powr"

    def test_serialization_roundtrip(self):
        cmd = CommandModel(
            action=ActionType.MOVE,
            actor_id=5,
            target_x=10,
            target_y=20,
            queued=True,
        )
        data = cmd.model_dump()
        restored = CommandModel(**data)
        assert restored == cmd


class TestOpenRAAction:
    def test_empty_action(self):
        action = OpenRAAction()
        assert action.commands == []

    def test_single_command(self):
        action = OpenRAAction(
            commands=[CommandModel(action=ActionType.NO_OP)]
        )
        assert len(action.commands) == 1

    def test_multiple_commands(self):
        action = OpenRAAction(
            commands=[
                CommandModel(action=ActionType.MOVE, actor_id=1, target_x=10, target_y=20),
                CommandModel(action=ActionType.ATTACK, actor_id=2, target_actor_id=99),
                CommandModel(action=ActionType.BUILD, item_type="powr"),
            ]
        )
        assert len(action.commands) == 3
        assert action.commands[0].action == ActionType.MOVE
        assert action.commands[1].action == ActionType.ATTACK
        assert action.commands[2].action == ActionType.BUILD

    def test_serialization_roundtrip(self):
        action = OpenRAAction(
            commands=[
                CommandModel(action=ActionType.MOVE, actor_id=1, target_x=10, target_y=20),
            ]
        )
        data = action.model_dump()
        restored = OpenRAAction(**data)
        assert len(restored.commands) == 1
        assert restored.commands[0].actor_id == 1


class TestEconomyInfo:
    def test_defaults(self):
        eco = EconomyInfo()
        assert eco.cash == 0
        assert eco.ore == 0
        assert eco.power_provided == 0
        assert eco.power_drained == 0
        assert eco.resource_capacity == 0
        assert eco.harvester_count == 0

    def test_with_values(self):
        eco = EconomyInfo(cash=5000, power_provided=100, power_drained=60, harvester_count=2)
        assert eco.cash == 5000
        assert eco.power_provided == 100
        assert eco.power_drained == 60
        assert eco.harvester_count == 2


class TestMilitaryInfo:
    def test_defaults(self):
        mil = MilitaryInfo()
        assert mil.units_killed == 0
        assert mil.units_lost == 0
        assert mil.army_value == 0

    def test_with_values(self):
        mil = MilitaryInfo(units_killed=5, units_lost=2, army_value=3000)
        assert mil.units_killed == 5
        assert mil.units_lost == 2
        assert mil.army_value == 3000


class TestUnitInfoModel:
    def test_required_fields(self):
        unit = UnitInfoModel(actor_id=1, type="e1")
        assert unit.actor_id == 1
        assert unit.type == "e1"
        assert unit.hp_percent == 1.0
        assert unit.is_idle is True

    def test_full_unit(self):
        unit = UnitInfoModel(
            actor_id=42,
            type="1tnk",
            pos_x=1024,
            pos_y=2048,
            cell_x=4,
            cell_y=8,
            hp_percent=0.75,
            is_idle=False,
            current_activity="Attack",
            owner="Nod",
            can_attack=True,
        )
        assert unit.hp_percent == 0.75
        assert unit.is_idle is False
        assert unit.can_attack is True


class TestBuildingInfoModel:
    def test_required_fields(self):
        bldg = BuildingInfoModel(actor_id=10, type="powr")
        assert bldg.actor_id == 10
        assert bldg.type == "powr"
        assert bldg.is_powered is True

    def test_producing_building(self):
        bldg = BuildingInfoModel(
            actor_id=20,
            type="barr",
            is_producing=True,
            production_progress=0.5,
            producing_item="e1",
        )
        assert bldg.is_producing is True
        assert bldg.producing_item == "e1"


class TestProductionInfoModel:
    def test_required_fields(self):
        prod = ProductionInfoModel(queue_type="Infantry", item="e1")
        assert prod.queue_type == "Infantry"
        assert prod.item == "e1"
        assert prod.progress == 0.0
        assert prod.paused is False


class TestMapInfoModel:
    def test_defaults(self):
        m = MapInfoModel()
        assert m.width == 0
        assert m.height == 0
        assert m.map_name == ""

    def test_with_values(self):
        m = MapInfoModel(width=128, height=128, map_name="Allied vs Soviet")
        assert m.width == 128
        assert m.map_name == "Allied vs Soviet"


class TestOpenRAObservation:
    def test_default_observation(self):
        obs = OpenRAObservation()
        assert obs.tick == 0
        assert obs.units == []
        assert obs.buildings == []
        assert obs.done is False
        assert obs.result == ""

    def test_full_observation(self):
        obs = OpenRAObservation(
            tick=150,
            economy=EconomyInfo(cash=5000, power_provided=100),
            military=MilitaryInfo(units_killed=3),
            units=[
                UnitInfoModel(actor_id=1, type="e1"),
                UnitInfoModel(actor_id=2, type="1tnk"),
            ],
            buildings=[
                BuildingInfoModel(actor_id=10, type="powr"),
            ],
            production=[
                ProductionInfoModel(queue_type="Infantry", item="e1", progress=0.5),
            ],
            visible_enemies=[
                UnitInfoModel(actor_id=99, type="e1", owner="Enemy"),
            ],
            map_info=MapInfoModel(width=128, height=128),
            available_production=["e1", "e3", "1tnk"],
            done=False,
            reward=0.5,
            result="",
        )
        assert obs.tick == 150
        assert len(obs.units) == 2
        assert len(obs.buildings) == 1
        assert len(obs.production) == 1
        assert len(obs.visible_enemies) == 1
        assert obs.economy.cash == 5000
        assert obs.available_production == ["e1", "e3", "1tnk"]

    def test_terminal_observation(self):
        obs = OpenRAObservation(done=True, result="win", reward=1.0)
        assert obs.done is True
        assert obs.result == "win"

    def test_serialization_roundtrip(self):
        obs = OpenRAObservation(
            tick=100,
            economy=EconomyInfo(cash=3000),
            units=[UnitInfoModel(actor_id=1, type="e1")],
        )
        data = obs.model_dump()
        restored = OpenRAObservation(**data)
        assert restored.tick == 100
        assert restored.economy.cash == 3000
        assert len(restored.units) == 1


class TestOpenRAState:
    def test_defaults(self):
        state = OpenRAState()
        assert state.game_tick == 0
        assert state.map_name == ""
        assert state.opponent_type == "bot_normal"
        assert state.step_count == 0

    def test_with_values(self):
        state = OpenRAState(
            episode_id="abc123",
            step_count=50,
            game_tick=500,
            map_name="Test Map",
            opponent_type="bot_hard",
        )
        assert state.episode_id == "abc123"
        assert state.step_count == 50
        assert state.game_tick == 500

    def test_serialization_roundtrip(self):
        state = OpenRAState(episode_id="test", game_tick=100)
        data = state.model_dump()
        restored = OpenRAState(**data)
        assert restored.episode_id == "test"
        assert restored.game_tick == 100
