"""Tests for OpenRAEnvironment using mocked bridge and process manager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openra_env.generated import rl_bridge_pb2
from openra_env.models import ActionType, CommandModel, OpenRAAction
from openra_env.server.openra_environment import OpenRAEnvironment


def _make_proto_observation(tick=0, cash=1000, done=False, result=""):
    """Create a minimal protobuf GameObservation for testing."""
    obs = rl_bridge_pb2.GameObservation()
    obs.tick = tick
    obs.economy.cash = cash
    obs.economy.ore = 0
    obs.economy.power_provided = 100
    obs.economy.power_drained = 50
    obs.economy.resource_capacity = 2000
    obs.economy.harvester_count = 1
    obs.military.units_killed = 0
    obs.military.units_lost = 0
    obs.military.buildings_killed = 0
    obs.military.buildings_lost = 0
    obs.military.army_value = 500
    obs.military.active_unit_count = 3
    obs.map_info.width = 64
    obs.map_info.height = 64
    obs.map_info.map_name = "Test Map"
    obs.done = done
    obs.result = result
    obs.reward = 0.0

    # Add a unit
    unit = obs.units.add()
    unit.actor_id = 1
    unit.type = "e1"
    unit.pos_x = 100
    unit.pos_y = 200
    unit.cell_x = 4
    unit.cell_y = 8
    unit.hp_percent = 1.0
    unit.is_idle = True
    unit.owner = "Player"

    # Add a building
    bldg = obs.buildings.add()
    bldg.actor_id = 10
    bldg.type = "powr"
    bldg.pos_x = 50
    bldg.pos_y = 50
    bldg.hp_percent = 1.0
    bldg.owner = "Player"
    bldg.is_powered = True

    return obs


class TestOpenRAEnvironmentReset:
    @patch("openra_env.server.openra_environment.OpenRAProcessManager")
    @patch("openra_env.server.openra_environment.BridgeClient")
    def test_reset_returns_observation(self, MockBridge, MockProcess):
        mock_bridge = MockBridge.return_value
        mock_bridge.close = AsyncMock()
        mock_bridge.wait_for_ready = AsyncMock(return_value=True)
        mock_bridge.start_session = AsyncMock(return_value=_make_proto_observation(tick=0))

        mock_process = MockProcess.return_value
        mock_process.kill = MagicMock()
        mock_process.launch = MagicMock(return_value=12345)

        env = OpenRAEnvironment(openra_path="/fake/path")
        env._bridge = mock_bridge
        env._process = mock_process

        obs = env.reset()

        assert obs.tick == 0
        assert obs.economy.cash == 1000
        assert len(obs.units) == 1
        assert obs.units[0].type == "e1"
        assert obs.map_info.width == 64
        mock_process.kill.assert_called_once()
        mock_process.launch.assert_called_once()

    @patch("openra_env.server.openra_environment.OpenRAProcessManager")
    @patch("openra_env.server.openra_environment.BridgeClient")
    def test_reset_with_seed(self, MockBridge, MockProcess):
        mock_bridge = MockBridge.return_value
        mock_bridge.close = AsyncMock()
        mock_bridge.wait_for_ready = AsyncMock(return_value=True)
        mock_bridge.start_session = AsyncMock(return_value=_make_proto_observation())

        mock_process = MockProcess.return_value
        mock_process.kill = MagicMock()
        mock_process.launch = MagicMock(return_value=12345)

        env = OpenRAEnvironment(openra_path="/fake/path")
        env._bridge = mock_bridge
        env._process = mock_process

        env.reset(seed=42)
        assert env._config.seed == 42

    @patch("openra_env.server.openra_environment.OpenRAProcessManager")
    @patch("openra_env.server.openra_environment.BridgeClient")
    def test_reset_with_episode_id(self, MockBridge, MockProcess):
        mock_bridge = MockBridge.return_value
        mock_bridge.close = AsyncMock()
        mock_bridge.wait_for_ready = AsyncMock(return_value=True)
        mock_bridge.start_session = AsyncMock(return_value=_make_proto_observation())

        mock_process = MockProcess.return_value
        mock_process.kill = MagicMock()
        mock_process.launch = MagicMock(return_value=12345)

        env = OpenRAEnvironment(openra_path="/fake/path")
        env._bridge = mock_bridge
        env._process = mock_process

        env.reset(episode_id="custom-ep-001")
        assert env.state.episode_id == "custom-ep-001"

    @patch("openra_env.server.openra_environment.OpenRAProcessManager")
    @patch("openra_env.server.openra_environment.BridgeClient")
    def test_reset_raises_if_bridge_not_ready(self, MockBridge, MockProcess):
        mock_bridge = MockBridge.return_value
        mock_bridge.close = AsyncMock()
        mock_bridge.wait_for_ready = AsyncMock(return_value=False)

        mock_process = MockProcess.return_value
        mock_process.kill = MagicMock()
        mock_process.launch = MagicMock()

        env = OpenRAEnvironment(openra_path="/fake/path")
        env._bridge = mock_bridge
        env._process = mock_process

        with pytest.raises(RuntimeError, match="gRPC bridge failed to start"):
            env.reset()


class TestOpenRAEnvironmentStep:
    def _setup_env(self, MockBridge, MockProcess):
        mock_bridge = MockBridge.return_value
        mock_bridge.close = AsyncMock()
        mock_bridge.wait_for_ready = AsyncMock(return_value=True)
        mock_bridge.start_session = AsyncMock(return_value=_make_proto_observation(tick=0))

        mock_process = MockProcess.return_value
        mock_process.kill = MagicMock()
        mock_process.launch = MagicMock(return_value=12345)

        env = OpenRAEnvironment(openra_path="/fake/path")
        env._bridge = mock_bridge
        env._process = mock_process
        return env, mock_bridge, mock_process

    @patch("openra_env.server.openra_environment.OpenRAProcessManager")
    @patch("openra_env.server.openra_environment.BridgeClient")
    def test_step_returns_observation(self, MockBridge, MockProcess):
        env, mock_bridge, _ = self._setup_env(MockBridge, MockProcess)
        env.reset()

        mock_bridge.step = AsyncMock(return_value=_make_proto_observation(tick=10, cash=1500))

        action = OpenRAAction(commands=[CommandModel(action=ActionType.NO_OP)])
        obs = env.step(action)

        assert obs.tick == 10
        assert obs.economy.cash == 1500
        assert env.state.step_count == 1
        assert env.state.game_tick == 10

    @patch("openra_env.server.openra_environment.OpenRAProcessManager")
    @patch("openra_env.server.openra_environment.BridgeClient")
    def test_step_increments_step_count(self, MockBridge, MockProcess):
        env, mock_bridge, _ = self._setup_env(MockBridge, MockProcess)
        env.reset()

        mock_bridge.step = AsyncMock(return_value=_make_proto_observation(tick=10))
        action = OpenRAAction(commands=[CommandModel(action=ActionType.NO_OP)])

        env.step(action)
        assert env.state.step_count == 1

        mock_bridge.step = AsyncMock(return_value=_make_proto_observation(tick=20))
        env.step(action)
        assert env.state.step_count == 2

    @patch("openra_env.server.openra_environment.OpenRAProcessManager")
    @patch("openra_env.server.openra_environment.BridgeClient")
    def test_step_with_multiple_commands(self, MockBridge, MockProcess):
        env, mock_bridge, _ = self._setup_env(MockBridge, MockProcess)
        env.reset()

        mock_bridge.step = AsyncMock(return_value=_make_proto_observation(tick=10))

        action = OpenRAAction(commands=[
            CommandModel(action=ActionType.MOVE, actor_id=1, target_x=10, target_y=20),
            CommandModel(action=ActionType.BUILD, item_type="powr"),
        ])
        obs = env.step(action)
        assert obs.tick == 10

    @patch("openra_env.server.openra_environment.OpenRAProcessManager")
    @patch("openra_env.server.openra_environment.BridgeClient")
    def test_step_terminal_observation(self, MockBridge, MockProcess):
        env, mock_bridge, _ = self._setup_env(MockBridge, MockProcess)
        env.reset()

        mock_bridge.step = AsyncMock(
            return_value=_make_proto_observation(tick=1000, done=True, result="win")
        )

        action = OpenRAAction(commands=[CommandModel(action=ActionType.NO_OP)])
        obs = env.step(action)

        assert obs.done is True
        assert obs.result == "win"
        assert obs.reward > 0  # Should include victory reward


class TestOpenRAEnvironmentState:
    @patch("openra_env.server.openra_environment.OpenRAProcessManager")
    @patch("openra_env.server.openra_environment.BridgeClient")
    def test_initial_state(self, MockBridge, MockProcess):
        env = OpenRAEnvironment(openra_path="/fake/path")
        state = env.state
        assert state.step_count == 0
        assert state.game_tick == 0

    @patch("openra_env.server.openra_environment.OpenRAProcessManager")
    @patch("openra_env.server.openra_environment.BridgeClient")
    def test_state_after_reset(self, MockBridge, MockProcess):
        mock_bridge = MockBridge.return_value
        mock_bridge.close = AsyncMock()
        mock_bridge.wait_for_ready = AsyncMock(return_value=True)
        mock_bridge.start_session = AsyncMock(return_value=_make_proto_observation())

        mock_process = MockProcess.return_value
        mock_process.kill = MagicMock()
        mock_process.launch = MagicMock()

        env = OpenRAEnvironment(openra_path="/fake/path", map_name="test_map")
        env._bridge = mock_bridge
        env._process = mock_process

        env.reset(episode_id="ep-001")

        assert env.state.episode_id == "ep-001"
        assert env.state.map_name == "test_map"
        assert env.state.step_count == 0


class TestOpenRAEnvironmentClose:
    @patch("openra_env.server.openra_environment.OpenRAProcessManager")
    @patch("openra_env.server.openra_environment.BridgeClient")
    def test_close_cleans_up(self, MockBridge, MockProcess):
        mock_bridge = MockBridge.return_value
        mock_bridge.close = AsyncMock()

        mock_process = MockProcess.return_value
        mock_process.kill = MagicMock()

        env = OpenRAEnvironment(openra_path="/fake/path")
        env._bridge = mock_bridge
        env._process = mock_process

        env.close()

        mock_bridge.close.assert_called_once()
        mock_process.kill.assert_called_once()
