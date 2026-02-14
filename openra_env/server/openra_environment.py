"""OpenRA Environment server implementing the OpenEnv Environment interface.

This is the core environment that manages OpenRA game instances,
translates between the OpenEnv API and the gRPC bridge protocol,
and computes rewards.
"""

import asyncio
import logging
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from openra_env.models import (
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
from openra_env.reward import OpenRARewardFunction, RewardWeights
from openra_env.server.bridge_client import BridgeClient, commands_to_proto, observation_to_dict
from openra_env.server.openra_process import OpenRAConfig, OpenRAProcessManager

logger = logging.getLogger(__name__)


class OpenRAEnvironment(Environment):
    """OpenRA RL Environment.

    Manages OpenRA game instances and provides a Gymnasium-style API
    (reset/step/state) for training RL agents.

    Each reset() launches a new OpenRA subprocess with the ExternalBotBridge
    trait enabled, connects via gRPC, and returns the initial observation.
    Each step() sends actions and returns the next observation with reward.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        openra_path: Optional[str] = None,
        mod: str = "ra",
        map_name: str = "singles.oramap",
        grpc_port: int = 9999,
        bot_type: str = "normal",
        reward_weights: Optional[RewardWeights] = None,
    ):
        super().__init__()
        self._config = OpenRAConfig(
            openra_path=openra_path or OpenRAConfig.openra_path,
            mod=mod,
            map_name=map_name,
            grpc_port=grpc_port,
            bot_type=bot_type,
        )
        self._process = OpenRAProcessManager(self._config)
        self._bridge = BridgeClient(port=grpc_port)
        self._reward_fn = OpenRARewardFunction(weights=reward_weights)
        self._state = OpenRAState()
        # Persistent event loop for async gRPC bridge operations.
        # The gRPC streaming state must stay within the same loop.
        self._loop = asyncio.new_event_loop()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OpenRAObservation:
        """Reset the environment for a new episode.

        1. Kill any existing OpenRA process
        2. Launch a new OpenRA instance with ExternalBotBridge
        3. Connect gRPC bridge
        4. Return initial observation
        """
        return self._loop.run_until_complete(self._async_reset(seed, episode_id, **kwargs))

    async def _async_reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OpenRAObservation:
        # Clean up previous episode
        await self._bridge.close()
        self._process.kill()

        # Initialize new episode state
        ep_id = episode_id or str(uuid.uuid4())
        self._state = OpenRAState(
            episode_id=ep_id,
            step_count=0,
            game_tick=0,
            map_name=self._config.map_name,
            opponent_type=f"bot_{self._config.bot_type}",
        )
        self._reward_fn.reset()

        # Update seed if provided
        if seed is not None:
            self._config.seed = seed

        # Launch OpenRA
        logger.info(f"Launching OpenRA: map={self._config.map_name}, mod={self._config.mod}")
        self._process.launch()
        logger.info(f"OpenRA process launched (PID={self._process.pid})")

        # Wait for gRPC server to be ready
        # Software rendering (llvmpipe in Docker) can take 60-120s to start
        logger.info("Waiting for gRPC bridge to become ready...")
        ready = await self._bridge.wait_for_ready(max_retries=120, retry_interval=2.0)
        if not ready:
            # Log process status for debugging
            alive = self._process.is_alive()
            logger.error(f"Bridge failed to start. Process alive={alive}")
            raise RuntimeError("OpenRA gRPC bridge failed to start")

        # Start streaming session and get initial observation
        proto_obs = await self._bridge.start_session()
        obs_dict = observation_to_dict(proto_obs)

        # Compute initial reward (should be 0)
        reward = self._reward_fn.compute(obs_dict)

        return self._build_observation(obs_dict, reward)

    def step(
        self,
        action: OpenRAAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> OpenRAObservation:
        """Execute an action and return the next observation."""
        return self._loop.run_until_complete(self._async_step(action))

    async def _async_step(self, action: OpenRAAction) -> OpenRAObservation:
        self._state.step_count += 1

        # Convert action to protobuf
        cmd_dicts = [cmd.model_dump() for cmd in action.commands]
        proto_action = commands_to_proto(cmd_dicts)

        # Send action and receive next observation
        proto_obs = await self._bridge.step(proto_action)
        obs_dict = observation_to_dict(proto_obs)

        # Update state
        self._state.game_tick = obs_dict["tick"]

        # Compute reward
        reward = self._reward_fn.compute(obs_dict)

        return self._build_observation(obs_dict, reward)

    @property
    def state(self) -> OpenRAState:
        return self._state

    def _build_observation(self, obs_dict: dict, reward: float) -> OpenRAObservation:
        """Convert a raw observation dict to an OpenRAObservation model."""
        return OpenRAObservation(
            tick=obs_dict["tick"],
            economy=EconomyInfo(**obs_dict["economy"]),
            military=MilitaryInfo(**obs_dict["military"]),
            units=[UnitInfoModel(**u) for u in obs_dict["units"]],
            buildings=[BuildingInfoModel(**b) for b in obs_dict["buildings"]],
            production=[ProductionInfoModel(**p) for p in obs_dict["production"]],
            visible_enemies=[UnitInfoModel(**u) for u in obs_dict["visible_enemies"]],
            map_info=MapInfoModel(**obs_dict["map_info"]),
            available_production=obs_dict.get("available_production", []),
            done=obs_dict["done"],
            reward=reward,
            result=obs_dict.get("result", ""),
        )

    def close(self) -> None:
        """Clean up resources."""
        try:
            self._loop.run_until_complete(self._bridge.close())
        except Exception:
            pass
        self._process.kill()
        try:
            self._loop.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
