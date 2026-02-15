"""gRPC bridge client for communicating with the OpenRA ExternalBotBridge.

This client connects to the gRPC server running inside the OpenRA process
and handles bidirectional streaming of observations and actions.

Protocol:
  - Bidirectional streaming RPC (GameSession): game sends observations, agent sends actions
  - Unary RPC (GetState): query current game state on demand
  - Lock-step: each observation waits for one action before the game advances
"""

import asyncio
import base64
import logging
from typing import AsyncIterator, Optional

import grpc

from openra_env.generated import rl_bridge_pb2, rl_bridge_pb2_grpc

logger = logging.getLogger(__name__)


class BridgeClient:
    """Async gRPC client for the OpenRA RL Bridge.

    Uses bidirectional streaming: the game sends observations each tick,
    and the agent sends actions in response (lock-step).
    """

    def __init__(self, host: str = "localhost", port: int = 9999, timeout_s: float = 30.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[rl_bridge_pb2_grpc.RLBridgeStub] = None
        self._session_call = None
        self._action_queue: asyncio.Queue[rl_bridge_pb2.AgentAction] = asyncio.Queue()
        self._connected = False

    async def connect(self) -> None:
        """Establish gRPC channel."""
        target = f"{self.host}:{self.port}"
        self._channel = grpc.aio.insecure_channel(
            target,
            options=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.max_send_message_length", 16 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 10000),
                ("grpc.keepalive_timeout_ms", 5000),
            ],
        )
        self._stub = rl_bridge_pb2_grpc.RLBridgeStub(self._channel)
        self._connected = True
        logger.info(f"Connected to OpenRA bridge at {target}")

    async def wait_for_ready(self, max_retries: int = 30, retry_interval: float = 1.0) -> bool:
        """Wait for the gRPC server to become available."""
        for attempt in range(max_retries):
            try:
                await self.connect()
                state = await self.get_state()
                logger.info(f"Bridge ready after {attempt + 1} attempts, phase={state.phase}")
                return True
            except grpc.aio.AioRpcError as e:
                if attempt < max_retries - 1:
                    logger.debug(f"Bridge not ready (attempt {attempt + 1}): {e.code()}")
                    await asyncio.sleep(retry_interval)
                else:
                    logger.error(f"Bridge failed to become ready after {max_retries} attempts")
                    return False
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.debug(f"Connection attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(retry_interval)
                else:
                    return False
        return False

    async def start_session(self) -> rl_bridge_pb2.GameObservation:
        """Start a bidirectional streaming session and return the first observation.

        The game sends observations, we respond with actions. This call
        starts the stream and returns the initial observation.
        """
        if not self._connected:
            await self.connect()

        self._action_queue = asyncio.Queue()
        self._session_call = self._stub.GameSession(self._action_request_iterator())

        first_obs = await self._session_call.read()
        if first_obs is None:
            raise ConnectionError("Bridge stream closed before sending initial observation")

        logger.info(f"Session started, initial tick={first_obs.tick}")
        return first_obs

    async def _action_request_iterator(self) -> AsyncIterator[rl_bridge_pb2.AgentAction]:
        """Yield actions from the queue as the gRPC stream requests them."""
        while True:
            action = await self._action_queue.get()
            yield action

    async def step(self, action: rl_bridge_pb2.AgentAction) -> rl_bridge_pb2.GameObservation:
        """Send an action and receive the next observation (lock-step).

        This implements the lock-step protocol: the game pauses after each
        observation until we send an action, then runs one or more ticks
        and sends the next observation.
        """
        if self._session_call is None:
            raise RuntimeError("Session not started. Call start_session() first.")

        await self._action_queue.put(action)

        obs = await self._session_call.read()
        if obs is None:
            raise ConnectionError("Bridge stream closed unexpectedly")
        return obs

    async def get_state(self) -> rl_bridge_pb2.GameState:
        """Query current game state via unary RPC."""
        if not self._connected or self._stub is None:
            raise RuntimeError("Not connected. Call connect() first.")
        request = rl_bridge_pb2.StateRequest()
        return await self._stub.GetState(request, timeout=self.timeout_s)

    async def close(self) -> None:
        """Close the gRPC channel and clean up."""
        if self._session_call is not None:
            self._session_call.cancel()
            self._session_call = None

        if self._channel is not None:
            await self._channel.close()
            self._channel = None

        self._stub = None
        self._connected = False
        logger.info("Bridge connection closed")

    @property
    def is_connected(self) -> bool:
        return self._connected


def observation_to_dict(obs: rl_bridge_pb2.GameObservation) -> dict:
    """Convert a protobuf GameObservation to a plain dict for the OpenEnv layer."""
    return {
        "tick": obs.tick,
        "economy": {
            "cash": obs.economy.cash,
            "ore": obs.economy.ore,
            "power_provided": obs.economy.power_provided,
            "power_drained": obs.economy.power_drained,
            "resource_capacity": obs.economy.resource_capacity,
            "harvester_count": obs.economy.harvester_count,
        },
        "military": {
            "units_killed": obs.military.units_killed,
            "units_lost": obs.military.units_lost,
            "buildings_killed": obs.military.buildings_killed,
            "buildings_lost": obs.military.buildings_lost,
            "army_value": obs.military.army_value,
            "active_unit_count": obs.military.active_unit_count,
        },
        "units": [
            {
                "actor_id": u.actor_id,
                "type": u.type,
                "pos_x": u.pos_x,
                "pos_y": u.pos_y,
                "cell_x": u.cell_x,
                "cell_y": u.cell_y,
                "hp_percent": u.hp_percent,
                "is_idle": u.is_idle,
                "current_activity": u.current_activity,
                "owner": u.owner,
                "can_attack": u.can_attack,
                "facing": u.facing,
                "experience_level": u.experience_level,
                "stance": u.stance,
                "speed": u.speed,
                "attack_range": u.attack_range,
                "passenger_count": u.passenger_count,
                "is_building": u.is_building,
            }
            for u in obs.units
        ],
        "buildings": [
            {
                "actor_id": b.actor_id,
                "type": b.type,
                "pos_x": b.pos_x,
                "pos_y": b.pos_y,
                "hp_percent": b.hp_percent,
                "owner": b.owner,
                "is_producing": b.is_producing,
                "production_progress": b.production_progress,
                "producing_item": b.producing_item,
                "is_powered": b.is_powered,
                "is_repairing": b.is_repairing,
                "sell_value": b.sell_value,
                "rally_x": b.rally_x,
                "rally_y": b.rally_y,
                "power_amount": b.power_amount,
                "can_produce": list(b.can_produce),
                "cell_x": b.cell_x,
                "cell_y": b.cell_y,
            }
            for b in obs.buildings
        ],
        "production": [
            {
                "queue_type": p.queue_type,
                "item": p.item,
                "progress": p.progress,
                "remaining_ticks": p.remaining_ticks,
                "remaining_cost": p.remaining_cost,
                "paused": p.paused,
            }
            for p in obs.production
        ],
        "visible_enemies": [
            {
                "actor_id": u.actor_id,
                "type": u.type,
                "pos_x": u.pos_x,
                "pos_y": u.pos_y,
                "cell_x": u.cell_x,
                "cell_y": u.cell_y,
                "hp_percent": u.hp_percent,
                "is_idle": u.is_idle,
                "current_activity": u.current_activity,
                "owner": u.owner,
                "can_attack": u.can_attack,
                "facing": u.facing,
                "experience_level": u.experience_level,
                "stance": u.stance,
                "speed": u.speed,
                "attack_range": u.attack_range,
                "passenger_count": u.passenger_count,
                "is_building": u.is_building,
            }
            for u in obs.visible_enemies
        ],
        "visible_enemy_buildings": [
            {
                "actor_id": b.actor_id,
                "type": b.type,
                "pos_x": b.pos_x,
                "pos_y": b.pos_y,
                "hp_percent": b.hp_percent,
                "owner": b.owner,
                "is_producing": b.is_producing,
                "production_progress": b.production_progress,
                "producing_item": b.producing_item,
                "is_powered": b.is_powered,
                "is_repairing": b.is_repairing,
                "sell_value": b.sell_value,
                "rally_x": b.rally_x,
                "rally_y": b.rally_y,
                "power_amount": b.power_amount,
                "can_produce": list(b.can_produce),
                "cell_x": b.cell_x,
                "cell_y": b.cell_y,
            }
            for b in obs.visible_enemy_buildings
        ],
        "map_info": {
            "width": obs.map_info.width,
            "height": obs.map_info.height,
            "map_name": obs.map_info.map_name,
        },
        "available_production": list(obs.available_production),
        "done": obs.done,
        "reward": obs.reward,
        "result": obs.result,
        "spatial_map": base64.b64encode(bytes(obs.spatial_map)).decode("ascii"),
        "spatial_channels": obs.spatial_channels,
    }


def commands_to_proto(commands: list[dict]) -> rl_bridge_pb2.AgentAction:
    """Convert a list of command dicts to a protobuf AgentAction."""
    action_type_map = {
        "no_op": rl_bridge_pb2.NO_OP,
        "move": rl_bridge_pb2.MOVE,
        "attack_move": rl_bridge_pb2.ATTACK_MOVE,
        "attack": rl_bridge_pb2.ATTACK,
        "stop": rl_bridge_pb2.STOP,
        "harvest": rl_bridge_pb2.HARVEST,
        "build": rl_bridge_pb2.BUILD,
        "train": rl_bridge_pb2.TRAIN,
        "deploy": rl_bridge_pb2.DEPLOY,
        "sell": rl_bridge_pb2.SELL,
        "repair": rl_bridge_pb2.REPAIR,
        "place_building": rl_bridge_pb2.PLACE_BUILDING,
        "cancel_production": rl_bridge_pb2.CANCEL_PRODUCTION,
        "set_rally_point": rl_bridge_pb2.SET_RALLY_POINT,
        "guard": rl_bridge_pb2.GUARD,
        "set_stance": rl_bridge_pb2.SET_STANCE,
        "enter_transport": rl_bridge_pb2.ENTER_TRANSPORT,
        "unload": rl_bridge_pb2.UNLOAD,
        "power_down": rl_bridge_pb2.POWER_DOWN,
        "set_primary": rl_bridge_pb2.SET_PRIMARY,
    }

    proto_commands = []
    for cmd in commands:
        action_str = cmd.get("action", "no_op")
        proto_cmd = rl_bridge_pb2.Command(
            action=action_type_map.get(action_str, rl_bridge_pb2.NO_OP),
            actor_id=cmd.get("actor_id", 0),
            target_actor_id=cmd.get("target_actor_id", 0),
            target_x=cmd.get("target_x", 0),
            target_y=cmd.get("target_y", 0),
            item_type=cmd.get("item_type", ""),
            queued=cmd.get("queued", False),
        )
        proto_commands.append(proto_cmd)

    return rl_bridge_pb2.AgentAction(commands=proto_commands)
