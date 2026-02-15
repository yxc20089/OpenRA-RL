"""OpenRA Environment server implementing the OpenEnv MCPEnvironment interface.

This is the core environment that manages OpenRA game instances,
translates between the OpenEnv API and the gRPC bridge protocol,
computes rewards, and exposes MCP tools for LLM agents.
"""

import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

from openra_env.game_data import (
    get_all_building_types,
    get_all_unit_types,
    get_building_stats,
    get_faction_info,
    get_tech_tree,
    get_unit_stats,
)
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
from openra_env.reward import OpenRARewardFunction, RewardWeights
from openra_env.server.bridge_client import BridgeClient, commands_to_proto, observation_to_dict
from openra_env.server.openra_process import OpenRAConfig, OpenRAProcessManager

logger = logging.getLogger(__name__)


class OpenRAEnvironment(MCPEnvironment):
    """OpenRA RL Environment with MCP tool support.

    Manages OpenRA game instances and provides both:
    - Gymnasium-style API (reset/step/state) for RL training
    - MCP tools for LLM agent interaction (via ListToolsAction/CallToolAction)

    Each reset() launches a new OpenRA subprocess with the ExternalBotBridge
    trait enabled, connects via gRPC, and returns the initial observation.
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
        record_replays: bool = False,
    ):
        # Create MCP server and register tools
        mcp = FastMCP("openra")
        self._register_tools(mcp)
        super().__init__(mcp)

        self._config = OpenRAConfig(
            openra_path=openra_path or OpenRAConfig.openra_path,
            mod=mod,
            map_name=map_name,
            grpc_port=grpc_port,
            bot_type=bot_type,
            record_replays=record_replays,
        )
        self._process = OpenRAProcessManager(self._config)
        self._bridge = BridgeClient(port=grpc_port)
        self._reward_fn = OpenRARewardFunction(weights=reward_weights)
        self._state = OpenRAState()
        self._last_obs: Optional[dict] = None
        # Persistent event loop for async gRPC bridge operations.
        self._loop = asyncio.new_event_loop()

    def _register_tools(self, mcp: FastMCP) -> None:
        """Register all MCP tools for LLM agent interaction."""
        env = self

        # ── Read Tools (return from cached observation) ──────────────────

        @mcp.tool()
        def get_game_state() -> dict:
            """Get a full summary of the current game state including economy,
            military stats, unit counts, building counts, and enemy visibility."""
            obs = env._last_obs
            if obs is None:
                return {"error": "No observation available. Call advance() or reset first."}
            return {
                "tick": obs["tick"],
                "done": obs["done"],
                "result": obs.get("result", ""),
                "economy": obs["economy"],
                "military": obs["military"],
                "own_units": len(obs["units"]),
                "own_buildings": len(obs["buildings"]),
                "visible_enemy_units": len(obs["visible_enemies"]),
                "visible_enemy_buildings": len(obs.get("visible_enemy_buildings", [])),
                "production_queues": len(obs["production"]),
                "available_production": obs.get("available_production", []),
                "map": obs["map_info"],
            }

        @mcp.tool()
        def get_economy() -> dict:
            """Get current economic state: cash, ore, power, harvesters."""
            obs = env._last_obs
            if obs is None:
                return {"error": "No observation available."}
            return obs["economy"]

        @mcp.tool()
        def get_units() -> list[dict]:
            """Get list of own units with id, type, position, hp, activity, stance."""
            obs = env._last_obs
            if obs is None:
                return []
            return [
                {
                    "actor_id": u["actor_id"],
                    "type": u["type"],
                    "cell_x": u["cell_x"],
                    "cell_y": u["cell_y"],
                    "hp_percent": round(u["hp_percent"], 2),
                    "is_idle": u["is_idle"],
                    "current_activity": u["current_activity"],
                    "can_attack": u["can_attack"],
                    "stance": u["stance"],
                    "attack_range": u["attack_range"],
                }
                for u in obs["units"]
            ]

        @mcp.tool()
        def get_buildings() -> list[dict]:
            """Get list of own buildings with id, type, position, hp, production status, power."""
            obs = env._last_obs
            if obs is None:
                return []
            return [
                {
                    "actor_id": b["actor_id"],
                    "type": b["type"],
                    "cell_x": b["cell_x"],
                    "cell_y": b["cell_y"],
                    "hp_percent": round(b["hp_percent"], 2),
                    "is_producing": b["is_producing"],
                    "producing_item": b["producing_item"],
                    "production_progress": round(b["production_progress"], 2),
                    "is_powered": b["is_powered"],
                    "is_repairing": b["is_repairing"],
                    "power_amount": b["power_amount"],
                    "rally_x": b["rally_x"],
                    "rally_y": b["rally_y"],
                    "can_produce": b["can_produce"],
                }
                for b in obs["buildings"]
            ]

        @mcp.tool()
        def get_enemies() -> dict:
            """Get visible enemy units and buildings."""
            obs = env._last_obs
            if obs is None:
                return {"units": [], "buildings": []}
            return {
                "units": [
                    {
                        "actor_id": u["actor_id"],
                        "type": u["type"],
                        "cell_x": u["cell_x"],
                        "cell_y": u["cell_y"],
                        "hp_percent": round(u["hp_percent"], 2),
                        "owner": u["owner"],
                        "can_attack": u["can_attack"],
                    }
                    for u in obs["visible_enemies"]
                ],
                "buildings": [
                    {
                        "actor_id": b["actor_id"],
                        "type": b["type"],
                        "cell_x": b["cell_x"],
                        "cell_y": b["cell_y"],
                        "hp_percent": round(b["hp_percent"], 2),
                        "owner": b["owner"],
                    }
                    for b in obs.get("visible_enemy_buildings", [])
                ],
            }

        @mcp.tool()
        def get_production() -> dict:
            """Get production queue items and available buildable types."""
            obs = env._last_obs
            if obs is None:
                return {"queue": [], "available": []}
            return {
                "queue": [
                    {
                        "queue_type": p["queue_type"],
                        "item": p["item"],
                        "progress": round(p["progress"], 2),
                        "remaining_ticks": p["remaining_ticks"],
                        "paused": p["paused"],
                    }
                    for p in obs["production"]
                ],
                "available": obs.get("available_production", []),
            }

        @mcp.tool()
        def get_map_info() -> dict:
            """Get map dimensions and name."""
            obs = env._last_obs
            if obs is None:
                return {"error": "No observation available."}
            return obs["map_info"]

        # ── Game Knowledge Tools (static mod data) ───────────────────────

        @mcp.tool()
        def lookup_unit(unit_type: str) -> dict:
            """Look up stats for a unit type (e.g., 'e1', '3tnk', 'mig').
            Returns cost, HP, speed, armor, prerequisites, and description."""
            result = get_unit_stats(unit_type)
            if result is None:
                all_types = get_all_unit_types()
                return {"error": f"Unknown unit type '{unit_type}'", "available_types": all_types}
            return result

        @mcp.tool()
        def lookup_building(building_type: str) -> dict:
            """Look up stats for a building type (e.g., 'powr', 'weap', 'stek').
            Returns cost, HP, power, prerequisites, and description."""
            result = get_building_stats(building_type)
            if result is None:
                all_types = get_all_building_types()
                return {"error": f"Unknown building type '{building_type}'", "available_types": all_types}
            return result

        @mcp.tool()
        def lookup_tech_tree(faction: str = "soviet") -> dict:
            """Get the tech tree / build order for a faction or side.
            Accepts faction names ('russia', 'england') or sides ('allied', 'soviet')."""
            return get_tech_tree(faction)

        @mcp.tool()
        def lookup_faction(faction: str) -> dict:
            """Get faction info including all available units and buildings.
            Faction names: 'england', 'france', 'germany', 'russia', 'ukraine'."""
            result = get_faction_info(faction)
            if result is None:
                return {"error": f"Unknown faction '{faction}'", "factions": ["england", "france", "germany", "russia", "ukraine"]}
            return result

        # ── Action Tools (advance game state) ────────────────────────────

        @mcp.tool()
        def advance(ticks: int = 1) -> dict:
            """Advance the game by sending a no-op action. Use to let the game
            progress without issuing commands. Returns updated game summary."""
            commands = [CommandModel(action=ActionType.NO_OP)]
            return env._execute_commands(commands)

        @mcp.tool()
        def move_units(unit_ids: list[int], target_x: int, target_y: int, queued: bool = False) -> dict:
            """Move units to a map cell position. Units pathfind automatically."""
            commands = [
                CommandModel(action=ActionType.MOVE, actor_id=uid, target_x=target_x, target_y=target_y, queued=queued)
                for uid in unit_ids
            ]
            return env._execute_commands(commands)

        @mcp.tool()
        def attack_move(unit_ids: list[int], target_x: int, target_y: int, queued: bool = False) -> dict:
            """Move units toward a cell, attacking enemies encountered along the way."""
            commands = [
                CommandModel(action=ActionType.ATTACK_MOVE, actor_id=uid, target_x=target_x, target_y=target_y, queued=queued)
                for uid in unit_ids
            ]
            return env._execute_commands(commands)

        @mcp.tool()
        def attack_target(unit_ids: list[int], target_actor_id: int, queued: bool = False) -> dict:
            """Order units to attack a specific enemy actor by ID."""
            commands = [
                CommandModel(action=ActionType.ATTACK, actor_id=uid, target_actor_id=target_actor_id, queued=queued)
                for uid in unit_ids
            ]
            return env._execute_commands(commands)

        @mcp.tool()
        def stop_units(unit_ids: list[int]) -> dict:
            """Stop units from their current activity."""
            commands = [CommandModel(action=ActionType.STOP, actor_id=uid) for uid in unit_ids]
            return env._execute_commands(commands)

        @mcp.tool()
        def build_unit(unit_type: str) -> dict:
            """Start training a unit (infantry, vehicle, aircraft, ship).
            The unit_type is the internal name (e.g., 'e1', '3tnk', 'mig')."""
            commands = [CommandModel(action=ActionType.TRAIN, item_type=unit_type)]
            return env._execute_commands(commands)

        @mcp.tool()
        def build_structure(building_type: str) -> dict:
            """Start constructing a building. Building will need to be placed
            when ready using place_building(). building_type is the internal
            name (e.g., 'powr', 'barr', 'weap')."""
            commands = [CommandModel(action=ActionType.BUILD, item_type=building_type)]
            return env._execute_commands(commands)

        @mcp.tool()
        def place_building(building_type: str, cell_x: int, cell_y: int) -> dict:
            """Place a completed building at the specified cell position.
            Call after build_structure() completes (check production progress)."""
            commands = [CommandModel(action=ActionType.PLACE_BUILDING, item_type=building_type, target_x=cell_x, target_y=cell_y)]
            return env._execute_commands(commands)

        @mcp.tool()
        def cancel_production(item_type: str) -> dict:
            """Cancel production of an item currently in a production queue."""
            commands = [CommandModel(action=ActionType.CANCEL_PRODUCTION, item_type=item_type)]
            return env._execute_commands(commands)

        @mcp.tool()
        def deploy_unit(unit_id: int) -> dict:
            """Deploy a unit (e.g., MCV → Construction Yard)."""
            commands = [CommandModel(action=ActionType.DEPLOY, actor_id=unit_id)]
            return env._execute_commands(commands)

        @mcp.tool()
        def sell_building(building_id: int) -> dict:
            """Sell a building for partial refund."""
            commands = [CommandModel(action=ActionType.SELL, actor_id=building_id)]
            return env._execute_commands(commands)

        @mcp.tool()
        def repair_building(building_id: int) -> dict:
            """Toggle repair on a building. Costs credits over time."""
            commands = [CommandModel(action=ActionType.REPAIR, actor_id=building_id)]
            return env._execute_commands(commands)

        @mcp.tool()
        def set_rally_point(building_id: int, cell_x: int, cell_y: int) -> dict:
            """Set rally point for a production building. Newly produced units
            will move to this location."""
            commands = [CommandModel(action=ActionType.SET_RALLY_POINT, actor_id=building_id, target_x=cell_x, target_y=cell_y)]
            return env._execute_commands(commands)

        @mcp.tool()
        def guard_target(unit_ids: list[int], target_actor_id: int, queued: bool = False) -> dict:
            """Order units to guard another actor, following and protecting it."""
            commands = [
                CommandModel(action=ActionType.GUARD, actor_id=uid, target_actor_id=target_actor_id, queued=queued)
                for uid in unit_ids
            ]
            return env._execute_commands(commands)

        @mcp.tool()
        def set_stance(unit_ids: list[int], stance: str) -> dict:
            """Set combat stance for units.
            Stances: 'hold_fire' (0), 'return_fire' (1), 'defend' (2), 'attack_anything' (3)."""
            stance_map = {"hold_fire": 0, "return_fire": 1, "defend": 2, "attack_anything": 3}
            stance_val = stance_map.get(stance.lower(), 3)
            commands = [
                CommandModel(action=ActionType.SET_STANCE, actor_id=uid, target_x=stance_val)
                for uid in unit_ids
            ]
            return env._execute_commands(commands)

        @mcp.tool()
        def harvest(unit_id: int, cell_x: int = 0, cell_y: int = 0) -> dict:
            """Send a harvester to collect ore. If cell_x/cell_y are provided,
            harvest at that location. Otherwise, auto-harvest nearest ore."""
            commands = [CommandModel(action=ActionType.HARVEST, actor_id=unit_id, target_x=cell_x, target_y=cell_y)]
            return env._execute_commands(commands)

        @mcp.tool()
        def power_down(building_id: int) -> dict:
            """Toggle power on/off for a building. Reduces power consumption
            but disables the building's function."""
            commands = [CommandModel(action=ActionType.POWER_DOWN, actor_id=building_id)]
            return env._execute_commands(commands)

        @mcp.tool()
        def set_primary(building_id: int) -> dict:
            """Set a production building as the primary producer. New units will
            exit from this building."""
            commands = [CommandModel(action=ActionType.SET_PRIMARY, actor_id=building_id)]
            return env._execute_commands(commands)

        @mcp.tool()
        def get_replay_path() -> dict:
            """Get the path to the most recent replay file from this session."""
            replay_dir = env._get_replay_dir()
            if not replay_dir.exists():
                return {"error": "No replay directory found"}
            replays = sorted(replay_dir.rglob("*.orarep"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not replays:
                return {"error": "No replay files found"}
            return {"path": str(replays[0]), "size_bytes": replays[0].stat().st_size}

    # ── Internal helpers ─────────────────────────────────────────────────

    def _execute_commands(self, commands: list[CommandModel]) -> dict:
        """Send commands, step the game, update cache, return summary."""
        action = OpenRAAction(commands=commands)
        obs_dict = self._loop.run_until_complete(self._async_step_internal(action))
        self._last_obs = obs_dict
        return {
            "tick": obs_dict["tick"],
            "done": obs_dict["done"],
            "result": obs_dict.get("result", ""),
            "economy": obs_dict["economy"],
            "own_units": len(obs_dict["units"]),
            "own_buildings": len(obs_dict["buildings"]),
            "visible_enemies": len(obs_dict["visible_enemies"]),
        }

    async def _async_step_internal(self, action: OpenRAAction) -> dict:
        """Core step logic: send action via gRPC, receive observation dict."""
        self._state.step_count += 1

        cmd_dicts = [cmd.model_dump() for cmd in action.commands]
        proto_action = commands_to_proto(cmd_dicts)

        proto_obs = await self._bridge.step(proto_action)
        obs_dict = observation_to_dict(proto_obs)

        self._state.game_tick = obs_dict["tick"]
        return obs_dict

    def _get_replay_dir(self) -> Path:
        """Get the OpenRA replays directory for current mod."""
        if sys.platform == "darwin":
            base = Path.home() / "Library/Application Support/OpenRA"
        else:
            base = Path.home() / ".openra"
        return base / "Replays" / self._config.mod

    # ── OpenEnv Interface ────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OpenRAObservation:
        """Reset the environment for a new episode."""
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
        self._last_obs = None

        # Update seed if provided
        if seed is not None:
            self._config.seed = seed

        # Launch OpenRA
        logger.info(f"Launching OpenRA: map={self._config.map_name}, mod={self._config.mod}")
        self._process.launch()
        logger.info(f"OpenRA process launched (PID={self._process.pid})")

        # Wait for gRPC server to be ready
        logger.info("Waiting for gRPC bridge to become ready...")
        ready = await self._bridge.wait_for_ready(max_retries=120, retry_interval=2.0)
        if not ready:
            alive = self._process.is_alive()
            logger.error(f"Bridge failed to start. Process alive={alive}")
            raise RuntimeError("OpenRA gRPC bridge failed to start")

        # Start streaming session and get initial observation
        proto_obs = await self._bridge.start_session()
        obs_dict = observation_to_dict(proto_obs)
        self._last_obs = obs_dict

        # Compute initial reward (should be 0)
        reward = self._reward_fn.compute(obs_dict)

        return self._build_observation(obs_dict, reward)

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions (OpenRAAction for backward compat)."""
        if isinstance(action, OpenRAAction):
            obs_dict = self._loop.run_until_complete(self._async_step_internal(action))
            self._last_obs = obs_dict
            reward = self._reward_fn.compute(obs_dict)
            return self._build_observation(obs_dict, reward)

        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": f"Unknown action type: {type(action).__name__}. Use MCP tools or OpenRAAction."},
        )

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
            visible_enemy_buildings=[BuildingInfoModel(**b) for b in obs_dict.get("visible_enemy_buildings", [])],
            map_info=MapInfoModel(**obs_dict["map_info"]),
            available_production=obs_dict.get("available_production", []),
            done=obs_dict["done"],
            reward=reward,
            result=obs_dict.get("result", ""),
            spatial_map=obs_dict.get("spatial_map", ""),
            spatial_channels=obs_dict.get("spatial_channels", 0),
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
