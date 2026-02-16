"""OpenRA Environment server implementing the OpenEnv MCPEnvironment interface.

This is the core environment that manages OpenRA game instances,
translates between the OpenEnv API and the gRPC bridge protocol,
computes rewards, and exposes MCP tools for LLM agents.
"""

import asyncio
import logging
import os
import sys
import threading
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
        ai_slot: str = "",
        reward_weights: Optional[RewardWeights] = None,
        record_replays: bool = False,
    ):
        # Create MCP server and register tools
        mcp = FastMCP("openra")
        self._register_tools(mcp)
        super().__init__(mcp)

        # Allow environment variables to override defaults
        bot_type = os.environ.get("BOT_TYPE", bot_type)
        ai_slot = os.environ.get("AI_SLOT", ai_slot)
        if os.environ.get("RECORD_REPLAYS", "").lower() in ("true", "1", "yes"):
            record_replays = True

        self._config = OpenRAConfig(
            openra_path=openra_path or OpenRAConfig.openra_path,
            mod=mod,
            map_name=map_name,
            grpc_port=grpc_port,
            bot_type=bot_type,
            ai_slot=ai_slot,
            record_replays=record_replays,
        )
        self._process = OpenRAProcessManager(self._config)
        self._bridge = BridgeClient(port=grpc_port)
        self._reward_fn = OpenRARewardFunction(weights=reward_weights)
        self._state = OpenRAState()
        self._last_obs: Optional[dict] = None
        self._unit_groups: dict[str, list[int]] = {}  # named groups of unit IDs
        self._pending_placements: dict[str, dict] = {}  # building_type → {cell_x, cell_y}
        self._attempted_placements: dict[str, int] = {}  # building_type → attempt_count (for failure detection)
        self._placement_results: list[str] = []  # alerts from auto-placement attempts
        self._player_faction: str = ""
        self._enemy_faction: str = ""
        # Persistent event loop for async gRPC bridge operations.
        # Runs in a background thread so it doesn't conflict with
        # FastAPI/uvicorn's event loop when MCP tools call it.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="openra-bridge-loop"
        )
        self._loop_thread.start()

    def _register_tools(self, mcp: FastMCP) -> None:
        """Register all MCP tools for LLM agent interaction."""
        env = self

        # ── Read Tools (return from cached observation) ──────────────────

        @mcp.tool()
        def get_game_state() -> dict:
            """Get a full summary of the current game state including economy,
            military stats, unit counts, building counts, enemy visibility, and alerts."""
            env._refresh_obs()
            obs = env._last_obs
            if obs is None:
                return {"error": "No observation available. Call advance() or reset first."}

            eco = obs["economy"]
            power_balance = eco["power_provided"] - eco["power_drained"]

            # Compute alerts
            alerts = []

            # Under attack: enemy units near our buildings
            for enemy in obs["visible_enemies"]:
                for bldg in obs["buildings"]:
                    dx = abs(enemy.get("cell_x", 0) - bldg.get("cell_x", 0))
                    dy = abs(enemy.get("cell_y", 0) - bldg.get("cell_y", 0))
                    if dx + dy < 12:
                        alerts.append(
                            f"UNDER ATTACK: enemy {enemy['type']} id={enemy['actor_id']} "
                            f"near your {bldg['type']} at ({bldg['cell_x']},{bldg['cell_y']})"
                        )
                        break

            # Damaged buildings
            for bldg in obs["buildings"]:
                if bldg["hp_percent"] < 0.5:
                    alerts.append(f"DAMAGED: {bldg['type']} at {bldg['hp_percent']:.0%} HP — repair_building({bldg['actor_id']})")

            # Power crisis
            if power_balance < 0:
                alerts.append(f"LOW POWER: {power_balance:+d} — build powr immediately")

            # Idle funds with few harvesters
            total_funds = eco["cash"] + eco.get("ore", 0)
            if total_funds > 2000 and eco["harvester_count"] < 2:
                alerts.append(f"IDLE FUNDS: ${total_funds} with {eco['harvester_count']} harvester(s) — build refinery or harvester")

            # Nothing being produced
            if not obs["production"] and len(obs["buildings"]) >= 3:
                alerts.append("IDLE PRODUCTION: nothing being built or trained — queue something")

            # Building ready to place (skip if auto-placement is pending)
            pending = getattr(env, "_pending_placements", {})
            for p in obs["production"]:
                if p["queue_type"] == "Building" and p["progress"] >= 0.99:
                    if p["item"] not in pending:
                        alerts.append(f"READY TO PLACE: {p['item']} — call place_building()")

            # Auto-placement results from build_and_place
            placement_results = getattr(env, "_placement_results", [])
            if placement_results:
                alerts.extend(placement_results)
                placement_results.clear()

            # Combat units on default ReturnFire stance
            returnfire_count = sum(
                1 for u in obs["units"]
                if u.get("can_attack") and u.get("stance", 1) == 1
            )
            if returnfire_count > 0:
                alerts.append(f"STANCE WARNING: {returnfire_count} combat unit(s) on ReturnFire — set to attack_anything")

            # Compact summaries with actor IDs for planning
            units_summary = [
                {"id": u["actor_id"], "type": u["type"], "idle": u["is_idle"],
                 "can_attack": u["can_attack"], "stance": u["stance"],
                 "cell_x": u["cell_x"], "cell_y": u["cell_y"]}
                for u in obs["units"]
            ]
            buildings_summary = [
                {"id": b["actor_id"], "type": b["type"],
                 "cell_x": b["cell_x"], "cell_y": b["cell_y"]}
                for b in obs["buildings"]
            ]
            enemy_summary = [
                {"id": e["actor_id"], "type": e["type"],
                 "cell_x": e["cell_x"], "cell_y": e["cell_y"]}
                for e in obs["visible_enemies"]
            ]

            return {
                "tick": obs["tick"],
                "done": obs["done"],
                "result": obs.get("result", ""),
                "faction": getattr(env, "_player_faction", ""),
                "economy": obs["economy"],
                "power_balance": power_balance,
                "military": obs["military"],
                "own_units": len(obs["units"]),
                "own_buildings": len(obs["buildings"]),
                "building_types": list(set(b["type"] for b in obs["buildings"])),
                "visible_enemy_units": len(obs["visible_enemies"]),
                "visible_enemy_buildings": len(obs.get("visible_enemy_buildings", [])),
                "production_queues": len(obs["production"]),
                "production_items": [f"{p['item']}@{p['progress']:.0%}" for p in obs["production"]],
                "available_production": obs.get("available_production", []),
                "units_summary": units_summary,
                "buildings_summary": buildings_summary,
                "enemy_summary": enemy_summary,
                "alerts": alerts,
                "map": obs["map_info"],
            }

        @mcp.tool()
        def get_economy() -> dict:
            """Get current economic state: cash, ore, power, harvesters."""
            env._refresh_obs()
            obs = env._last_obs
            if obs is None:
                return {"error": "No observation available."}
            return obs["economy"]

        @mcp.tool()
        def get_units() -> list[dict]:
            """Get list of own units with id, type, position, hp, activity, stance."""
            env._refresh_obs()
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
            env._refresh_obs()
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
            env._refresh_obs()
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
            env._refresh_obs()
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
            env._refresh_obs()
            obs = env._last_obs
            if obs is None:
                return {"error": "No observation available."}
            return obs["map_info"]

        @mcp.tool()
        def get_terrain_at(cell_x: int, cell_y: int) -> dict:
            """Check terrain at a map cell. Returns passability and whether it's
            water. Useful before placing buildings (spen/syrd need water)."""
            env._refresh_obs()
            obs = env._last_obs
            if obs is None:
                return {"error": "No observation available"}

            spatial = obs.get("spatial_map", "")
            map_info = obs.get("map_info", {})
            w = map_info.get("width", 0)
            h = map_info.get("height", 0)
            channels = obs.get("spatial_channels", 0)

            if not spatial or w == 0 or channels == 0:
                return {"error": "No spatial map data available"}
            if cell_x < 0 or cell_x >= w or cell_y < 0 or cell_y >= h:
                return {"error": f"Out of bounds: ({cell_x},{cell_y}), map is {w}x{h}"}

            import base64
            import struct
            try:
                raw = base64.b64decode(spatial)
                # Row-major channels-last: index = (y * w + x) * channels + ch
                base_idx = (cell_y * w + cell_x) * channels
                terrain_idx = struct.unpack_from("f", raw, base_idx * 4)[0]
                passable = struct.unpack_from("f", raw, (base_idx + 3) * 4)[0]
                return {
                    "cell_x": cell_x,
                    "cell_y": cell_y,
                    "terrain_index": int(terrain_idx),
                    "passable": passable > 0.5,
                    "note": "Water cells are impassable to land units. spen/syrd require water.",
                }
            except Exception as e:
                return {"error": f"Failed to decode terrain: {e}"}

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
            """Wait for the game to advance by the specified number of ticks.
            The game runs at normal speed (~25 ticks/sec). Use this to let
            time pass without issuing commands (e.g., while waiting for
            production to complete). Returns updated game summary."""
            ticks = max(1, min(ticks, 500))  # clamp to [1, 500]
            try:
                future = asyncio.run_coroutine_threadsafe(
                    env._bridge.wait_ticks(ticks), env._loop
                )
                proto_obs = future.result(timeout=300)
                obs_dict = observation_to_dict(proto_obs)
                env._last_obs = obs_dict
            except Exception:
                # Connection lost — check if game ended while waiting
                env._refresh_obs()
                obs_dict = env._last_obs
                if obs_dict is None or not obs_dict.get("done"):
                    raise

            env._state.game_tick = obs_dict["tick"]
            return {
                "tick": obs_dict["tick"],
                "done": obs_dict["done"],
                "result": obs_dict.get("result", ""),
                "economy": obs_dict["economy"],
                "own_units": len(obs_dict["units"]),
                "own_buildings": len(obs_dict["buildings"]),
                "visible_enemies": len(obs_dict["visible_enemies"]),
            }

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
        def build_unit(unit_type: str, count: int = 1) -> dict:
            """Start training units (infantry, vehicle, aircraft, ship).
            The unit_type is the internal name (e.g., 'e1', '3tnk', 'mig').
            Use count > 1 to queue multiple of the same type."""
            count = max(1, min(count, 10))
            commands = [CommandModel(action=ActionType.TRAIN, item_type=unit_type)
                        for _ in range(count)]
            return env._execute_commands(commands)

        @mcp.tool()
        def build_structure(building_type: str) -> dict:
            """Start constructing a building. Building will need to be placed
            when ready using place_building(). building_type is the internal
            name (e.g., 'powr', 'barr', 'weap').
            Prefer build_and_place() which auto-places when done."""
            commands = [CommandModel(action=ActionType.BUILD, item_type=building_type)]
            return env._execute_commands(commands)

        @mcp.tool()
        def build_and_place(building_type: str, cell_x: int = 0, cell_y: int = 0) -> dict:
            """Build a structure and auto-place it when construction finishes.
            Coordinates are optional — the engine auto-finds a valid position
            near your base if omitted or invalid.
            This is the preferred way to build — no need to call place_building separately."""
            commands = [CommandModel(action=ActionType.BUILD, item_type=building_type)]
            result = env._execute_commands(commands)
            if "error" not in result:
                env._pending_placements[building_type] = {"cell_x": cell_x, "cell_y": cell_y}
            return result

        @mcp.tool()
        def place_building(building_type: str, cell_x: int = 0, cell_y: int = 0) -> dict:
            """Place a completed building at the specified cell position.
            Cell coordinates are optional — the game engine auto-finds the best
            valid position near your base if the given position is invalid or omitted."""
            # Check building is ready in production queue
            env._refresh_obs()
            pre_obs = env._last_obs
            if pre_obs:
                ready = any(
                    p["queue_type"] == "Building" and p["item"] == building_type and p["progress"] >= 0.99
                    for p in pre_obs["production"]
                )
                if not ready:
                    return {
                        "error": f"'{building_type}' not ready to place. Build with build_structure() first and advance() until done.",
                        "tick": pre_obs.get("tick", 0),
                    }

            env._pending_placements.pop(building_type, None)
            commands = [CommandModel(action=ActionType.PLACE_BUILDING,
                                    item_type=building_type, target_x=cell_x, target_y=cell_y)]
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

        # ── Placement Helper ────────────────────────────────────────────

        # Building footprint sizes (width x height in cells) from RA rules
        _FOOTPRINTS = {
            "fact": (3, 4), "proc": (3, 4),
            "powr": (2, 3), "apwr": (3, 3),
            "barr": (2, 3), "tent": (2, 3),
            "weap": (3, 3), "fix": (3, 3), "stek": (3, 3),
            "dome": (2, 3), "hpad": (2, 3), "afld": (3, 2),
            "iron": (2, 2), "pdox": (2, 2),
            "mslo": (2, 1), "sam": (2, 1),
            "spen": (3, 3), "syrd": (3, 3), "atek": (2, 3),
        }

        @mcp.tool()
        def get_valid_placements(building_type: str, max_results: int = 8) -> dict:
            """Get suggested placement positions for a building near your base.
            Returns positions sorted by distance from Construction Yard.
            Use the first position with place_building(). If it fails, try the next."""
            env._refresh_obs()
            obs = env._last_obs
            if obs is None:
                return {"error": "No observation available"}

            # Find Construction Yard
            cy = None
            for b in obs["buildings"]:
                if b["type"] == "fact":
                    cy = b
                    break
            if cy is None:
                return {"error": "No Construction Yard found — deploy MCV first"}

            cx, cy_y = cy["cell_x"], cy["cell_y"]
            bw, bh = _FOOTPRINTS.get(building_type, (2, 2))

            # Mark occupied cells from all existing buildings (with 1-cell padding)
            occupied = set()
            for b in obs["buildings"]:
                fw, fh = _FOOTPRINTS.get(b["type"], (2, 2))
                bx, by = b["cell_x"], b["cell_y"]
                for dx in range(-1, fw + 1):
                    for dy in range(-1, fh + 1):
                        occupied.add((bx + dx, by + dy))

            # Map bounds
            map_info = obs.get("map_info", {})
            map_w = map_info.get("width", 128)
            map_h = map_info.get("height", 128)

            # Generate candidates sorted by distance from CY
            candidates = []
            max_radius = 12  # RA build radius from CY
            for dx in range(-max_radius, max_radius + 1):
                for dy in range(-max_radius, max_radius + 1):
                    px, py = cx + dx, cy_y + dy
                    dist = abs(dx) + abs(dy)
                    if dist < 2 or dist > max_radius:
                        continue

                    # Check map bounds
                    if px < 0 or py < 0 or px + bw > map_w or py + bh > map_h:
                        continue

                    # Check no overlap with occupied cells
                    overlap = False
                    for ox in range(bw):
                        for oy in range(bh):
                            if (px + ox, py + oy) in occupied:
                                overlap = True
                                break
                        if overlap:
                            break

                    if not overlap:
                        candidates.append({"cell_x": px, "cell_y": py, "distance": dist})

            candidates.sort(key=lambda c: c["distance"])
            suggestions = candidates[:max(1, min(max_results, 15))]

            return {
                "building_type": building_type,
                "size": f"{bw}x{bh}",
                "cy_position": {"cell_x": cx, "cell_y": cy_y},
                "suggestions": suggestions,
            }

        # ── Unit Group Tools ────────────────────────────────────────────

        @mcp.tool()
        def assign_group(group_name: str, unit_ids: list[int]) -> dict:
            """Assign units to a named group (like Ctrl+1 in-game).
            Groups persist across turns. Use group names in other commands.
            Example: assign_group("attackers", [155, 160, 170])"""
            env._unit_groups[group_name] = list(unit_ids)
            return {"group": group_name, "unit_count": len(unit_ids), "unit_ids": unit_ids}

        @mcp.tool()
        def add_to_group(group_name: str, unit_ids: list[int]) -> dict:
            """Add units to an existing group (like Shift+Ctrl+1)."""
            existing = env._unit_groups.get(group_name, [])
            for uid in unit_ids:
                if uid not in existing:
                    existing.append(uid)
            env._unit_groups[group_name] = existing
            return {"group": group_name, "unit_count": len(existing), "unit_ids": existing}

        @mcp.tool()
        def get_groups() -> dict:
            """List all unit groups and their members."""
            # Prune dead units from groups
            env._refresh_obs()
            alive_ids = set()
            if env._last_obs:
                alive_ids = {u["actor_id"] for u in env._last_obs.get("units", [])}
            result = {}
            for name, ids in env._unit_groups.items():
                alive = [uid for uid in ids if uid in alive_ids]
                env._unit_groups[name] = alive  # auto-prune dead units
                if alive:
                    result[name] = alive
            return result

        @mcp.tool()
        def command_group(
            group_name: str,
            command: str,
            target_x: int = 0,
            target_y: int = 0,
            target_actor_id: int = 0,
            stance: str = "attack_anything",
        ) -> dict:
            """Send a command to all units in a named group.
            command: "attack_move", "move_units", "attack_target", "set_stance", "stop_units"
            For attack_move/move_units: provide target_x, target_y
            For attack_target: provide target_actor_id
            For set_stance: provide stance name"""
            ids = env._unit_groups.get(group_name, [])
            if not ids:
                return {"error": f"Group '{group_name}' not found or empty"}

            # Prune dead units
            env._refresh_obs()
            if env._last_obs:
                alive_ids = {u["actor_id"] for u in env._last_obs.get("units", [])}
                ids = [uid for uid in ids if uid in alive_ids]
                env._unit_groups[group_name] = ids
            if not ids:
                return {"error": f"All units in group '{group_name}' are dead"}

            if command == "attack_move":
                commands = [CommandModel(action=ActionType.ATTACK_MOVE, actor_id=uid,
                                        target_x=target_x, target_y=target_y) for uid in ids]
            elif command == "move_units":
                commands = [CommandModel(action=ActionType.MOVE, actor_id=uid,
                                        target_x=target_x, target_y=target_y) for uid in ids]
            elif command == "attack_target":
                commands = [CommandModel(action=ActionType.ATTACK, actor_id=uid,
                                        target_actor_id=target_actor_id) for uid in ids]
            elif command == "set_stance":
                stance_map = {"hold_fire": 0, "return_fire": 1, "defend": 2, "attack_anything": 3}
                stance_val = stance_map.get(stance.lower(), 3)
                commands = [CommandModel(action=ActionType.SET_STANCE, actor_id=uid,
                                        target_x=stance_val) for uid in ids]
            elif command == "stop_units":
                commands = [CommandModel(action=ActionType.STOP, actor_id=uid) for uid in ids]
            else:
                return {"error": f"Unknown group command '{command}'"}

            result = env._execute_commands(commands)
            result["group"] = group_name
            result["units_commanded"] = len(ids)
            return result

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

        @mcp.tool()
        def surrender() -> dict:
            """Surrender / resign the current game. Ends the game as a loss.
            Use this when you want to concede the match."""
            commands = [CommandModel(action=ActionType.SURRENDER)]
            return env._execute_commands(commands)

        @mcp.tool()
        def batch(actions: list[dict]) -> dict:
            """Send multiple commands that all execute concurrently (same game tick).

            Actions use same format as individual tools:
              {"tool": "build_unit", "unit_type": "e1", "count": 3}
              {"tool": "attack_move", "unit_ids": [155, 160], "target_x": 50, "target_y": 30}
              {"tool": "set_stance", "unit_ids": "all_combat", "stance": "attack_anything"}
              {"tool": "deploy_unit", "unit_id": 120}

            Special unit selectors (instead of listing IDs):
              "all_combat" — all own combat units
              "all_idle" — all idle combat units

            All commands are sent in a single call. The game resolves any
            conflicts by its own logic.

            Returns: game state summary after commands are processed.
            """
            env._refresh_obs()
            obs = env._last_obs
            if obs is None:
                return {"error": "No observation available"}
            if obs.get("done"):
                return {"error": "Game is over", "done": True, "result": obs.get("result", "")}

            all_commands = []
            action_names = []
            for action in actions:
                cmds = env._action_to_commands(action, obs)
                all_commands.extend(cmds)
                action_names.append(action.get("tool", "?"))

            if not all_commands:
                return {"error": "No valid commands generated"}

            try:
                result = env._execute_commands(all_commands)
                result["actions"] = action_names
                return result
            except Exception as e:
                return {"error": f"Command execution failed: {e}"}

        @mcp.tool()
        def plan(steps: list[dict]) -> dict:
            """Execute steps sequentially. Each step's commands are sent, then
            the observation is refreshed before the next step. Use conditions
            to gate steps on game state.

            Each step is a dict with:
              actions: list of action dicts to execute
              condition: optional — only execute if condition is met, else skip

            Conditions: "enemies_visible", "no_enemies_visible", "under_attack",
              "building_ready", "funds_above:2000", "funds_below:500"
              (funds = cash + ore; "cash_above"/"cash_below" also work)

            Example — deploy then build:
            [
              {"actions": [{"tool": "deploy_unit", "unit_id": 120}]},
              {"actions": [{"tool": "build_structure", "building_type": "powr"}]},
              {"condition": "building_ready",
               "actions": [{"tool": "place_building", "building_type": "powr"}]}
            ]

            Returns: game state summary + execution log.
            """
            execution_log = []
            start_tick = env._last_obs["tick"] if env._last_obs else 0

            for i, step in enumerate(steps):
                step_num = i + 1
                env._refresh_obs()
                obs = env._last_obs
                if obs is None:
                    execution_log.append(f"Step {step_num}: ERROR no observation")
                    break
                if obs.get("done"):
                    execution_log.append(f"Step {step_num}: game over")
                    break

                condition = step.get("condition")
                if condition and not env._check_plan_condition(condition, obs):
                    execution_log.append(f"Step {step_num}: SKIPPED ({condition} = false)")
                    continue

                actions = step.get("actions", [])
                all_commands = []
                action_names = []
                for action in actions:
                    cmds = env._action_to_commands(action, obs)
                    all_commands.extend(cmds)
                    action_names.append(action.get("tool", "?"))

                if all_commands:
                    try:
                        result = env._execute_commands(all_commands)
                        if result.get("done"):
                            execution_log.append(
                                f"Step {step_num}: {', '.join(action_names)} -> game over"
                            )
                            break
                    except Exception as e:
                        execution_log.append(
                            f"Step {step_num}: {', '.join(action_names)} -> ERROR {e}"
                        )
                        break

                execution_log.append(f"Step {step_num}: {', '.join(action_names)} OK")

            env._refresh_obs()
            obs = env._last_obs or {}
            end_tick = obs.get("tick", start_tick)
            executed = sum(1 for e in execution_log if "OK" in e)
            skipped = sum(1 for e in execution_log if "SKIPPED" in e)

            return {
                "steps_total": len(steps),
                "steps_executed": executed,
                "steps_skipped": skipped,
                "tick": end_tick,
                "done": obs.get("done", False),
                "result": obs.get("result", ""),
                "economy": obs.get("economy", {}),
                "own_units": len(obs.get("units", [])),
                "own_buildings": len(obs.get("buildings", [])),
                "visible_enemies": len(obs.get("visible_enemies", [])),
                "execution_log": execution_log,
            }

    # ── Internal helpers ─────────────────────────────────────────────────

    def _check_plan_condition(self, condition: str, obs: dict) -> bool:
        """Evaluate a plan condition against current observation."""
        if condition == "enemies_visible":
            return len(obs.get("visible_enemies", [])) > 0
        elif condition == "no_enemies_visible":
            return len(obs.get("visible_enemies", [])) == 0
        elif condition == "under_attack":
            for enemy in obs.get("visible_enemies", []):
                for bldg in obs.get("buildings", []):
                    dx = abs(enemy.get("cell_x", 0) - bldg.get("cell_x", 0))
                    dy = abs(enemy.get("cell_y", 0) - bldg.get("cell_y", 0))
                    if dx + dy < 12:
                        return True
            return False
        elif condition == "building_ready":
            return any(
                p["queue_type"] == "Building" and p["progress"] >= 0.99
                for p in obs.get("production", [])
            )
        elif condition.startswith("cash_above:") or condition.startswith("funds_above:"):
            threshold = int(condition.split(":")[1])
            eco = obs.get("economy", {})
            return eco.get("cash", 0) + eco.get("ore", 0) > threshold
        elif condition.startswith("cash_below:") or condition.startswith("funds_below:"):
            threshold = int(condition.split(":")[1])
            eco = obs.get("economy", {})
            return eco.get("cash", 0) + eco.get("ore", 0) < threshold
        return True  # unknown condition → proceed

    def _resolve_unit_ids(self, selector, obs: dict) -> list[int]:
        """Resolve unit selectors like 'all_combat', 'all_idle', or group names to actual actor IDs."""
        if isinstance(selector, list):
            return selector
        if selector == "all_combat":
            return [u["actor_id"] for u in obs.get("units", []) if u.get("can_attack")]
        if selector == "all_idle":
            return [u["actor_id"] for u in obs.get("units", [])
                    if u.get("can_attack") and u.get("is_idle")]
        # Check named groups
        if selector in self._unit_groups:
            return list(self._unit_groups[selector])
        return []

    def _action_to_commands(self, action: dict, obs: dict) -> list[CommandModel]:
        """Convert a plan action dict to a list of CommandModel objects."""
        tool = action.get("tool", "")
        unit_ids = self._resolve_unit_ids(action.get("unit_ids", []), obs)
        queued = action.get("queued", False)

        if tool == "build_unit":
            count = max(1, action.get("count", 1))
            return [CommandModel(action=ActionType.TRAIN, item_type=action["unit_type"])
                    for _ in range(count)]
        elif tool == "build_structure":
            return [CommandModel(action=ActionType.BUILD, item_type=action["building_type"])]
        elif tool == "build_and_place":
            btype = action["building_type"]
            self._pending_placements[btype] = {
                "cell_x": action.get("cell_x", 0), "cell_y": action.get("cell_y", 0)
            }
            return [CommandModel(action=ActionType.BUILD, item_type=btype)]
        elif tool == "place_building":
            return [CommandModel(action=ActionType.PLACE_BUILDING,
                                item_type=action["building_type"],
                                target_x=action.get("cell_x", 0), target_y=action.get("cell_y", 0))]
        elif tool == "attack_move":
            return [CommandModel(action=ActionType.ATTACK_MOVE, actor_id=uid,
                                target_x=action["target_x"], target_y=action["target_y"],
                                queued=queued)
                    for uid in unit_ids]
        elif tool == "move_units":
            return [CommandModel(action=ActionType.MOVE, actor_id=uid,
                                target_x=action["target_x"], target_y=action["target_y"],
                                queued=queued)
                    for uid in unit_ids]
        elif tool == "attack_target":
            return [CommandModel(action=ActionType.ATTACK, actor_id=uid,
                                target_actor_id=action["target_actor_id"], queued=queued)
                    for uid in unit_ids]
        elif tool == "set_stance":
            stance_map = {"hold_fire": 0, "return_fire": 1, "defend": 2, "attack_anything": 3}
            stance_val = stance_map.get(action.get("stance", "attack_anything").lower(), 3)
            return [CommandModel(action=ActionType.SET_STANCE, actor_id=uid, target_x=stance_val)
                    for uid in unit_ids]
        elif tool == "deploy_unit":
            return [CommandModel(action=ActionType.DEPLOY, actor_id=action["unit_id"])]
        elif tool == "set_rally_point":
            return [CommandModel(action=ActionType.SET_RALLY_POINT,
                                actor_id=action["building_id"],
                                target_x=action["cell_x"], target_y=action["cell_y"])]
        elif tool == "repair_building":
            return [CommandModel(action=ActionType.REPAIR, actor_id=action["building_id"])]
        elif tool == "stop_units":
            return [CommandModel(action=ActionType.STOP, actor_id=uid) for uid in unit_ids]
        elif tool == "harvest":
            return [CommandModel(action=ActionType.HARVEST, actor_id=action["unit_id"],
                                target_x=action.get("cell_x", 0),
                                target_y=action.get("cell_y", 0))]
        elif tool == "cancel_production":
            return [CommandModel(action=ActionType.CANCEL_PRODUCTION, item_type=action["item_type"])]
        elif tool == "surrender":
            return [CommandModel(action=ActionType.SURRENDER)]
        else:
            return []

    def _refresh_obs(self) -> None:
        """Update _last_obs from the bridge's background observation reader.

        In real-time mode, the game runs continuously. This fetches the
        latest cached observation so read tools return fresh state.
        """
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._bridge.observe(), self._loop
            )
            proto_obs = future.result(timeout=5)
            if proto_obs is not None:
                self._last_obs = observation_to_dict(proto_obs)
                self._state.game_tick = self._last_obs["tick"]
        except Exception:
            pass  # Keep existing _last_obs if refresh fails
        self._process_pending_placements()

    # Naval buildings that require water tiles
    _WATER_BUILDINGS = {"spen", "syrd"}

    def _process_pending_placements(self) -> None:
        """Auto-place buildings that finished construction via build_and_place.

        Uses a two-phase approach:
        1. When a building is ready, send PLACE_BUILDING and mark as "attempted"
        2. On next call, check if the building is still in queue — if so, placement failed
        """
        if not self._last_obs:
            return
        production = self._last_obs.get("production", [])
        attempted = getattr(self, "_attempted_placements", {})

        # Phase 2: Check previously attempted placements for failure
        failed = []
        for btype, attempts in list(attempted.items()):
            still_in_queue = any(
                p["queue_type"] == "Building" and p["item"] == btype and p["progress"] >= 0.99
                for p in production
            )
            if still_in_queue:
                # Building is still in queue → placement failed
                if attempts >= 2:
                    # Multiple attempts failed — report and auto-cancel
                    if btype in self._WATER_BUILDINGS:
                        reason = f"{btype} requires water tiles — must be placed on water, not land"
                    else:
                        reason = "no valid placement found near base"
                    self._placement_results.append(
                        f"PLACEMENT FAILED: {btype} — {reason}. "
                        f"Auto-cancelling. Use cancel_production(\"{btype}\") if stuck."
                    )
                    # Auto-cancel the stuck production
                    try:
                        cancel_cmd = [CommandModel(action=ActionType.CANCEL_PRODUCTION, item_type=btype)]
                        self._execute_commands(cancel_cmd)
                    except Exception:
                        pass
                    failed.append(btype)
                else:
                    # Retry once with different offset
                    attempted[btype] = attempts + 1
            else:
                # Building no longer in queue → placement succeeded
                self._placement_results.append(f"AUTO-PLACED: {btype}")
                failed.append(btype)  # remove from attempted tracking

        for btype in failed:
            attempted.pop(btype, None)
            self._pending_placements.pop(btype, None)

        # Phase 1: Send placement commands for newly ready buildings
        if not getattr(self, "_pending_placements", None):
            return
        for btype, coords in list(self._pending_placements.items()):
            if btype in attempted:
                continue  # already being tracked
            ready = any(
                p["queue_type"] == "Building" and p["item"] == btype and p["progress"] >= 0.99
                for p in production
            )
            if not ready:
                continue
            try:
                commands = [CommandModel(
                    action=ActionType.PLACE_BUILDING,
                    item_type=btype,
                    target_x=coords["cell_x"],
                    target_y=coords["cell_y"],
                )]
                self._execute_commands(commands)
                attempted[btype] = 1
            except Exception:
                self._placement_results.append(
                    f"PLACEMENT FAILED: {btype} — command error. "
                    f"Use cancel_production(\"{btype}\") to clear queue."
                )
                self._pending_placements.pop(btype, None)

    def _execute_commands(self, commands: list[CommandModel]) -> dict:
        """Send commands, step the game, update cache, return summary."""
        action = OpenRAAction(commands=commands)
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._async_step_internal(action), self._loop
            )
            obs_dict = future.result(timeout=300)
            self._last_obs = obs_dict
        except Exception:
            # Connection lost — check if game ended while we weren't looking
            self._refresh_obs()
            obs_dict = self._last_obs
            if obs_dict is None:
                raise
            if not obs_dict.get("done"):
                raise

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
        """Get the OpenRA replays directory for current mod.

        OpenRA stores replays at {SupportDir}/Replays/{mod}/{version}/.
        On macOS: ~/Library/Application Support/OpenRA/
        On Linux: ~/.config/openra/ (modern) or ~/.openra/ (legacy)
        Also checks {EngineDir}/Support/ (local override).
        """
        candidates = []

        # Local Support dir (takes priority if it exists)
        engine_support = Path(self._config.openra_path) / "Support"
        if engine_support.exists():
            candidates.append(engine_support / "Replays" / self._config.mod)

        if sys.platform == "darwin":
            candidates.append(Path.home() / "Library/Application Support/OpenRA/Replays" / self._config.mod)
        else:
            # Modern path (XDG_CONFIG_HOME or ~/.config/openra)
            xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
            candidates.append(Path(xdg) / "openra/Replays" / self._config.mod)
            # Legacy path
            candidates.append(Path.home() / ".openra/Replays" / self._config.mod)

        for base in candidates:
            if base.exists():
                return base

        # Fallback: return first candidate (will be created if needed)
        return candidates[0]

    # ── OpenEnv Interface ────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OpenRAObservation:
        """Reset the environment for a new episode."""
        future = asyncio.run_coroutine_threadsafe(
            self._async_reset(seed, episode_id, **kwargs), self._loop
        )
        return future.result(timeout=300)

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
        self._unit_groups.clear()
        self._pending_placements.clear()
        self._attempted_placements.clear()
        self._placement_results.clear()

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

        # Get faction info from GameState
        try:
            game_state = await self._bridge.get_state()
            self._player_faction = game_state.player_faction or ""
            self._enemy_faction = game_state.enemy_faction or ""
        except Exception:
            self._player_faction = ""
            self._enemy_faction = ""

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
            future = asyncio.run_coroutine_threadsafe(
                self._async_step_internal(action), self._loop
            )
            obs_dict = future.result(timeout=300)
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
            future = asyncio.run_coroutine_threadsafe(
                self._bridge.close(), self._loop
            )
            future.result(timeout=10)
        except Exception:
            pass
        self._process.kill()
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
            self._loop.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
