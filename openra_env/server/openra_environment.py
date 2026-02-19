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
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

from openra_env.game_data import (
    get_all_building_types,
    get_all_buildings_for_side,
    get_all_unit_types,
    get_all_units_for_side,
    get_building_stats,
    get_faction_info,
    get_tech_tree,
    get_unit_stats,
)
from openra_env.opponent_intel import get_opponent_profile, get_opponent_summary
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
        planning_enabled: bool = True,
        planning_max_turns: int = 10,
        planning_max_time_s: float = 60.0,
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
        self._last_production_progress: dict[str, float] = {}  # item → progress for stall detection
        self._prev_buildings: dict[int, str] = {}  # actor_id → type for loss detection
        self._prev_unit_ids: dict[int, str] = {}  # actor_id → type for loss detection
        self._enemy_ever_seen: bool = False  # suppress NO SCOUTING after first contact

        # Planning phase configuration (env vars override constructor args)
        planning_env = os.environ.get("PLANNING_ENABLED", "")
        if planning_env:
            self._planning_enabled = planning_env.lower() in ("true", "1", "yes")
        else:
            self._planning_enabled = planning_enabled
        self._planning_max_turns = int(os.environ.get("PLANNING_MAX_TURNS", str(planning_max_turns)))
        self._planning_max_time_s = float(os.environ.get("PLANNING_MAX_TIME", str(planning_max_time_s)))
        self._planning_active = False
        self._planning_start_time: float = 0.0
        self._planning_turns_used: int = 0
        self._planning_strategy: str = ""

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
            attackers = []  # enemies near base buildings
            for enemy in obs["visible_enemies"]:
                for bldg in obs["buildings"]:
                    dx = abs(enemy.get("cell_x", 0) - bldg.get("cell_x", 0))
                    dy = abs(enemy.get("cell_y", 0) - bldg.get("cell_y", 0))
                    if dx + dy < 12:
                        attackers.append(enemy)
                        break
            if len(attackers) <= 3:
                for enemy in attackers:
                    alerts.append(
                        f"UNDER ATTACK: enemy {enemy['type']} id={enemy['actor_id']} "
                        f"near base"
                    )
            elif attackers:
                from collections import Counter
                type_counts = Counter(e["type"] for e in attackers)
                breakdown = ", ".join(f"{cnt}x {t}" for t, cnt in type_counts.most_common())
                alerts.append(f"UNDER ATTACK: {len(attackers)} enemies near base ({breakdown})")

            # Damaged buildings
            for bldg in obs["buildings"]:
                if bldg["hp_percent"] < 0.5:
                    alerts.append(f"DAMAGED: {bldg['type']} at {bldg['hp_percent']:.0%} HP — repair_building({bldg['actor_id']})")

            # Power crisis — 1/3 speed is devastating
            if power_balance < 0:
                alerts.append(
                    f"LOW POWER: {power_balance:+d} — ALL production slowed to 1/3 speed! "
                    f"Build powr IMMEDIATELY, it is your #1 priority"
                )
            elif 0 <= power_balance < 30:
                building_power = any(
                    p["item"] in ("powr", "apwr")
                    for p in obs["production"]
                )
                if not building_power:
                    alerts.append(
                        f"POWER TIGHT: only {power_balance:+d} surplus — "
                        f"build powr before adding more buildings"
                    )

            # Idle funds with few harvesters
            total_funds = eco["cash"] + eco.get("ore", 0)
            if total_funds > 2000 and eco["harvester_count"] < 2:
                alerts.append(f"IDLE FUNDS: ${total_funds} with {eco['harvester_count']} harvester(s) — build refinery or harvester")

            # Ore storage near capacity — income is being wasted
            ore = eco.get("ore", 0)
            res_cap = eco.get("resource_capacity", 0)
            if res_cap > 0 and ore >= res_cap * 0.9:
                alerts.append(
                    f"ORE FULL: {ore}/{res_cap} storage used — "
                    f"build silo ($150, +1500 capacity) or refinery to avoid wasting income"
                )

            # Nothing being produced
            if not obs["production"] and len(obs["buildings"]) >= 3:
                alerts.append("IDLE PRODUCTION: nothing being built or trained — queue something")

            # Production stalled due to $0 funds
            total_funds = eco["cash"] + eco.get("ore", 0)
            current_progress = {p["item"]: p["progress"] for p in obs["production"]
                                if p["progress"] < 0.99}
            last_progress = getattr(env, "_last_production_progress", {})
            if total_funds == 0 and current_progress:
                for item, prog in current_progress.items():
                    if item in last_progress and abs(prog - last_progress[item]) < 0.01:
                        alerts.append(
                            f"STALLED: {item}@{prog:.0%} — $0 funds, "
                            f"construction pauses without income. "
                            f"Cancel or build refinery/harvester first."
                        )
                        break  # one alert is enough
            env._last_production_progress = current_progress

            # Building ready to place or stuck in auto-placement
            pending = getattr(env, "_pending_placements", {})
            attempted = getattr(env, "_attempted_placements", {})
            for p in obs["production"]:
                if p["queue_type"] in env._PLACEABLE_QUEUE_TYPES and p["progress"] >= 0.99:
                    btype = p["item"]
                    if btype in attempted:
                        alerts.append(f"BUILDING STUCK: {btype} placement failing — call get_valid_placements(\"{btype}\") or cancel_production(\"{btype}\")")
                    elif btype not in pending:
                        alerts.append(f"READY TO PLACE: {btype} — call place_building()")

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

            # Idle army — nudge to attack or scout
            idle_combat = [u for u in obs["units"] if u.get("can_attack") and u.get("is_idle")]
            if len(idle_combat) >= 4:
                alerts.append(
                    f"IDLE ARMY: {len(idle_combat)} combat units idle — "
                    f"attack_move toward enemy or scout unexplored areas"
                )

            # No defenses — nudge to build turrets
            _DEFENSE_BUILDINGS = {"gun", "ftur", "tsla", "sam", "agun", "pbox", "hbox"}
            building_types = {b["type"] for b in obs["buildings"]}
            if len(obs["buildings"]) >= 4 and not (building_types & _DEFENSE_BUILDINGS):
                alerts.append("NO DEFENSES: build gun turrets or SAM sites to protect your base")

            # Track enemy discovery history
            if obs.get("visible_enemies") or obs.get("visible_enemy_buildings"):
                env._enemy_ever_seen = True

            # No scouting — nudge to explore (suppress after first contact)
            if obs["tick"] > 750 and not obs["visible_enemies"] and not obs.get("visible_enemy_buildings"):
                if not getattr(env, "_enemy_ever_seen", False):
                    alerts.append(
                        "NO SCOUTING: you haven't found the enemy — send a unit to explore the map"
                    )

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

            result = {
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

            # Include planning phase context
            if env._planning_active:
                result["planning_active"] = True
                result["planning_turns_remaining"] = max(
                    0, env._planning_max_turns - env._planning_turns_used
                )
            if env._planning_strategy:
                result["planning_strategy"] = env._planning_strategy

            return result

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
                is_passable = passable > 0.5
                tidx = int(terrain_idx)
                if is_passable:
                    note = "Passable terrain."
                elif tidx in (7, 8):
                    note = "Water — impassable to land units. spen/syrd require water."
                else:
                    note = "Impassable terrain (cliff or obstacle)."
                return {
                    "cell_x": cell_x,
                    "cell_y": cell_y,
                    "terrain_index": tidx,
                    "passable": is_passable,
                    "note": note,
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

        # ── Bulk Knowledge Tools (rich context in one call) ─────────────

        @mcp.tool()
        def get_faction_briefing() -> dict:
            """Get complete briefing for your faction: all available units with
            full stats, all available buildings with full stats, tech tree, and
            faction info. One call gives you everything about your faction's
            military capabilities. Ideal for planning phase."""
            faction = env._player_faction
            if not faction:
                # Infer from observation
                env._refresh_obs()
                obs = env._last_obs
                if obs:
                    avail = obs.get("available_production", [])
                    bldg_types = [b["type"] for b in obs.get("buildings", [])]
                    if "tent" in avail or "tent" in bldg_types:
                        faction = "england"
                    else:
                        faction = "russia"

            faction_info = get_faction_info(faction)
            if faction_info is None:
                return {"error": f"Could not determine faction (got '{faction}')"}

            side = faction_info["side"]
            units = get_all_units_for_side(side)
            buildings = get_all_buildings_for_side(side)
            tech_tree = get_tech_tree(side)

            return {
                "faction": faction,
                "side": side,
                "description": faction_info.get("description", ""),
                "unique_units": faction_info.get("unique_units", []),
                "tech_tree": tech_tree.get(side, []),
                "units": units,
                "buildings": buildings,
            }

        @mcp.tool()
        def get_map_analysis() -> dict:
            """Analyze the map and produce a strategic summary: resource patch
            locations, water presence, passability overview, quadrant breakdown,
            and key terrain features. Essential for planning."""
            env._refresh_obs()
            obs = env._last_obs
            if obs is None:
                return {"error": "No observation available. Call reset first."}

            map_info = obs.get("map_info", {})
            w = map_info.get("width", 0)
            h = map_info.get("height", 0)
            channels = obs.get("spatial_channels", 0)
            spatial = obs.get("spatial_map", "")

            # Base/enemy position
            buildings = obs.get("buildings", [])
            units = obs.get("units", [])
            all_pos = (
                [(b["cell_x"], b["cell_y"]) for b in buildings]
                + [(u["cell_x"], u["cell_y"]) for u in units]
            )
            if all_pos:
                base_x = sum(p[0] for p in all_pos) // len(all_pos)
                base_y = sum(p[1] for p in all_pos) // len(all_pos)
            else:
                base_x, base_y = w // 2, h // 2
            enemy_x = max(2, min(w - 2, w - base_x))
            enemy_y = max(2, min(h - 2, h - base_y))
            distance = abs(enemy_x - base_x) + abs(enemy_y - base_y)

            result = {
                "map_name": map_info.get("map_name", "?"),
                "width": w,
                "height": h,
                "base_position": {"x": base_x, "y": base_y},
                "enemy_estimated_position": {"x": enemy_x, "y": enemy_y},
                "base_to_enemy_distance": distance,
            }

            if not spatial or w == 0 or channels == 0:
                result["note"] = "No spatial data available for detailed analysis"
                return result

            import base64
            import struct

            try:
                raw = base64.b64decode(spatial)
            except Exception:
                result["note"] = "Failed to decode spatial data"
                return result

            total_cells = w * h
            passable_count = 0
            water_count = 0
            resource_cells = []
            half_w = w // 2
            half_h = h // 2
            quad_stats = {
                "NW": {"passable": 0, "total": 0, "resources": 0},
                "NE": {"passable": 0, "total": 0, "resources": 0},
                "SW": {"passable": 0, "total": 0, "resources": 0},
                "SE": {"passable": 0, "total": 0, "resources": 0},
            }

            for y in range(h):
                for x in range(w):
                    base_idx = (y * w + x) * channels
                    try:
                        passable = struct.unpack_from("f", raw, (base_idx + 3) * 4)[0]
                        resource = struct.unpack_from("f", raw, (base_idx + 2) * 4)[0]
                    except struct.error:
                        continue

                    # Determine quadrant
                    quad = ("N" if y < half_h else "S") + ("W" if x < half_w else "E")
                    quad_stats[quad]["total"] += 1

                    if passable > 0.5:
                        passable_count += 1
                        quad_stats[quad]["passable"] += 1
                    else:
                        water_count += 1

                    if resource > 0:
                        resource_cells.append((x, y, resource))
                        quad_stats[quad]["resources"] += 1

            # Passability
            passable_ratio = passable_count / max(total_cells, 1)
            result["passable_ratio"] = round(passable_ratio, 2)
            result["has_water"] = water_count > (total_cells * 0.02)

            # Map type
            water_ratio = water_count / max(total_cells, 1)
            if water_ratio > 0.4:
                map_type = "island/naval"
            elif water_ratio > 0.1:
                map_type = "mixed terrain"
            elif passable_ratio > 0.8:
                map_type = "open land"
            else:
                map_type = "confined terrain"
            result["map_type"] = map_type

            # Cluster resource cells into patches using simple grid-based grouping
            patches = []
            if resource_cells:
                visited = set()
                for rx, ry, rd in resource_cells:
                    if (rx, ry) in visited:
                        continue
                    # BFS to find connected resource cells (8-connectivity, radius 3)
                    cluster = []
                    queue = [(rx, ry)]
                    visited.add((rx, ry))
                    while queue:
                        cx, cy = queue.pop(0)
                        # Find density for this cell
                        cell_density = 0
                        for rcx, rcy, rcd in resource_cells:
                            if rcx == cx and rcy == cy:
                                cell_density = rcd
                                break
                        cluster.append((cx, cy, cell_density))
                        for dx in range(-3, 4):
                            for dy in range(-3, 4):
                                nx, ny = cx + dx, cy + dy
                                if (nx, ny) not in visited:
                                    for rcx, rcy, rcd in resource_cells:
                                        if rcx == nx and rcy == ny:
                                            visited.add((nx, ny))
                                            queue.append((nx, ny))
                                            break

                    if len(cluster) >= 2:  # Only report meaningful patches
                        cx = sum(c[0] for c in cluster) // len(cluster)
                        cy = sum(c[1] for c in cluster) // len(cluster)
                        total_density = sum(c[2] for c in cluster)
                        near_base = abs(cx - base_x) + abs(cy - base_y) < distance // 3
                        patches.append({
                            "center_x": cx,
                            "center_y": cy,
                            "cells": len(cluster),
                            "total_density": round(total_density, 1),
                            "near_base": near_base,
                        })

            # Sort patches: nearest to base first
            patches.sort(key=lambda p: abs(p["center_x"] - base_x) + abs(p["center_y"] - base_y))
            result["resource_patches"] = patches[:10]  # Cap at 10

            # Quadrant summary
            quadrant_summary = {}
            for quad, stats in quad_stats.items():
                total = max(stats["total"], 1)
                note = ""
                if quad == ("N" if base_y < half_h else "S") + ("W" if base_x < half_w else "E"):
                    note = "your base area"
                elif quad == ("N" if enemy_y < half_h else "S") + ("W" if enemy_x < half_w else "E"):
                    note = "enemy base area"
                quadrant_summary[quad] = {
                    "passable_ratio": round(stats["passable"] / total, 2),
                    "resource_cells": stats["resources"],
                    "note": note,
                }
            result["quadrant_summary"] = quadrant_summary

            # Strategic notes
            notes = []
            if result["has_water"]:
                notes.append("Naval buildings possible — water detected on map")
            else:
                notes.append("Land-only map — skip naval buildings (spen/syrd)")
            if patches:
                nearest = patches[0]
                dist_to_ore = abs(nearest["center_x"] - base_x) + abs(nearest["center_y"] - base_y)
                notes.append(f"Nearest ore patch at ({nearest['center_x']},{nearest['center_y']}), "
                           f"{dist_to_ore} cells from base, {nearest['cells']} resource cells")
            if distance < 40:
                notes.append("Short distance to enemy — expect early aggression, prioritize defense")
            elif distance > 100:
                notes.append("Long distance to enemy — time to build economy before attacking")
            result["strategic_notes"] = notes

            return result

        @mcp.tool()
        def batch_lookup(queries: list[dict]) -> dict:
            """Look up multiple units, buildings, factions, or tech trees in one call.
            Each query: {"type": "unit"|"building"|"faction"|"tech_tree", "name": "..."}
            Example: [{"type":"unit","name":"3tnk"}, {"type":"building","name":"weap"}]
            Returns all results at once — efficient for researching multiple items."""
            results = []
            for q in queries:
                qtype = q.get("type", "").lower()
                name = q.get("name", "")
                if qtype == "unit":
                    data = get_unit_stats(name)
                    if data is None:
                        results.append({"error": f"Unknown unit '{name}'", "query": q})
                    else:
                        results.append({"type": "unit", "name": name, **data})
                elif qtype == "building":
                    data = get_building_stats(name)
                    if data is None:
                        results.append({"error": f"Unknown building '{name}'", "query": q})
                    else:
                        results.append({"type": "building", "name": name, **data})
                elif qtype == "faction":
                    data = get_faction_info(name)
                    if data is None:
                        results.append({"error": f"Unknown faction '{name}'", "query": q})
                    else:
                        results.append({"type": "faction", **data})
                elif qtype == "tech_tree":
                    data = get_tech_tree(name)
                    results.append({"type": "tech_tree", "name": name, **data})
                else:
                    results.append({"error": f"Unknown query type '{qtype}'", "query": q})
            return {"results": results, "count": len(results)}

        # ── Planning Phase Tools ────────────────────────────────────────

        @mcp.tool()
        def get_opponent_intel() -> dict:
            """Get intelligence report on the opponent AI. Returns behavioral
            profile, win rate, typical strategy, recommended counters, and
            recent match history. Use this during planning to prepare your strategy."""
            difficulty = env._config.bot_type  # "easy", "normal", "hard"
            profile = get_opponent_profile(difficulty)
            if profile is None:
                return {
                    "difficulty": difficulty,
                    "note": "No detailed profile available for this opponent type.",
                }
            result = dict(profile)
            result["your_faction"] = env._player_faction
            result["enemy_faction"] = env._enemy_faction
            return result

        @mcp.tool()
        def start_planning_phase() -> dict:
            """Begin the pre-game planning phase. Returns map metadata, faction info,
            opponent intelligence, tech tree, and available units/buildings.

            During planning, use game knowledge tools (lookup_unit, lookup_building,
            lookup_tech_tree, lookup_faction) and get_opponent_intel() to formulate
            your strategy. End planning with end_planning_phase(strategy=...).

            Planning has a turn limit and time limit. If exceeded, planning ends
            automatically."""
            if not env._planning_enabled:
                return {
                    "planning_enabled": False,
                    "message": "Planning phase is disabled. Proceed directly to gameplay.",
                }

            if env._planning_active:
                return {
                    "error": "Planning phase already active.",
                    "turns_used": env._planning_turns_used,
                    "turns_remaining": max(0, env._planning_max_turns - env._planning_turns_used),
                }

            env._planning_active = True
            env._planning_start_time = time.time()
            env._planning_turns_used = 0
            env._planning_strategy = ""

            # Gather initial game metadata
            env._refresh_obs()
            obs = env._last_obs or {}

            map_info = obs.get("map_info", {})
            buildings = obs.get("buildings", [])
            units = obs.get("units", [])

            # Base position
            all_positions = (
                [(b["cell_x"], b["cell_y"]) for b in buildings]
                + [(u["cell_x"], u["cell_y"]) for u in units]
            )
            if all_positions:
                base_x = sum(p[0] for p in all_positions) // len(all_positions)
                base_y = sum(p[1] for p in all_positions) // len(all_positions)
            else:
                base_x = map_info.get("width", 128) // 2
                base_y = map_info.get("height", 128) // 2

            # Enemy spawn estimate (opposite side of map)
            map_w = map_info.get("width", 128)
            map_h = map_info.get("height", 128)
            enemy_x = max(2, min(map_w - 2, map_w - base_x))
            enemy_y = max(2, min(map_h - 2, map_h - base_y))

            # Faction and tech tree
            faction = env._player_faction
            enemy_faction = env._enemy_faction
            faction_info = get_faction_info(faction) if faction else None
            side = faction_info["side"] if faction_info else "unknown"
            tech_tree = get_tech_tree(side) if side != "unknown" else {}

            # Opponent intel
            opponent_profile = get_opponent_profile(env._config.bot_type)
            opponent_summary = get_opponent_summary(env._config.bot_type)

            # Key units/buildings with full stats (top 8 by cost)
            key_units = {}
            key_buildings = {}
            if side != "unknown":
                all_units = get_all_units_for_side(side)
                sorted_units = sorted(all_units.items(), key=lambda x: x[1].get("cost", 0), reverse=True)
                for utype, udata in sorted_units[:8]:
                    key_units[utype] = udata
                all_bldgs = get_all_buildings_for_side(side)
                sorted_bldgs = sorted(all_bldgs.items(), key=lambda x: x[1].get("cost", 0), reverse=True)
                for btype, bdata in sorted_bldgs[:8]:
                    key_buildings[btype] = bdata

            return {
                "planning_active": True,
                "max_turns": env._planning_max_turns,
                "max_time_seconds": env._planning_max_time_s,
                "map": map_info,
                "base_position": {"x": base_x, "y": base_y},
                "enemy_estimated_position": {"x": enemy_x, "y": enemy_y},
                "your_faction": faction,
                "your_side": side,
                "enemy_faction": enemy_faction,
                "tech_tree": tech_tree,
                "available_units": faction_info.get("available_units", []) if faction_info else [],
                "available_buildings": faction_info.get("available_buildings", []) if faction_info else [],
                "key_units": key_units,
                "key_buildings": key_buildings,
                "starting_units": [
                    {"type": u["type"], "id": u["actor_id"], "cell_x": u["cell_x"], "cell_y": u["cell_y"]}
                    for u in units
                ],
                "starting_buildings": [
                    {"type": b["type"], "id": b["actor_id"], "cell_x": b["cell_x"], "cell_y": b["cell_y"]}
                    for b in buildings
                ],
                "opponent_intel": opponent_profile,
                "opponent_summary": opponent_summary,
                "instructions": (
                    "You are in PLANNING MODE. Formulate your strategy before the game begins. "
                    "Use get_faction_briefing() for ALL unit/building stats in one call, "
                    "get_map_analysis() for terrain/resource intel, "
                    "and get_opponent_intel() for enemy behavioral profile. "
                    "Use batch_lookup() to look up multiple items at once. "
                    "When ready, call end_planning_phase(strategy='your strategy here') "
                    "to begin gameplay."
                ),
            }

        @mcp.tool()
        def end_planning_phase(strategy: str = "") -> dict:
            """End the planning phase and transition to gameplay.

            Args:
                strategy: Your formulated strategy as a text summary. This will be
                         available as context during gameplay.

            Returns game state summary to begin gameplay."""
            if not env._planning_active:
                return {
                    "error": "No planning phase active.",
                    "planning_enabled": env._planning_enabled,
                }

            elapsed = time.time() - env._planning_start_time
            env._planning_active = False
            env._planning_strategy = strategy.strip() if strategy else ""
            env._planning_turns_used = env._planning_turns_used
            env._state.planning_strategy = env._planning_strategy
            env._state.planning_turns_used = env._planning_turns_used

            # Get current game state to hand off
            env._refresh_obs()
            obs = env._last_obs or {}

            return {
                "planning_complete": True,
                "planning_duration_seconds": round(elapsed, 1),
                "planning_turns_used": env._planning_turns_used,
                "strategy_recorded": bool(env._planning_strategy),
                "strategy": env._planning_strategy,
                "tick": obs.get("tick", 0),
                "economy": obs.get("economy", {}),
                "own_units": len(obs.get("units", [])),
                "own_buildings": len(obs.get("buildings", [])),
                "message": "Planning complete. Game is live. Deploy your MCV and execute your strategy!",
            }

        @mcp.tool()
        def get_planning_status() -> dict:
            """Check the status of the planning phase — turns used, time remaining."""
            if not env._planning_enabled:
                return {"planning_enabled": False}
            if not env._planning_active:
                return {
                    "planning_active": False,
                    "strategy": env._planning_strategy or "(none)",
                }

            elapsed = time.time() - env._planning_start_time
            return {
                "planning_active": True,
                "turns_used": env._planning_turns_used,
                "turns_remaining": max(0, env._planning_max_turns - env._planning_turns_used),
                "time_elapsed_seconds": round(elapsed, 1),
                "time_remaining_seconds": round(max(0, env._planning_max_time_s - elapsed), 1),
            }

        # ── Action Tools (advance game state) ────────────────────────────

        @mcp.tool()
        def advance(ticks: int = 1) -> dict:
            """Wait for the game to advance by the specified number of ticks
            (max 500 per call). The game runs at normal speed (~25 ticks/sec).
            Use this to let time pass without issuing commands (e.g., while
            waiting for production to complete). Returns updated game summary."""
            requested = ticks
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
            # Track losses and trigger auto-placement
            env._update_loss_tracking()
            env._process_pending_placements()
            result = {
                "tick": obs_dict["tick"],
                "done": obs_dict["done"],
                "result": obs_dict.get("result", ""),
                "economy": obs_dict["economy"],
                "own_units": len(obs_dict["units"]),
                "own_buildings": len(obs_dict["buildings"]),
                "visible_enemies": len(obs_dict["visible_enemies"]),
            }
            if requested > 500:
                result["note"] = f"Clamped from {requested} to 500 ticks (max per call)."
            return result

        @mcp.tool()
        def move_units(unit_ids: str, target_x: int, target_y: int, queued: bool = False) -> dict:
            """Move units to a map cell position. Units pathfind automatically.
            unit_ids: comma-separated actor IDs (e.g. "145,146"), "all_combat", "all_idle", or a group name."""
            env._refresh_obs()
            resolved = env._resolve_unit_ids(unit_ids, env._last_obs or {})
            if not resolved:
                return {"error": "No matching units found"}
            commands = [
                CommandModel(action=ActionType.MOVE, actor_id=uid, target_x=target_x, target_y=target_y, queued=queued)
                for uid in resolved
            ]
            result = env._execute_commands(commands)
            return env._add_unit_feedback(result, resolved)

        @mcp.tool()
        def attack_move(unit_ids: str, target_x: int, target_y: int, queued: bool = False) -> dict:
            """Move units toward a cell, attacking enemies encountered along the way.
            unit_ids: comma-separated actor IDs (e.g. "145,146"), "all_combat", "all_idle", or a group name."""
            env._refresh_obs()
            resolved = env._resolve_unit_ids(unit_ids, env._last_obs or {})
            if not resolved:
                return {"error": "No matching units found"}
            commands = [
                CommandModel(action=ActionType.ATTACK_MOVE, actor_id=uid, target_x=target_x, target_y=target_y, queued=queued)
                for uid in resolved
            ]
            result = env._execute_commands(commands)
            return env._add_unit_feedback(result, resolved)

        @mcp.tool()
        def attack_target(unit_ids: str, target_actor_id: int, queued: bool = False) -> dict:
            """Order units to attack a specific enemy actor by ID.
            unit_ids: comma-separated actor IDs (e.g. "145,146"), "all_combat", "all_idle", or a group name."""
            env._refresh_obs()
            resolved = env._resolve_unit_ids(unit_ids, env._last_obs or {})
            if not resolved:
                return {"error": "No matching units found"}
            commands = [
                CommandModel(action=ActionType.ATTACK, actor_id=uid, target_actor_id=target_actor_id, queued=queued)
                for uid in resolved
            ]
            result = env._execute_commands(commands)
            return env._add_unit_feedback(result, resolved)

        @mcp.tool()
        def stop_units(unit_ids: str) -> dict:
            """Stop units from their current activity.
            unit_ids: comma-separated actor IDs (e.g. "145,146"), "all_combat", "all_idle", or a group name."""
            env._refresh_obs()
            resolved = env._resolve_unit_ids(unit_ids, env._last_obs or {})
            if not resolved:
                return {"error": "No matching units found"}
            commands = [CommandModel(action=ActionType.STOP, actor_id=uid) for uid in resolved]
            result = env._execute_commands(commands)
            return env._add_unit_feedback(result, resolved)

        @mcp.tool()
        def build_unit(unit_type: str, count: int = 1) -> dict:
            """Start training units (infantry, vehicle, aircraft, ship).
            The unit_type is the internal name (e.g., 'e1', '3tnk', 'mig').
            Use count > 1 to queue multiple of the same type."""
            # Validate against available production
            env._refresh_obs()
            if env._last_obs:
                available = env._last_obs.get("available_production", [])
                if available and unit_type not in available:
                    all_buildings = get_all_building_types()
                    avail_units = [u for u in available if u not in all_buildings]
                    diag = env._diagnose_unavailable(unit_type)
                    result = {
                        "error": diag["reason"],
                        "available_units": avail_units,
                    }
                    if "missing_prerequisites" in diag:
                        result["missing_prerequisites"] = diag["missing_prerequisites"]
                    return result
                # Check funds
                eco = env._last_obs.get("economy", {})
                total_funds = eco.get("cash", 0) + eco.get("ore", 0)
                unit_stats = get_unit_stats(unit_type)
                unit_cost = unit_stats["cost"] if unit_stats else 0
                if unit_cost > 0 and total_funds < unit_cost:
                    return {
                        "error": f"Insufficient funds: ${total_funds} available, "
                                 f"{unit_type} costs ${unit_cost}. "
                                 f"Build refinery or wait for income.",
                    }
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
            # Reject if same building already in production queue
            env._refresh_obs()
            if env._last_obs:
                available = env._last_obs.get("available_production", [])
                if available and building_type not in available:
                    all_buildings = get_all_building_types()
                    avail_bldgs = [b for b in available if b in all_buildings]
                    diag = env._diagnose_unavailable(building_type)
                    result = {
                        "error": diag["reason"],
                        "available_buildings": avail_bldgs,
                    }
                    if "missing_prerequisites" in diag:
                        result["missing_prerequisites"] = diag["missing_prerequisites"]
                    return result
                already = any(
                    p["queue_type"] in env._PLACEABLE_QUEUE_TYPES and p["item"] == building_type
                    for p in env._last_obs.get("production", [])
                )
                if already:
                    return {"error": f"'{building_type}' is already being built. Wait for it to finish or cancel_production(\"{building_type}\") first."}
            commands = [CommandModel(action=ActionType.BUILD, item_type=building_type)]
            result = env._execute_commands(commands)
            # Proactive power warning
            stats = get_building_stats(building_type)
            if stats and env._last_obs:
                power_drain = stats.get("power", 0)
                if power_drain < 0:
                    eco = env._last_obs.get("economy", {})
                    current_balance = eco.get("power_provided", 0) - eco.get("power_drained", 0)
                    if current_balance + power_drain < 0:
                        result["warning"] = (
                            f"POWER WARNING: {building_type} drains {abs(power_drain)} power. "
                            f"Current balance: {current_balance:+d}. "
                            f"Build powr first or production will slow to 1/3 speed!"
                        )
            return result

        @mcp.tool()
        def build_and_place(building_type: str, cell_x: int = 0, cell_y: int = 0) -> dict:
            """Build a structure and auto-place it when construction finishes.
            Coordinates are optional — the engine auto-finds a valid position
            near your base if omitted or invalid.
            This is the preferred way to build — no need to call place_building separately."""
            # Validate and reject duplicates
            env._refresh_obs()
            if env._last_obs:
                available = env._last_obs.get("available_production", [])
                if available and building_type not in available:
                    all_buildings = get_all_building_types()
                    avail_bldgs = [b for b in available if b in all_buildings]
                    diag = env._diagnose_unavailable(building_type)
                    result = {
                        "error": diag["reason"],
                        "available_buildings": avail_bldgs,
                    }
                    if "missing_prerequisites" in diag:
                        result["missing_prerequisites"] = diag["missing_prerequisites"]
                    return result
                already = any(
                    p["queue_type"] in env._PLACEABLE_QUEUE_TYPES and p["item"] == building_type
                    for p in env._last_obs.get("production", [])
                )
                if already:
                    return {"error": f"'{building_type}' is already being built. Wait for it to finish or cancel_production(\"{building_type}\") first."}
            commands = [CommandModel(action=ActionType.BUILD, item_type=building_type)]
            result = env._execute_commands(commands)
            env._pending_placements[building_type] = {"cell_x": cell_x, "cell_y": cell_y}
            # Proactive power warning
            stats = get_building_stats(building_type)
            if stats and env._last_obs:
                power_drain = stats.get("power", 0)
                if power_drain < 0:
                    eco = env._last_obs.get("economy", {})
                    current_balance = eco.get("power_provided", 0) - eco.get("power_drained", 0)
                    if current_balance + power_drain < 0:
                        result["warning"] = (
                            f"POWER WARNING: {building_type} drains {abs(power_drain)} power. "
                            f"Current balance: {current_balance:+d}. "
                            f"Build powr first or production will slow to 1/3 speed!"
                        )
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
                    p["queue_type"] in env._PLACEABLE_QUEUE_TYPES and p["item"] == building_type and p["progress"] >= 0.99
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
        def guard_target(unit_ids: str, target_actor_id: int, queued: bool = False) -> dict:
            """Order units to guard another actor, following and protecting it.
            unit_ids: comma-separated actor IDs (e.g. "145,146"), "all_combat", "all_idle", or a group name."""
            env._refresh_obs()
            resolved = env._resolve_unit_ids(unit_ids, env._last_obs or {})
            if not resolved:
                return {"error": "No matching units found"}
            commands = [
                CommandModel(action=ActionType.GUARD, actor_id=uid, target_actor_id=target_actor_id, queued=queued)
                for uid in resolved
            ]
            result = env._execute_commands(commands)
            return env._add_unit_feedback(result, resolved)

        @mcp.tool()
        def set_stance(unit_ids: str, stance: str) -> dict:
            """Set combat stance for units.
            Stances: 'hold_fire' (0), 'return_fire' (1), 'defend' (2), 'attack_anything' (3).
            unit_ids: comma-separated actor IDs (e.g. "145,146"), "all_combat", "all_idle", or a group name."""
            env._refresh_obs()
            resolved = env._resolve_unit_ids(unit_ids, env._last_obs or {})
            if not resolved:
                return {"error": "No matching units found"}
            stance_map = {"hold_fire": 0, "return_fire": 1, "defend": 2, "attack_anything": 3}
            stance_val = stance_map.get(stance.lower(), 3)
            commands = [
                CommandModel(action=ActionType.SET_STANCE, actor_id=uid, target_x=stance_val)
                for uid in resolved
            ]
            result = env._execute_commands(commands)
            return env._add_unit_feedback(result, resolved)

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

        @mcp.tool()
        def get_valid_placements(building_type: str, max_results: int = 8) -> dict:
            """Get suggested placement positions for a building near your base.
            Returns positions sorted by distance from Construction Yard.
            Use the first position with place_building(). If it fails, try the next."""
            env._refresh_obs()
            obs = env._last_obs
            if obs is None:
                return {"error": "No observation available"}

            candidates = env._find_placement_candidates(building_type, obs)
            if not candidates:
                return {"error": "No Construction Yard found — deploy MCV first"}

            bw, bh = env._FOOTPRINTS.get(building_type, (2, 2))
            # Find CY position for response
            cy_pos = {"cell_x": 0, "cell_y": 0}
            for b in obs.get("buildings", []):
                if b["type"] == "fact":
                    cy_pos = {"cell_x": b["cell_x"], "cell_y": b["cell_y"]}
                    break

            suggestions = candidates[:max(1, min(max_results, 15))]

            return {
                "building_type": building_type,
                "size": f"{bw}x{bh}",
                "cy_position": cy_pos,
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
            return env._add_unit_feedback(result, ids)

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
            result = env._execute_commands(commands)
            if not result.get("done"):
                # Game may need a tick to process surrender
                try:
                    adv_result = advance(ticks=50)
                    if adv_result.get("done"):
                        return adv_result
                except Exception:
                    pass
            return result

        @mcp.tool()
        def batch(actions: list[dict]) -> dict:
            """Send multiple commands that all execute concurrently (same game tick).

            Actions use same format as individual tools:
              {"tool": "build_unit", "unit_type": "e1", "count": 3}
              {"tool": "attack_move", "unit_ids": "155,160", "target_x": 50, "target_y": 30}
              {"tool": "set_stance", "unit_ids": "all_combat", "stance": "attack_anything"}
              {"tool": "deploy_unit", "unit_id": 120}

            Unit selectors for unit_ids:
              "all_combat" — all own combat units
              "all_idle" — all idle combat units
              "155,160" — comma-separated actor IDs
              group name — a previously assigned group

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

            # Tools that cannot be batched (flow-control or read-only)
            _BATCH_UNSUPPORTED = {
                "advance", "get_game_state", "get_units", "get_buildings",
                "get_terrain_at", "get_map_analysis", "get_valid_placements",
                "lookup_unit", "lookup_building", "get_replay",
                "surrender", "plan", "batch",
            }

            all_commands = []
            action_names = []
            for action in actions:
                tool = action.get("tool", "?")
                if tool in _BATCH_UNSUPPORTED:
                    action_names.append(f"{tool}:SKIPPED (use standalone)")
                    continue
                cmds = env._action_to_commands(action, obs)
                if cmds:
                    all_commands.extend(cmds)
                    action_names.append(tool)
                else:
                    action_names.append(f"{tool}:FAILED")

            if not all_commands:
                return {"error": "No valid commands generated", "actions": action_names}

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
                p["queue_type"] in self._PLACEABLE_QUEUE_TYPES and p["progress"] >= 0.99
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
        """Resolve unit selectors to actual actor IDs.

        Accepts:
          - list of ints: [145, 146] — validated against living units
          - "all_combat": all units that can attack
          - "all_idle": idle units that can attack
          - group name: units in a named group
          - comma-separated IDs: "145,146,148"
          - stringified list: "[145, 146]"
        """
        living_ids = {u["actor_id"] for u in obs.get("units", [])}

        if isinstance(selector, list):
            return self._filter_living(selector, living_ids)
        if not isinstance(selector, str):
            return []
        selector = selector.strip()
        if selector == "all_combat":
            return [u["actor_id"] for u in obs.get("units", []) if u.get("can_attack")]
        if selector == "all_idle":
            return [u["actor_id"] for u in obs.get("units", [])
                    if u.get("can_attack") and u.get("is_idle")]
        # Check named groups
        if selector in self._unit_groups:
            group_ids = list(self._unit_groups[selector])
            return self._filter_living(group_ids, living_ids)
        # Parse string-encoded lists: "[145, 146]" or "145,146,148"
        cleaned = selector.strip("[] ")
        if cleaned:
            try:
                parsed = [int(x.strip()) for x in cleaned.split(",") if x.strip()]
                return self._filter_living(parsed, living_ids)
            except ValueError:
                pass
        return []

    def _filter_living(self, unit_ids: list[int], living_ids: set[int]) -> list[int]:
        """Filter unit IDs to only those that are alive, warn about dead ones."""
        alive = [uid for uid in unit_ids if uid in living_ids]
        dead = [uid for uid in unit_ids if uid not in living_ids]
        if dead:
            self._placement_results.append(
                f"DEAD UNITS: IDs {dead} not found — units were destroyed or invalid"
            )
        return alive

    def _add_unit_feedback(self, result: dict, commanded_ids: list[int]) -> dict:
        """Append commanded_units feedback to a tool result.

        Looks up the commanded unit IDs in the latest observation and adds
        their current position and activity so the agent can verify commands
        were received and see where units are.
        """
        if not self._last_obs or not commanded_ids:
            return result
        units_by_id = {u["actor_id"]: u for u in self._last_obs.get("units", [])}
        result["commanded_units"] = [
            {
                "id": uid,
                "type": units_by_id[uid]["type"],
                "cell_x": units_by_id[uid]["cell_x"],
                "cell_y": units_by_id[uid]["cell_y"],
                "activity": units_by_id[uid].get("current_activity", "Unknown"),
            }
            for uid in commanded_ids
            if uid in units_by_id
        ]
        return result

    def _diagnose_unavailable(self, item_type: str) -> dict:
        """Diagnose why a unit/building is unavailable for production.

        Returns a dict with 'reason' and optionally 'missing_prerequisites'.
        """
        stats = get_unit_stats(item_type) or get_building_stats(item_type)
        if not stats:
            return {"reason": f"'{item_type}' is not a known unit or building type."}

        # Buildings require a Construction Yard to produce
        if get_building_stats(item_type) and self._last_obs:
            owned_types = {b["type"] for b in self._last_obs.get("buildings", [])}
            if "fact" not in owned_types:
                return {"reason": "No Construction Yard (fact) — deploy an MCV or you cannot build structures."}

        prereqs = stats.get("prerequisites", [])
        if not prereqs:
            return {"reason": f"'{item_type}' is not available. Check your faction."}

        # Check which prerequisite buildings we're missing
        owned_types = set()
        if self._last_obs:
            owned_types = {b["type"] for b in self._last_obs.get("buildings", [])}

        missing = []
        for prereq in prereqs:
            # Handle "barr|tent" style alternatives
            alternatives = prereq.split("|")
            if not any(alt in owned_types for alt in alternatives):
                missing.append(prereq)

        if missing:
            missing_str = ", ".join(missing)
            return {
                "reason": f"'{item_type}' unavailable — requires {missing_str} which you don't have.",
                "missing_prerequisites": missing,
            }

        # Prereqs are met but still unavailable — likely faction mismatch
        side = stats.get("side", "")
        if side:
            return {"reason": f"'{item_type}' is not available for your faction (it's {side}-only)."}

        return {"reason": f"'{item_type}' is not available. Check your faction and tech tree."}

    def _action_to_commands(self, action: dict, obs: dict) -> list[CommandModel]:
        """Convert a plan action dict to a list of CommandModel objects."""
        tool = action.get("tool", "")
        unit_ids = self._resolve_unit_ids(action.get("unit_ids", []), obs)
        queued = action.get("queued", False)

        if tool == "build_unit":
            unit_type = action.get("unit_type", "")
            available = obs.get("available_production", [])
            if available and unit_type not in available:
                return []  # unavailable — batch() will mark as FAILED
            count = max(1, action.get("count", 1))
            return [CommandModel(action=ActionType.TRAIN, item_type=unit_type)
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

    # Queue types that produce placeable structures (Building + Defense)
    _PLACEABLE_QUEUE_TYPES = {"Building", "Defense"}

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
        # Defense buildings (most are 1x1 from ^Defense base)
        "gun": (1, 1), "ftur": (1, 1), "tsla": (1, 1),
        "agun": (1, 1), "pbox": (1, 1), "hbox": (1, 1), "gap": (1, 1),
    }

    def _find_placement_candidates(self, building_type: str, obs: dict) -> list[dict]:
        """Find valid placement positions for a building near the Construction Yard.

        Searches the full build radius around the CY, avoiding occupied cells.
        Returns candidates sorted by distance from CY (closest first).
        """
        buildings = obs.get("buildings", [])

        # Find Construction Yard
        cy = None
        for b in buildings:
            if b["type"] == "fact":
                cy = b
                break
        if cy is None:
            return []

        cx, cy_y = cy["cell_x"], cy["cell_y"]
        bw, bh = self._FOOTPRINTS.get(building_type, (2, 2))

        # Mark occupied cells from all existing buildings (with 1-cell padding)
        occupied = set()
        for b in buildings:
            fw, fh = self._FOOTPRINTS.get(b["type"], (2, 2))
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
        max_radius = 15  # Full RA build radius
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
        return candidates

    _MAX_PLACEMENT_ATTEMPTS = 20

    def _update_loss_tracking(self) -> None:
        """Compare current buildings/units against previous snapshot, emit loss alerts."""
        obs = self._last_obs
        if obs is None:
            return

        # Current state
        cur_buildings = {b["actor_id"]: b["type"] for b in obs.get("buildings", [])}
        cur_units = {u["actor_id"]: u["type"] for u in obs.get("units", [])}

        # Building losses
        if self._prev_buildings:
            for actor_id, btype in self._prev_buildings.items():
                if actor_id not in cur_buildings:
                    self._placement_results.append(f"DESTROYED: {btype}")

        # Unit losses (summarized by type)
        _DEPLOY_MAP = {"mcv": "fact"}  # unit type → building type it deploys into
        if self._prev_unit_ids:
            lost_ids = set(self._prev_unit_ids) - set(cur_units)
            # Filter out deployments (MCV → Construction Yard)
            new_btypes = set(cur_buildings.values()) - set((self._prev_buildings or {}).values())
            for uid in list(lost_ids):
                utype = self._prev_unit_ids[uid]
                if utype in _DEPLOY_MAP and _DEPLOY_MAP[utype] in new_btypes:
                    lost_ids.discard(uid)
            # Filter out husk decay (wreckage disappearing, not a real loss)
            lost_ids = {uid for uid in lost_ids
                        if not self._prev_unit_ids[uid].endswith(".husk")}
            if lost_ids:
                from collections import Counter
                lost_types = Counter(self._prev_unit_ids[uid] for uid in lost_ids)
                breakdown = ", ".join(f"{cnt}x {t}" for t, cnt in lost_types.most_common())
                self._placement_results.append(
                    f"UNITS LOST: {len(lost_ids)} destroyed ({breakdown})"
                )

        # Update snapshots
        self._prev_buildings = cur_buildings
        self._prev_unit_ids = cur_units

    def _process_pending_placements(self) -> None:
        """Auto-place buildings that finished construction via build_and_place.

        Uses smart placement: finds valid positions in the full build radius
        around the Construction Yard, tries them in order. Cancels after
        _MAX_PLACEMENT_ATTEMPTS failures to avoid blocking the queue.
        """
        if not self._last_obs:
            return
        production = self._last_obs.get("production", [])
        attempted = getattr(self, "_attempted_placements", {})

        # Phase 2: Check previously attempted placements for failure
        failed = []
        for btype, attempt_idx in list(attempted.items()):
            still_in_queue = any(
                p["queue_type"] in self._PLACEABLE_QUEUE_TYPES and p["item"] == btype and p["progress"] >= 0.99
                for p in production
            )
            if still_in_queue:
                # Building is still in queue → last placement failed, try next candidate
                candidates = self._find_placement_candidates(btype, self._last_obs)
                if attempt_idx < min(len(candidates), self._MAX_PLACEMENT_ATTEMPTS):
                    # Try the next candidate position
                    pos = candidates[attempt_idx]
                    try:
                        commands = [CommandModel(
                            action=ActionType.PLACE_BUILDING,
                            item_type=btype,
                            target_x=pos["cell_x"],
                            target_y=pos["cell_y"],
                        )]
                        self._execute_commands(commands)
                        attempted[btype] = attempt_idx + 1
                    except Exception:
                        attempted[btype] = attempt_idx + 1
                else:
                    # Exhausted all candidates — report failure and cancel
                    if btype in self._WATER_BUILDINGS:
                        reason = f"{btype} requires water tiles — must be placed on water, not land"
                    else:
                        reason = f"no valid position found (tried {attempt_idx} spots)"
                    self._placement_results.append(
                        f"PLACEMENT FAILED: {btype} — {reason}. Auto-cancelling."
                    )
                    try:
                        cancel_cmd = [CommandModel(action=ActionType.CANCEL_PRODUCTION, item_type=btype)]
                        self._execute_commands(cancel_cmd)
                    except Exception:
                        pass
                    failed.append(btype)
            else:
                # Building no longer in queue → placement succeeded
                self._placement_results.append(f"AUTO-PLACED: {btype}")
                failed.append(btype)

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
                p["queue_type"] in self._PLACEABLE_QUEUE_TYPES and p["item"] == btype and p["progress"] >= 0.99
                for p in production
            )
            if not ready:
                continue

            # Water buildings can't auto-place on land — warn and skip
            if btype in self._WATER_BUILDINGS:
                self._placement_results.append(
                    f"WATER BUILDING: {btype} must be placed on water. "
                    f"Call get_valid_placements(\"{btype}\") then place_building() manually, "
                    f"or cancel_production(\"{btype}\") to unblock the queue."
                )
                self._pending_placements.pop(btype, None)
                continue

            # Find best placement position using full CY radius search
            candidates = self._find_placement_candidates(btype, self._last_obs)

            # Use user-specified coords if provided and valid, otherwise use best candidate
            cx, cy = coords["cell_x"], coords["cell_y"]
            if cx == 0 and cy == 0 and candidates:
                cx, cy = candidates[0]["cell_x"], candidates[0]["cell_y"]

            try:
                commands = [CommandModel(
                    action=ActionType.PLACE_BUILDING,
                    item_type=btype,
                    target_x=cx,
                    target_y=cy,
                )]
                self._execute_commands(commands)
                attempted[btype] = 1  # start tracking from candidate index 1
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

        # Track losses and trigger auto-placement
        self._update_loss_tracking()
        self._process_pending_placements()

        return {
            "tick": obs_dict["tick"],
            "done": obs_dict["done"],
            "result": obs_dict.get("result", ""),
            "economy": obs_dict["economy"],
            "own_units": len(obs_dict["units"]),
            "own_buildings": len(obs_dict["buildings"]),
            "visible_enemies": len(obs_dict["visible_enemies"]),
            "production": [f"{p['item']}@{p['progress']:.0%}" for p in obs_dict["production"]],
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
        self._planning_active = False
        self._planning_start_time = 0.0
        self._planning_turns_used = 0
        self._planning_strategy = ""
        self._enemy_ever_seen = False
        self._prev_buildings = {}
        self._prev_unit_ids = {}

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
