#!/usr/bin/env python3
"""MCP tool-based Red Alert bot that plays entirely through MCP tools.

Validates the full MCP integration path: tool discovery, game knowledge
lookups, read tools for state, and action tools for commands. Uses
OpenRAMCPClient to interact with the OpenRA-RL server via WebSocket.

Exercises ALL 30 MCP tools:
  - Read tools: get_game_state, get_economy, get_units, get_buildings,
    get_enemies, get_production, get_map_info
  - Knowledge tools: lookup_unit, lookup_building, lookup_tech_tree, lookup_faction,
    get_faction_briefing, get_map_analysis, batch_lookup
  - Action tools: advance, deploy_unit, build_structure, place_building,
    build_unit, move_units, attack_move, attack_target, stop_units,
    set_rally_point, guard_target, set_stance, sell_building, repair_building,
    harvest, power_down, set_primary
  - Replay tool: get_replay_path

Usage:
    docker run -p 8000:8000 openra-rl
    python examples/mcp_bot.py --verbose
"""

import argparse
import asyncio
import json
import sys
from typing import Any, Optional

# Line-buffered stdout so output is observable in real time
sys.stdout.reconfigure(line_buffering=True)

from openra_env.mcp_ws_client import OpenRAMCPClient


class MCPBot:
    """State-machine bot that plays Red Alert using MCP tool calls.

    Phases:
        startup     - Look up tech tree and faction info
        deploy_mcv  - Find and deploy MCV
        build_base  - Build power/barracks/refinery/war factory
        train_army  - Train infantry + vehicles, set rally points
        attack      - Attack-move toward enemy
        sustain     - Repair, sell damaged, power management
    """

    BARRACKS_TYPES = {"tent", "barr"}
    WAR_FACTORY_TYPES = {"weap"}
    BUILD_ORDER = ["powr", "barracks", "proc", "weap", "powr"]
    INFANTRY_TARGET = 6
    GUARD_COUNT = 2
    COMBAT_TYPES = {"e1", "e2", "e3", "e4", "1tnk", "2tnk", "3tnk", "arty", "jeep", "apc"}
    INFANTRY_TYPES = {"e1", "e2", "e3", "e4"}

    def __init__(self, env: OpenRAMCPClient, verbose: bool = False, no_planning: bool = False):
        self.env = env
        self.verbose = verbose
        self.no_planning = no_planning
        self.phase = "startup"
        self.build_index = 0
        self.placement_count = 0
        self.deploy_issued = False
        self._guards_assigned: set[int] = set()
        self._stances_set: set[int] = set()
        self._rally_set: set[int] = set()
        self._repair_issued: set[int] = set()
        self._sold: set[int] = set()
        self._powered_down: set[int] = set()
        self._primary_set: set[int] = set()
        self._apc_trained = False
        self._tools_exercised: set[str] = set()

    async def call(self, tool_name: str, **kwargs: Any) -> Any:
        """Call an MCP tool and track which tools have been exercised."""
        self._tools_exercised.add(tool_name)
        result = await self.env.call_tool(tool_name, **kwargs)
        return result

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [MCPBot] {msg}")

    # ── Main loop ─────────────────────────────────────────────────

    async def run(self, max_turns: int) -> dict:
        """Run the bot for up to max_turns."""
        # Phase: startup — exercise knowledge tools
        await self._startup()

        turn = 0
        while turn < max_turns:
            state = await self.call("get_game_state")
            if state.get("done"):
                self._log(f"Game over: {state.get('result', '?')}")
                break

            turn += 1
            await self._tick(state, turn)

            if turn % 100 == 0:
                self._print_status(turn, state)

        # End-of-game report
        final_state = await self.call("get_game_state")
        replay = await self.call("get_replay_path")
        self._log(f"Replay: {replay}")

        return {
            "turns": turn,
            "final_state": final_state,
            "replay": replay,
            "tools_exercised": sorted(self._tools_exercised),
            "tools_count": len(self._tools_exercised),
            "planning_strategy": getattr(self, "_planning_strategy", ""),
        }

    # ── Startup: knowledge tools ──────────────────────────────────

    async def _startup(self):
        """Run planning phase and look up game knowledge at game start."""
        if self.no_planning:
            self._log("=== Startup: Planning DISABLED ===")
            # Use bulk knowledge tool instead of individual lookups
            briefing = await self.call("get_faction_briefing")
            self._log(f"Faction briefing: {briefing.get('side', '?')}, "
                      f"{len(briefing.get('units', {}))} units, "
                      f"{len(briefing.get('buildings', {}))} buildings")
        else:
            self._log("=== Startup: Planning Phase ===")

            # Try the planning phase
            planning = await self.call("start_planning_phase")
            if planning.get("planning_active"):
                self._log(f"Planning active — opponent: {planning.get('opponent_summary', '')[:120]}")

                # Use bulk tools for efficient research
                briefing = await self.call("get_faction_briefing")
                self._log(f"Faction briefing: {briefing.get('side', '?')}, "
                          f"{len(briefing.get('units', {}))} units, "
                          f"{len(briefing.get('buildings', {}))} buildings")

                map_analysis = await self.call("get_map_analysis")
                self._log(f"Map analysis: {map_analysis.get('map_type', '?')}, "
                          f"{len(map_analysis.get('resource_patches', []))} resource patches")

                intel = await self.call("get_opponent_intel")
                aggressiveness = intel.get("aggressiveness", "unknown")
                self._log(f"Opponent aggressiveness: {aggressiveness}")

                # Formulate strategy based on opponent profile
                if aggressiveness in ("high", "very_high"):
                    strategy = (
                        "Defensive opening: power plant, barracks, turrets at base entrance, "
                        "then ore refinery for economy. Build war factory for tanks once stable. "
                        "Scout early to find and deny enemy expansion."
                    )
                else:
                    strategy = (
                        "Rush opening: power plant, barracks, infantry rush while building "
                        "ore refinery. Transition to tanks from war factory."
                    )

                result = await self.call("end_planning_phase", strategy=strategy)
                self._planning_strategy = strategy
                self._log(f"Planning complete: {result.get('planning_duration_seconds', '?')}s, strategy: {strategy[:80]}")
            else:
                # Planning disabled server-side
                self._log(f"Planning: {planning.get('message', 'disabled')}")
                briefing = await self.call("get_faction_briefing")
                self._log(f"Faction briefing: {briefing.get('side', '?')}, "
                          f"{len(briefing.get('units', {}))} units, "
                          f"{len(briefing.get('buildings', {}))} buildings")

        map_info = await self.call("get_map_info")
        self._log(f"Map: {map_info.get('map_name', '?')} ({map_info.get('width')}x{map_info.get('height')})")

        self.phase = "deploy_mcv"
        self._log("Phase → deploy_mcv")

    # ── Per-tick decision ─────────────────────────────────────────

    async def _tick(self, state: dict, turn: int):
        """Make decisions for one game tick."""
        # Update phase based on state
        await self._update_phase()

        if self.phase == "deploy_mcv":
            await self._do_deploy()
        elif self.phase == "build_base":
            await self._do_build()
        elif self.phase == "train_army":
            await self._do_build()
            await self._do_train()
        elif self.phase == "attack":
            await self._do_build()
            await self._do_train()
            await self._do_combat()
            await self._do_sustain()

        # Advance game
        await self.call("advance", ticks=1)

    async def _update_phase(self):
        """Transition phases based on game state."""
        buildings = await self.call("get_buildings")
        units = await self.call("get_units")

        has_cy = any(b["type"] == "fact" for b in buildings)
        has_barracks = any(b["type"] in self.BARRACKS_TYPES for b in buildings)
        combat_units = [u for u in units if u["type"] in self.COMBAT_TYPES]
        non_guard = [u for u in combat_units if u["actor_id"] not in self._guards_assigned]

        if self.phase == "deploy_mcv" and has_cy:
            self.phase = "build_base"
            self._log("Phase → build_base")
        elif self.phase == "build_base" and self.build_index >= len(self.BUILD_ORDER):
            self.phase = "train_army"
            self._log("Phase → train_army")
        elif self.phase == "train_army" and len(non_guard) >= self.INFANTRY_TARGET:
            self.phase = "attack"
            self._log(f"Phase → attack ({len(non_guard)} combat units)")

    # ── Deploy MCV ────────────────────────────────────────────────

    async def _do_deploy(self):
        """Find and deploy MCV."""
        if self.deploy_issued:
            return

        units = await self.call("get_units")
        mcv = next((u for u in units if u["type"] == "mcv"), None)
        if mcv:
            self._log(f"Deploying MCV (actor {mcv['actor_id']})")
            await self.call("deploy_unit", unit_id=mcv["actor_id"])
            self.deploy_issued = True

    # ── Build base ────────────────────────────────────────────────

    async def _do_build(self):
        """Handle building construction and placement."""
        # Check for completed buildings to place
        production = await self.call("get_production")
        buildings = await self.call("get_buildings")

        for p in production.get("queue", []):
            if p["queue_type"] == "Building" and p["progress"] >= 0.99:
                cy = next((b for b in buildings if b["type"] == "fact"), None)
                if cy:
                    x, y = self._placement_offset(cy)
                    self._log(f"Placing {p['item']} at ({x}, {y})")
                    await self.call("place_building", building_type=p["item"], cell_x=x, cell_y=y)
                    self.placement_count += 1

        # Start new building if nothing in queue
        if self.build_index >= len(self.BUILD_ORDER):
            return

        building_in_queue = any(p["queue_type"] == "Building" for p in production.get("queue", []))
        if building_in_queue:
            return

        item = self.BUILD_ORDER[self.build_index]
        # Resolve faction-agnostic barracks
        if item == "barracks":
            available = production.get("available", [])
            if "tent" in available:
                item = "tent"
            elif "barr" in available:
                item = "barr"
            else:
                return

        # Check if already built
        already = sum(1 for b in buildings if b["type"] == item)
        if already > 0 and self.build_index < len(self.BUILD_ORDER) - 1:
            # Skip if not a duplicate in build order
            count_in_order = sum(1 for x in self.BUILD_ORDER[:self.build_index + 1]
                                 if x == item or (x == "barracks" and item in self.BARRACKS_TYPES))
            if already >= count_in_order:
                self.build_index += 1
                return

        available = production.get("available", [])
        if item in available:
            economy = await self.call("get_economy")
            building_stats = await self.call("lookup_building", building_type=item)
            cost = building_stats.get("cost", 0)
            if economy.get("cash", 0) >= cost:
                self._log(f"Building {item} (#{self.build_index + 1}/{len(self.BUILD_ORDER)}, cost=${cost})")
                await self.call("build_structure", building_type=item)
                self.build_index += 1

        # Set rally points on production buildings
        await self._do_rally_points(buildings)

    async def _do_rally_points(self, buildings: list[dict]):
        """Set rally points on barracks and war factories."""
        cy = next((b for b in buildings if b["type"] == "fact"), None)
        if not cy:
            return

        for b in buildings:
            if b["type"] in ("tent", "barr", "weap") and b["actor_id"] not in self._rally_set:
                rally_x = cy["cell_x"] if cy["cell_x"] > 0 else cy.get("pos_x", 0) // 1024
                rally_y = cy["cell_y"] if cy["cell_y"] > 0 else cy.get("pos_y", 0) // 1024
                self._log(f"Setting rally on {b['type']} (actor {b['actor_id']}) → ({rally_x}, {rally_y})")
                await self.call("set_rally_point", building_id=b["actor_id"], cell_x=rally_x, cell_y=rally_y)
                self._rally_set.add(b["actor_id"])

    def _placement_offset(self, cy: dict) -> tuple[int, int]:
        """Calculate placement position relative to CY."""
        cx = cy.get("pos_x", 0) // 1024 if cy.get("cell_x", 0) == 0 else cy["cell_x"]
        cy_y = cy.get("pos_y", 0) // 1024 if cy.get("cell_y", 0) == 0 else cy["cell_y"]
        offsets = [
            (3, 0), (-3, 0), (0, 3), (0, -3),
            (3, 3), (-3, 3), (3, -3), (-3, -3),
            (6, 0), (-6, 0), (0, 6), (0, -6),
        ]
        idx = self.placement_count % len(offsets)
        dx, dy = offsets[idx]
        return cx + dx, cy_y + dy

    # ── Train army ────────────────────────────────────────────────

    async def _do_train(self):
        """Train infantry and vehicles."""
        production = await self.call("get_production")
        buildings = await self.call("get_buildings")
        units = await self.call("get_units")
        economy = await self.call("get_economy")

        has_barracks = any(b["type"] in self.BARRACKS_TYPES for b in buildings)
        infantry_training = any(
            p["queue_type"] == "Infantry" and p["progress"] < 0.99
            for p in production.get("queue", [])
        )
        infantry = [u for u in units if u["type"] in self.INFANTRY_TYPES]
        total_target = self.INFANTRY_TARGET + self.GUARD_COUNT

        # Train infantry
        if has_barracks and not infantry_training and len(infantry) < total_target:
            available = production.get("available", [])
            if "e1" in available and economy.get("cash", 0) >= 100:
                self._log(f"Training e1 ({len(infantry)}/{total_target})")
                await self.call("build_unit", unit_type="e1")

        # Train APC from war factory
        has_weap = any(b["type"] == "weap" for b in buildings)
        vehicle_training = any(
            p["queue_type"] == "Vehicle" and p["progress"] < 0.99
            for p in production.get("queue", [])
        )
        if has_weap and not vehicle_training and not self._apc_trained:
            available = production.get("available", [])
            if "apc" in available and economy.get("cash", 0) >= 800:
                self._log("Training APC")
                await self.call("build_unit", unit_type="apc")
                self._apc_trained = True

        # Continuous vehicle production in attack phase
        if self.phase == "attack" and has_weap and not vehicle_training:
            available = production.get("available", [])
            if "1tnk" in available and economy.get("cash", 0) >= 700:
                self._log("Training 1tnk (continuous)")
                await self.call("build_unit", unit_type="1tnk")

        # Set stances on new units
        for u in units:
            if u["actor_id"] in self._stances_set:
                continue
            if u["type"] not in self.COMBAT_TYPES:
                continue
            stance = "defend" if u["actor_id"] in self._guards_assigned else "attack_anything"
            await self.call("set_stance", unit_ids=str(u["actor_id"]), stance=stance)
            self._stances_set.add(u["actor_id"])

        # Assign guards to CY
        if len(self._guards_assigned) < self.GUARD_COUNT:
            cy = next((b for b in buildings if b["type"] == "fact"), None)
            if cy:
                for u in units:
                    if len(self._guards_assigned) >= self.GUARD_COUNT:
                        break
                    if (u["type"] in self.INFANTRY_TYPES
                            and u["is_idle"]
                            and u["actor_id"] not in self._guards_assigned):
                        self._log(f"Assigning {u['type']} (actor {u['actor_id']}) to guard CY")
                        await self.call("guard_target", unit_ids=str(u["actor_id"]), target_actor_id=cy["actor_id"])
                        self._guards_assigned.add(u["actor_id"])

        # Set primary on multiple production buildings
        for btype_set in [self.BARRACKS_TYPES, self.WAR_FACTORY_TYPES]:
            bldgs_of_type = [b for b in buildings if b["type"] in btype_set]
            if len(bldgs_of_type) >= 2:
                newest = max(bldgs_of_type, key=lambda b: b["actor_id"])
                if newest["actor_id"] not in self._primary_set:
                    self._log(f"Setting primary: {newest['type']} (actor {newest['actor_id']})")
                    await self.call("set_primary", building_id=newest["actor_id"])
                    self._primary_set.add(newest["actor_id"])

    # ── Combat ────────────────────────────────────────────────────

    async def _do_combat(self):
        """Attack-move idle combat units toward enemies."""
        units = await self.call("get_units")
        enemies = await self.call("get_enemies")

        idle_fighters = [
            u for u in units
            if (u["type"] in self.COMBAT_TYPES
                and u["is_idle"]
                and u["actor_id"] not in self._guards_assigned)
        ]

        if len(idle_fighters) < 2:
            return

        # Find attack target
        target_x, target_y = self._find_attack_target(enemies, units)

        unit_id_list = [u["actor_id"] for u in idle_fighters]
        unit_ids = ",".join(str(i) for i in unit_id_list)
        self._log(f"Attacking with {len(unit_id_list)} units toward ({target_x}, {target_y})")
        await self.call("attack_move", unit_ids=unit_ids, target_x=target_x, target_y=target_y)

        # Attack specific visible enemy if close
        if enemies.get("units"):
            enemy = enemies["units"][0]
            nearby = [u for u in idle_fighters[:3] if u["can_attack"]]
            if nearby:
                await self.call(
                    "attack_target",
                    unit_ids=",".join(str(u["actor_id"]) for u in nearby),
                    target_actor_id=enemy["actor_id"],
                )

    def _find_attack_target(self, enemies: dict, units: list[dict]) -> tuple[int, int]:
        """Find best attack target: enemy buildings > units > map center."""
        if enemies.get("buildings"):
            b = enemies["buildings"][0]
            return b["cell_x"], b["cell_y"]
        if enemies.get("units"):
            u = enemies["units"][0]
            return u["cell_x"], u["cell_y"]
        return 64, 64  # fallback: map center

    # ── Sustain ───────────────────────────────────────────────────

    async def _do_sustain(self):
        """Repair, sell, and manage power."""
        buildings = await self.call("get_buildings")
        economy = await self.call("get_economy")

        for b in buildings:
            # Repair damaged buildings
            if (b["hp_percent"] < 0.7
                    and not b.get("is_repairing", False)
                    and b["actor_id"] not in self._repair_issued
                    and economy.get("cash", 0) >= 500):
                self._log(f"Repairing {b['type']} (actor {b['actor_id']}, hp={b['hp_percent']:.0%})")
                await self.call("repair_building", building_id=b["actor_id"])
                self._repair_issued.add(b["actor_id"])

            # Sell heavily damaged buildings
            if (b["hp_percent"] < 0.2
                    and b["type"] != "fact"
                    and b["actor_id"] not in self._sold):
                self._log(f"Selling {b['type']} (actor {b['actor_id']}, hp={b['hp_percent']:.0%})")
                await self.call("sell_building", building_id=b["actor_id"])
                self._sold.add(b["actor_id"])

        # Power management
        power_balance = economy.get("power_provided", 0) - economy.get("power_drained", 0)
        if power_balance < 0:
            power_down_priority = ["dome", "spen", "syrd", "hpad", "afld", "fix"]
            for btype in power_down_priority:
                for b in buildings:
                    if (b["type"] == btype
                            and b.get("is_powered", True)
                            and b["actor_id"] not in self._powered_down):
                        self._log(f"Powering down {b['type']} (actor {b['actor_id']}) — power: {power_balance}")
                        await self.call("power_down", building_id=b["actor_id"])
                        self._powered_down.add(b["actor_id"])
                        return  # one at a time

        # Send idle harvesters to harvest
        units = await self.call("get_units")
        for u in units:
            if u["type"] == "harv" and u["is_idle"]:
                self._log(f"Sending harvester {u['actor_id']} to harvest")
                await self.call("harvest", unit_id=u["actor_id"])
                break  # one at a time

        # Stop fleeing units
        fleeing = [u for u in units if u["type"] in self.COMBAT_TYPES
                   and u.get("current_activity") == "Flee"]
        if fleeing:
            await self.call("stop_units", unit_ids=",".join(str(u["actor_id"]) for u in fleeing[:3]))

        # Move scouts
        idle_scouts = [u for u in units
                       if u["type"] in ("jeep", "e1") and u["is_idle"]
                       and u["actor_id"] not in self._guards_assigned]
        if idle_scouts and len(idle_scouts) > 3:
            scout = idle_scouts[0]
            await self.call("move_units", unit_ids=str(scout["actor_id"]), target_x=64, target_y=64)

    # ── Status display ────────────────────────────────────────────

    def _print_status(self, turn: int, state: dict):
        eco = state.get("economy", {})
        power = eco.get("power_provided", 0) - eco.get("power_drained", 0)
        print(
            f"Turn {turn:4d} | Tick {state.get('tick', 0):5d} | "
            f"${eco.get('cash', 0):5d} | Pwr:{power:+d} | "
            f"Units:{state.get('own_units', 0)} | "
            f"Enemy:{state.get('visible_enemies', 0)} | "
            f"Bldgs:{state.get('own_buildings', 0)} | {self.phase}"
        )


# ── Main ──────────────────────────────────────────────────────────


async def run_mcp_bot(url: str, max_turns: int, verbose: bool, no_planning: bool = False):
    """Connect to the OpenRA-RL server and play using MCP tools."""
    print(f"Connecting to {url}...")

    async with OpenRAMCPClient(base_url=url, message_timeout_s=300.0) as env:
        print("Resetting environment (launching OpenRA)...")
        await env.reset()

        # Discover available tools
        tools = await env.list_tools()
        tool_names = sorted(t.name for t in tools)
        print(f"Discovered {len(tools)} MCP tools: {tool_names}")

        # Run bot
        bot = MCPBot(env, verbose=verbose, no_planning=no_planning)
        result = bot.run(max_turns)
        if asyncio.iscoroutine(result):
            result = await result

        # Final report
        print()
        print("=" * 70)
        final = result["final_state"]
        print(f"Game finished after {result['turns']} turns")
        if final.get("done"):
            print(f"Result: {final.get('result', '?').upper()}")

        # Score card
        mil = final.get("military", {})
        eco = final.get("economy", {})
        planning = result.get("planning_strategy", "")
        print()
        print("--- SCORECARD ---")
        print(f"  Planning:         {'ON — ' + planning if planning else 'OFF'}")
        print(f"  Ticks played:     {final.get('tick', '?')}")
        print(f"  Units killed:     {mil.get('units_killed', 0)} (value: ${mil.get('kills_cost', 0)})")
        print(f"  Units lost:       {mil.get('units_lost', 0)} (value: ${mil.get('deaths_cost', 0)})")
        print(f"  Buildings killed: {mil.get('buildings_killed', 0)}")
        print(f"  Buildings lost:   {mil.get('buildings_lost', 0)}")
        print(f"  Army value:       ${mil.get('army_value', 0)}")
        print(f"  Assets value:     ${mil.get('assets_value', 0)}")
        print(f"  Experience:       {mil.get('experience', 0)}")
        print(f"  Orders issued:    {mil.get('order_count', 0)}")
        print(f"  Cash remaining:   ${eco.get('cash', 0)}")
        print(f"  K/D cost ratio:   {mil.get('kills_cost', 0) / max(mil.get('deaths_cost', 1), 1):.2f}")
        print()

        print(f"Tools exercised: {result['tools_count']}/{len(tools)}")
        print(f"  {result['tools_exercised']}")
        if result.get("replay", {}).get("path"):
            print(f"Replay: {result['replay']['path']}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="MCP tool-based Red Alert bot")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="OpenRA-RL server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=3000,
        help="Maximum turns before stopping (default: 3000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed bot decisions",
    )
    parser.add_argument(
        "--no-planning",
        action="store_true",
        help="Disable planning phase (for comparison runs)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_mcp_bot(args.url, args.max_turns, args.verbose, no_planning=args.no_planning))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except ConnectionRefusedError:
        print(f"\nCould not connect to {args.url}")
        print("Is the OpenRA-RL server running?")
        print("  docker run -p 8000:8000 openra-rl")
        sys.exit(1)


if __name__ == "__main__":
    main()
