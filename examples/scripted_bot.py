#!/usr/bin/env python3
"""Scripted Red Alert bot that plays a full game via the OpenEnv client API.

Exercises ALL Sprint 4 observation fields and action types:
  - Observations: spatial_map, visible_enemy_buildings, unit facing/stance/speed/
    attack_range/experience/passengers, building cell coords/can_produce/power/
    rally/repair/sell_value
  - Actions: all 18 types including GUARD, SET_STANCE, ENTER_TRANSPORT, UNLOAD,
    SET_RALLY_POINT, REPAIR, SELL

Usage:
    docker run -p 8000:8000 openra-rl
    python examples/scripted_bot.py --verbose
"""

import argparse
import asyncio
import base64
import sys
from typing import List, Optional, Tuple

from openra_env.client import OpenRAEnv
from openra_env.models import (
    ActionType,
    BuildingInfoModel,
    CommandModel,
    OpenRAAction,
    OpenRAObservation,
    UnitInfoModel,
)

# Stance constants matching C# AutoTarget.UnitStance enum
STANCE_HOLD_FIRE = 0
STANCE_RETURN_FIRE = 1
STANCE_DEFEND = 2
STANCE_ATTACK_ANYTHING = 3

STANCE_NAMES = {0: "HoldFire", 1: "ReturnFire", 2: "Defend", 3: "AttackAnything"}


class ScriptedBot:
    """State-machine bot with a Red Alert build order exercising all actions.

    Phases:
        deploy_mcv   - Deploy MCV, set stance on starting units
        build_base   - Build power/barracks/war factory, set rally points
        train_army   - Train infantry + APC, guard CY, load transport
        attack       - Attack-move toward enemy buildings, unload APC
        sustain      - Continuous production, repair, sell damaged buildings
    """

    # Build order uses both faction names — bot picks whichever is available
    BARRACKS_TYPES = {"tent", "barr"}  # Allied / Soviet
    WAR_FACTORY_TYPES = {"weap"}
    BUILD_PRIORITY = [
        "powr",       # Power Plant ($300) — shared
        "barracks",   # Placeholder: tent (Allied) or barr (Soviet)
        "proc",       # Ore Refinery ($2000) — needed before war factory
        "weap",       # War Factory ($2000) — shared
        "powr",       # Second Power Plant
    ]

    INFANTRY_TRAIN_TARGET = 6
    GUARD_COUNT = 2  # infantry to guard CY
    TRANSPORT_TYPE = "apc"
    COMBAT_UNIT_TYPES = {"e1", "e2", "e3", "e4", "1tnk", "2tnk", "3tnk", "arty", "jeep", "apc"}
    INFANTRY_TYPES = {"e1", "e2", "e3", "e4"}
    VEHICLE_TYPES = {"1tnk", "2tnk", "3tnk", "arty", "jeep"}

    def __init__(self, verbose: bool = False):
        self.phase = "deploy_mcv"
        self.build_index = 0
        self.placement_count = 0
        self.deploy_issued = False
        self.verbose = verbose
        self._guards_assigned: set[int] = set()  # actor IDs guarding CY
        self._stances_set: set[int] = set()  # actor IDs with stance already set
        self._rally_set: set[int] = set()  # building actor IDs with rally point set
        self._apc_trained = False
        self._apc_loaded = False
        self._repair_issued: set[int] = set()  # building actor IDs being repaired
        self._sold: set[int] = set()  # building actor IDs sold

    def decide(self, obs: OpenRAObservation) -> OpenRAAction:
        """Given current observation, return commands for this tick."""
        commands: List[CommandModel] = []

        self._update_phase(obs)

        # Priority 1: Place completed buildings
        commands.extend(self._handle_placement(obs))

        # Priority 2: Deploy MCV
        if self.phase == "deploy_mcv":
            cmd = self._handle_deploy(obs)
            if cmd:
                commands.append(cmd)

        # Priority 3: Set rally points on production buildings
        commands.extend(self._handle_rally_points(obs))

        # Priority 4: Repair damaged buildings
        commands.extend(self._handle_repairs(obs))

        # Priority 5: Queue production (buildings + units)
        commands.extend(self._handle_production(obs))

        # Priority 6: Set stances on new units
        commands.extend(self._handle_stances(obs))

        # Priority 7: Assign guards to CY
        commands.extend(self._handle_guards(obs))

        # Priority 8: Load infantry into APC
        commands.extend(self._handle_transport(obs))

        # Priority 9: Combat — attack + unload
        commands.extend(self._handle_combat(obs))

        # Priority 10: Sell heavily damaged buildings
        commands.extend(self._handle_sell(obs))

        if not commands:
            commands.append(CommandModel(action=ActionType.NO_OP))

        return OpenRAAction(commands=commands)

    # ── Phase transitions ──────────────────────────────────────────

    def _update_phase(self, obs: OpenRAObservation):
        has_cy = any(b.type == "fact" for b in obs.buildings)
        has_barracks = any(b.type in self.BARRACKS_TYPES for b in obs.buildings)
        combat_units = [u for u in obs.units if u.type in self.COMBAT_UNIT_TYPES]
        non_guard_combat = [u for u in combat_units if u.actor_id not in self._guards_assigned]

        if self.phase == "deploy_mcv" and has_cy:
            self.phase = "build_base"
            self._log("Phase → build_base")
        elif self.phase == "build_base" and self.build_index >= len(self.BUILD_PRIORITY):
            self.phase = "train_army"
            self._log("Phase → train_army")
        elif self.phase == "train_army" and len(non_guard_combat) >= self.INFANTRY_TRAIN_TARGET:
            self.phase = "attack"
            self._log(f"Phase → attack ({len(non_guard_combat)} combat units ready)")
        elif self.phase == "attack" and has_barracks:
            # Stay in attack but also sustain production
            pass

    # ── Deploy MCV ─────────────────────────────────────────────────

    def _handle_deploy(self, obs: OpenRAObservation) -> Optional[CommandModel]:
        if self.deploy_issued:
            return None
        mcv = next((u for u in obs.units if u.type == "mcv"), None)
        if mcv:
            self.deploy_issued = True
            self._log(f"Deploying MCV (actor {mcv.actor_id}, facing={mcv.facing})")
            return CommandModel(action=ActionType.DEPLOY, actor_id=mcv.actor_id)
        return None

    # ── Building placement ─────────────────────────────────────────

    def _handle_placement(self, obs: OpenRAObservation) -> List[CommandModel]:
        commands = []
        cy = self._find_building(obs, "fact")
        if not cy:
            return commands

        for prod in obs.production:
            if prod.queue_type == "Building" and prod.progress >= 0.99:
                x, y = self._placement_offset(cy)
                self._log(f"Placing {prod.item} at cell ({x}, {y}) [attempt {self.placement_count}]")
                commands.append(CommandModel(
                    action=ActionType.PLACE_BUILDING,
                    item_type=prod.item,
                    target_x=x,
                    target_y=y,
                ))
                self.placement_count += 1
        return commands

    def _placement_offset(self, cy: BuildingInfoModel) -> Tuple[int, int]:
        """Calculate placement position relative to CY using cell coords."""
        # Use pos_x // 1024 as CenterPosition maps to cell more reliably
        cx = cy.pos_x // 1024
        cy_y = cy.pos_y // 1024
        # Many offsets to maximize chance of finding valid terrain
        offsets = [
            (3, 0), (-3, 0), (0, 3), (0, -3),
            (3, 3), (-3, 3), (3, -3), (-3, -3),
            (6, 0), (-6, 0), (0, 6), (0, -6),
            (2, 0), (-2, 0), (0, 2), (0, -2),
            (4, 0), (-4, 0), (0, 4), (0, -4),
        ]
        idx = self.placement_count % len(offsets)
        dx, dy = offsets[idx]
        return cx + dx, cy_y + dy

    # ── Rally points (Sprint 4 action) ─────────────────────────────

    def _handle_rally_points(self, obs: OpenRAObservation) -> List[CommandModel]:
        commands = []
        cy = self._find_building(obs, "fact")
        if not cy:
            return commands

        # Set rally point on barracks and war factory toward CY
        for b in obs.buildings:
            if b.type in ("tent", "weap") and b.actor_id not in self._rally_set:
                rally_x = cy.cell_x if cy.cell_x > 0 else cy.pos_x // 1024
                rally_y = cy.cell_y if cy.cell_y > 0 else cy.pos_y // 1024
                self._log(f"Setting rally on {b.type} (actor {b.actor_id}) → ({rally_x}, {rally_y})")
                commands.append(CommandModel(
                    action=ActionType.SET_RALLY_POINT,
                    actor_id=b.actor_id,
                    target_x=rally_x,
                    target_y=rally_y,
                ))
                self._rally_set.add(b.actor_id)
        return commands

    # ── Repair damaged buildings (Sprint 4 observation + existing action) ──

    def _handle_repairs(self, obs: OpenRAObservation) -> List[CommandModel]:
        commands = []
        for b in obs.buildings:
            if (b.hp_percent < 0.7
                    and not b.is_repairing
                    and b.actor_id not in self._repair_issued
                    and obs.economy.cash >= 500):
                self._log(f"Repairing {b.type} (actor {b.actor_id}, hp={b.hp_percent:.0%})")
                commands.append(CommandModel(
                    action=ActionType.REPAIR,
                    actor_id=b.actor_id,
                ))
                self._repair_issued.add(b.actor_id)
        return commands

    # ── Production ─────────────────────────────────────────────────

    def _handle_production(self, obs: OpenRAObservation) -> List[CommandModel]:
        commands = []

        # Building construction — treat any Building queue item as "in progress"
        # (includes completed-but-unplaced buildings that block the queue)
        building_in_queue = any(
            p.queue_type == "Building"
            for p in obs.production
        )
        if not building_in_queue and self.build_index < len(self.BUILD_PRIORITY):
            item_type = self._resolve_build_item(obs, self.BUILD_PRIORITY[self.build_index])
            if item_type is None:
                # Can't resolve this item yet, skip
                pass
            elif self._has_building_type(obs, item_type, self.build_index):
                self.build_index += 1
            elif self._can_produce_item(obs, item_type):
                self._log(f"Building {item_type} (#{self.build_index + 1}/{len(self.BUILD_PRIORITY)})")
                commands.append(CommandModel(action=ActionType.BUILD, item_type=item_type))
                self.build_index += 1

        # Infantry training
        has_barracks = any(b.type in self.BARRACKS_TYPES for b in obs.buildings)
        infantry_training = any(
            p.queue_type == "Infantry" and p.progress < 0.99
            for p in obs.production
        )
        infantry = [u for u in obs.units if u.type in self.INFANTRY_TYPES]
        total_target = self.INFANTRY_TRAIN_TARGET + self.GUARD_COUNT

        if has_barracks and not infantry_training and len(infantry) < total_target:
            if self._can_produce_item(obs, "e1") and obs.economy.cash >= 100:
                self._log(f"Training e1 ({len(infantry)}/{total_target})")
                commands.append(CommandModel(action=ActionType.TRAIN, item_type="e1"))

        # APC from war factory
        has_weap = any(b.type == "weap" for b in obs.buildings)
        vehicle_training = any(
            p.queue_type == "Vehicle" and p.progress < 0.99
            for p in obs.production
        )
        if (has_weap and not vehicle_training and not self._apc_trained
                and self._can_produce_item(obs, self.TRANSPORT_TYPE)
                and obs.economy.cash >= 800):
            self._log("Training APC for transport ops")
            commands.append(CommandModel(action=ActionType.TRAIN, item_type=self.TRANSPORT_TYPE))
            self._apc_trained = True

        # Continuous vehicle production in attack phase
        if (self.phase == "attack" and has_weap and not vehicle_training
                and obs.economy.cash >= 800):
            # Build light tanks if available
            if self._can_produce_item(obs, "1tnk"):
                self._log("Training 1tnk (continuous production)")
                commands.append(CommandModel(action=ActionType.TRAIN, item_type="1tnk"))

        return commands

    def _can_produce_item(self, obs: OpenRAObservation, item_type: str) -> bool:
        """Check if item is buildable using per-building can_produce (Sprint 4)."""
        # First check global available_production
        if item_type in obs.available_production:
            return True
        # Also check per-building can_produce lists
        for b in obs.buildings:
            if item_type in b.can_produce:
                return True
        return False

    # ── Stances (Sprint 4 action) ──────────────────────────────────

    def _handle_stances(self, obs: OpenRAObservation) -> List[CommandModel]:
        commands = []
        for u in obs.units:
            if u.actor_id in self._stances_set:
                continue
            if u.type not in self.COMBAT_UNIT_TYPES:
                continue

            # Guards get Defend stance, attackers get AttackAnything
            if u.actor_id in self._guards_assigned:
                desired = STANCE_DEFEND
            else:
                desired = STANCE_ATTACK_ANYTHING

            if u.stance != desired:
                self._log(
                    f"Setting {u.type} (actor {u.actor_id}) stance: "
                    f"{STANCE_NAMES.get(u.stance, '?')} → {STANCE_NAMES[desired]}"
                )
                commands.append(CommandModel(
                    action=ActionType.SET_STANCE,
                    actor_id=u.actor_id,
                    target_x=desired,
                ))
            self._stances_set.add(u.actor_id)
        return commands

    # ── Guard CY (Sprint 4 action) ────────────────────────────────

    def _handle_guards(self, obs: OpenRAObservation) -> List[CommandModel]:
        commands = []
        if len(self._guards_assigned) >= self.GUARD_COUNT:
            return commands

        cy = self._find_building(obs, "fact")
        if not cy:
            return commands

        # Find idle infantry not yet guarding
        for u in obs.units:
            if len(self._guards_assigned) >= self.GUARD_COUNT:
                break
            if (u.type in self.INFANTRY_TYPES
                    and u.is_idle
                    and u.actor_id not in self._guards_assigned):
                self._log(
                    f"Assigning {u.type} (actor {u.actor_id}, "
                    f"range={u.attack_range}) to guard CY"
                )
                commands.append(CommandModel(
                    action=ActionType.GUARD,
                    actor_id=u.actor_id,
                    target_actor_id=cy.actor_id,
                ))
                self._guards_assigned.add(u.actor_id)
        return commands

    # ── Transport: load/unload (Sprint 4 actions) ─────────────────

    def _handle_transport(self, obs: OpenRAObservation) -> List[CommandModel]:
        commands = []
        if self._apc_loaded:
            return commands

        apc = next(
            (u for u in obs.units
             if u.type == self.TRANSPORT_TYPE and u.passenger_count == 0),
            None,
        )
        if not apc:
            return commands

        # Load idle infantry (not guards) into the APC
        loaded = 0
        for u in obs.units:
            if loaded >= 4:  # APC capacity
                break
            if (u.type in self.INFANTRY_TYPES
                    and u.is_idle
                    and u.actor_id not in self._guards_assigned):
                self._log(
                    f"Loading {u.type} (actor {u.actor_id}, "
                    f"speed={u.speed}) into APC {apc.actor_id}"
                )
                commands.append(CommandModel(
                    action=ActionType.ENTER_TRANSPORT,
                    actor_id=u.actor_id,
                    target_actor_id=apc.actor_id,
                ))
                loaded += 1

        if loaded > 0:
            self._apc_loaded = True
        return commands

    # ── Combat ─────────────────────────────────────────────────────

    def _handle_combat(self, obs: OpenRAObservation) -> List[CommandModel]:
        commands = []
        if self.phase != "attack":
            return commands

        # Unload APC near enemy
        commands.extend(self._handle_unload(obs))

        # Attack-move idle fighters toward enemy
        idle_fighters = [
            u for u in obs.units
            if (u.type in self.COMBAT_UNIT_TYPES
                and u.is_idle
                and u.actor_id not in self._guards_assigned)
        ]

        if len(idle_fighters) < 2:
            return commands

        target_x, target_y = self._find_attack_target(obs)

        for unit in idle_fighters:
            commands.append(CommandModel(
                action=ActionType.ATTACK_MOVE,
                actor_id=unit.actor_id,
                target_x=target_x,
                target_y=target_y,
            ))

        if idle_fighters:
            self._log(
                f"Attacking with {len(idle_fighters)} units "
                f"toward ({target_x}, {target_y})"
            )
        return commands

    def _handle_unload(self, obs: OpenRAObservation) -> List[CommandModel]:
        """Unload APC when near enemies."""
        commands = []
        for u in obs.units:
            if u.type != self.TRANSPORT_TYPE or u.passenger_count <= 0:
                continue

            # Check if any enemy is within ~15 cells
            for enemy in obs.visible_enemies:
                dx = abs(u.cell_x - enemy.cell_x)
                dy = abs(u.cell_y - enemy.cell_y)
                if dx + dy < 15:
                    self._log(
                        f"Unloading APC (actor {u.actor_id}, "
                        f"{u.passenger_count} passengers) near enemy"
                    )
                    commands.append(CommandModel(
                        action=ActionType.UNLOAD,
                        actor_id=u.actor_id,
                    ))
                    break

            # Also unload near enemy buildings
            for eb in obs.visible_enemy_buildings:
                dx = abs(u.cell_x - eb.cell_x)
                dy = abs(u.cell_y - eb.cell_y)
                if dx + dy < 15:
                    self._log(
                        f"Unloading APC near enemy building {eb.type} "
                        f"(hp={eb.hp_percent:.0%})"
                    )
                    commands.append(CommandModel(
                        action=ActionType.UNLOAD,
                        actor_id=u.actor_id,
                    ))
                    break
        return commands

    def _find_attack_target(self, obs: OpenRAObservation) -> Tuple[int, int]:
        """Prioritize enemy buildings > enemy units > map center."""
        # Priority 1: visible enemy buildings (Sprint 4 field)
        if obs.visible_enemy_buildings:
            # Prefer production buildings
            prod_buildings = [
                b for b in obs.visible_enemy_buildings
                if b.type in ("fact", "tent", "weap", "hpad", "afld")
            ]
            target = prod_buildings[0] if prod_buildings else obs.visible_enemy_buildings[0]
            return target.cell_x, target.cell_y

        # Priority 2: visible enemy units
        if obs.visible_enemies:
            enemy = obs.visible_enemies[0]
            return enemy.cell_x, enemy.cell_y

        # Fallback: map center
        if obs.map_info.width > 0:
            return obs.map_info.width // 2, obs.map_info.height // 2
        return 64, 64

    # ── Sell heavily damaged buildings ─────────────────────────────

    def _handle_sell(self, obs: OpenRAObservation) -> List[CommandModel]:
        commands = []
        for b in obs.buildings:
            if (b.hp_percent < 0.2
                    and b.type != "fact"  # never sell CY
                    and b.actor_id not in self._sold):
                self._log(
                    f"Selling {b.type} (actor {b.actor_id}, hp={b.hp_percent:.0%}, "
                    f"refund=${b.sell_value})"
                )
                commands.append(CommandModel(
                    action=ActionType.SELL,
                    actor_id=b.actor_id,
                ))
                self._sold.add(b.actor_id)
        return commands

    # ── Helpers ────────────────────────────────────────────────────

    def _resolve_build_item(self, obs: OpenRAObservation, placeholder: str) -> Optional[str]:
        """Resolve faction-agnostic build item to actual producible type."""
        if placeholder == "barracks":
            # Find which barracks type is available
            for btype in self.BARRACKS_TYPES:
                if self._can_produce_item(obs, btype):
                    return btype
            return None
        return placeholder

    def _has_building_type(self, obs: OpenRAObservation, item_type: str, build_index: int) -> bool:
        """Check if we already have enough of this building type."""
        already_built = sum(1 for b in obs.buildings if b.type == item_type)
        # Count how many times this item appears up to current index
        resolved_order = []
        for i, p in enumerate(self.BUILD_PRIORITY[:build_index + 1]):
            if p == "barracks":
                resolved_order.append(item_type if item_type in self.BARRACKS_TYPES else p)
            else:
                resolved_order.append(p)
        target_count = resolved_order.count(item_type)
        return already_built >= target_count

    def _find_building(self, obs: OpenRAObservation, btype: str) -> Optional[BuildingInfoModel]:
        return next((b for b in obs.buildings if b.type == btype), None)

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Bot] {msg}")


# ── Status display ─────────────────────────────────────────────────


def print_status(step: int, obs: OpenRAObservation, bot: ScriptedBot):
    """Print a rich status line using Sprint 4 observation fields."""
    combat = [u for u in obs.units if u.type in bot.COMBAT_UNIT_TYPES]
    buildings = ", ".join(sorted(set(b.type for b in obs.buildings))) or "none"
    power_balance = obs.economy.power_provided - obs.economy.power_drained

    # Count enemy intel
    enemy_units = len(obs.visible_enemies)
    enemy_buildings = len(obs.visible_enemy_buildings)

    print(
        f"Step {step:4d} | Tick {obs.tick:5d} | "
        f"${obs.economy.cash:5d} | Pwr:{power_balance:+d} | "
        f"Units:{len(obs.units)} (combat:{len(combat)}) | "
        f"Enemy:{enemy_units}u/{enemy_buildings}b | "
        f"Bldgs:[{buildings}] | {bot.phase}"
    )


def print_detailed_status(obs: OpenRAObservation):
    """Print full observation details using all Sprint 4 fields."""
    print("\n── Detailed Observation ──")

    # Spatial map
    if obs.spatial_channels > 0 and obs.spatial_map:
        raw_bytes = base64.b64decode(obs.spatial_map)
        w, h = obs.map_info.width, obs.map_info.height
        expected_bytes = w * h * obs.spatial_channels * 4
        print(
            f"  Spatial: {w}x{h} map, {obs.spatial_channels} channels, "
            f"{len(raw_bytes)} bytes (expected {expected_bytes})"
        )
    else:
        print("  Spatial: not populated")

    # Economy
    e = obs.economy
    print(
        f"  Economy: ${e.cash} cash, {e.ore} ore, "
        f"power {e.power_provided}/{e.power_drained} "
        f"({e.power_provided - e.power_drained:+d}), "
        f"{e.harvester_count} harvesters"
    )

    # Production queue
    if obs.production:
        print(f"  Production queue ({len(obs.production)}):")
        for p in obs.production:
            print(f"    {p.queue_type}: {p.item} @ {p.progress:.0%} (paused={p.paused})")
    if obs.available_production:
        print(f"  Available production: {', '.join(obs.available_production[:15])}")
    else:
        print("  Available production: (none)")

    # Own buildings with Sprint 4 fields
    print(f"  Buildings ({len(obs.buildings)}):")
    for b in obs.buildings:
        extras = []
        if b.power_amount != 0:
            extras.append(f"pwr={b.power_amount:+d}")
        if b.is_producing:
            extras.append(f"producing={b.producing_item}@{b.production_progress:.0%}")
        if b.is_repairing:
            extras.append("REPAIRING")
        if b.rally_x >= 0:
            extras.append(f"rally=({b.rally_x},{b.rally_y})")
        if b.can_produce:
            extras.append(f"can_produce=[{','.join(b.can_produce[:5])}{'...' if len(b.can_produce) > 5 else ''}]")
        extra_str = f" ({', '.join(extras)})" if extras else ""
        print(
            f"    {b.type:6s} #{b.actor_id:4d} "
            f"cell=({b.cell_x},{b.cell_y}) "
            f"hp={b.hp_percent:.0%} "
            f"sell=${b.sell_value}{extra_str}"
        )

    # Own units with Sprint 4 fields
    print(f"  Units ({len(obs.units)}):")
    for u in obs.units[:10]:  # cap at 10 for readability
        stance_name = STANCE_NAMES.get(u.stance, f"?{u.stance}")
        extras = []
        if u.experience_level > 0:
            extras.append(f"vet={u.experience_level}")
        if u.passenger_count >= 0:
            extras.append(f"cargo={u.passenger_count}")
        extra_str = f" ({', '.join(extras)})" if extras else ""
        print(
            f"    {u.type:6s} #{u.actor_id:4d} "
            f"cell=({u.cell_x},{u.cell_y}) "
            f"hp={u.hp_percent:.0%} "
            f"face={u.facing:4d} spd={u.speed:3d} "
            f"rng={u.attack_range:5d} "
            f"stance={stance_name} "
            f"{'IDLE' if u.is_idle else u.current_activity}{extra_str}"
        )
    if len(obs.units) > 10:
        print(f"    ... and {len(obs.units) - 10} more")

    # Visible enemies
    if obs.visible_enemies:
        print(f"  Visible enemy units ({len(obs.visible_enemies)}):")
        for u in obs.visible_enemies[:5]:
            print(
                f"    {u.type:6s} #{u.actor_id:4d} "
                f"cell=({u.cell_x},{u.cell_y}) hp={u.hp_percent:.0%} "
                f"spd={u.speed} rng={u.attack_range}"
            )

    # Visible enemy buildings (Sprint 4 field)
    if obs.visible_enemy_buildings:
        print(f"  Visible enemy buildings ({len(obs.visible_enemy_buildings)}):")
        for b in obs.visible_enemy_buildings[:5]:
            print(
                f"    {b.type:6s} #{b.actor_id:4d} "
                f"cell=({b.cell_x},{b.cell_y}) hp={b.hp_percent:.0%} "
                f"pwr={b.power_amount:+d}"
            )


# ── Main loop ──────────────────────────────────────────────────────


async def run_bot(url: str, max_steps: int, verbose: bool):
    """Connect to the OpenRA-RL server and play one full game."""
    print(f"Connecting to {url}...")
    bot = ScriptedBot(verbose=verbose)

    async with OpenRAEnv(base_url=url, message_timeout_s=300.0) as env:
        print("Resetting environment...")
        result = await env.reset()
        obs = result.observation
        print(f"Game started! Map: {obs.map_info.map_name} ({obs.map_info.width}x{obs.map_info.height})")

        # Print initial detailed status
        if verbose:
            print_detailed_status(obs)

        print_status(0, obs, bot)

        step = 0
        total_reward = 0.0

        while not result.done and step < max_steps:
            action = bot.decide(result.observation)
            result = await env.step(action)
            step += 1
            total_reward += result.reward or 0.0
            obs = result.observation

            if step % 100 == 0:
                print_status(step, obs, bot)

            # Detailed dump at key milestones
            if verbose and step in (50, 200, 500, 1000):
                print_detailed_status(obs)

        # Final report
        print()
        print("=" * 70)
        obs = result.observation
        if obs.done:
            print(f"GAME OVER: {obs.result.upper()} after {step} steps (tick {obs.tick})")
        else:
            print(f"Reached max steps ({max_steps}) at tick {obs.tick}")

        print(f"Total reward:        {total_reward:.3f}")
        print(f"Final cash:          ${obs.economy.cash}")
        print(f"Power balance:       {obs.economy.power_provided - obs.economy.power_drained:+d}")
        print(f"Units killed:        {obs.military.units_killed}")
        print(f"Units lost:          {obs.military.units_lost}")
        print(f"Buildings killed:    {obs.military.buildings_killed}")
        print(f"Buildings lost:      {obs.military.buildings_lost}")
        print(f"Army value:          ${obs.military.army_value}")
        print(f"Own buildings:       {len(obs.buildings)}")
        print(f"Visible enemies:     {len(obs.visible_enemies)} units, {len(obs.visible_enemy_buildings)} buildings")

        # Spatial map stats
        if obs.spatial_channels > 0 and obs.spatial_map:
            raw_bytes = base64.b64decode(obs.spatial_map)
            n_floats = len(raw_bytes) // 4
            print(f"Spatial map:         {n_floats} floats ({obs.spatial_channels} channels)")
        else:
            print("Spatial map:         not populated")

        # Show veteran units
        vets = [u for u in obs.units if u.experience_level > 0]
        if vets:
            print(f"Veterans:            {', '.join(f'{u.type}#{u.actor_id}(lvl{u.experience_level})' for u in vets)}")

        if verbose:
            print_detailed_status(obs)

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Scripted Red Alert bot via OpenEnv")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="OpenRA-RL server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum steps before stopping (default: 5000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed bot decisions and observation dumps",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_bot(args.url, args.max_steps, args.verbose))
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
