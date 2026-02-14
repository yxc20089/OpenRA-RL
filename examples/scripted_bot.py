#!/usr/bin/env python3
"""Scripted Red Alert bot that plays a full game via the OpenEnv client API.

Follows a classic build order: deploy MCV, build power + barracks,
train infantry, then attack-move toward the enemy.

Usage:
    # Against Docker container:
    docker run -p 8000:8000 openra-rl
    python examples/scripted_bot.py

    # Against local server:
    python examples/scripted_bot.py --url http://localhost:8000

    # With verbose logging:
    python examples/scripted_bot.py --verbose
"""

import argparse
import asyncio
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


class ScriptedBot:
    """State-machine bot with a fixed Red Alert build order.

    Phases:
        deploy_mcv  - Find MCV, issue DEPLOY command
        build_base  - Build power plant, barracks, war factory
        train_army  - Train rifle infantry up to target count
        attack      - Send idle combat units toward enemy
    """

    # Buildings to construct in order: (type, queue_type)
    BUILD_ORDER = [
        ("powr", "Building"),   # Power Plant ($300)
        ("tent", "Building"),   # Barracks ($300)
        ("weap", "Building"),   # War Factory ($2000)
    ]

    TRAIN_TARGET = 5  # infantry to train before first attack wave
    COMBAT_UNIT_TYPES = {"e1", "e2", "e3", "1tnk", "2tnk", "3tnk", "arty", "jeep"}

    def __init__(self, verbose: bool = False):
        self.phase = "deploy_mcv"
        self.build_index = 0  # index into BUILD_ORDER
        self.placement_count = 0
        self.deploy_issued = False
        self.verbose = verbose
        self._last_log_tick = -1

    def decide(self, obs: OpenRAObservation) -> OpenRAAction:
        """Given current observation, return commands for this tick."""
        commands: List[CommandModel] = []

        # Phase transitions
        self._update_phase(obs)

        # Priority 1: Place any completed buildings
        commands.extend(self._handle_placement(obs))

        # Priority 2: Deploy MCV if needed
        if self.phase == "deploy_mcv":
            cmd = self._handle_deploy(obs)
            if cmd:
                commands.append(cmd)

        # Priority 3: Queue building/unit production
        commands.extend(self._handle_production(obs))

        # Priority 4: Send idle combat units to attack
        commands.extend(self._handle_combat(obs))

        if not commands:
            commands.append(CommandModel(action=ActionType.NO_OP))

        return OpenRAAction(commands=commands)

    def _update_phase(self, obs: OpenRAObservation):
        """Transition between phases based on game state."""
        has_cy = any(b.type == "fact" for b in obs.buildings)
        has_barracks = any(b.type == "tent" for b in obs.buildings)
        combat_units = [u for u in obs.units if u.type in self.COMBAT_UNIT_TYPES]

        if self.phase == "deploy_mcv" and has_cy:
            self.phase = "build_base"
            self._log("Phase: build_base (Construction Yard ready)")
        elif self.phase == "build_base" and self.build_index >= len(self.BUILD_ORDER):
            self.phase = "train_army"
            self._log("Phase: train_army (all buildings queued)")
        elif self.phase == "train_army" and has_barracks and len(combat_units) >= self.TRAIN_TARGET:
            self.phase = "attack"
            self._log(f"Phase: attack ({len(combat_units)} combat units ready)")

    def _handle_deploy(self, obs: OpenRAObservation) -> Optional[CommandModel]:
        """Deploy the MCV to create a Construction Yard."""
        if self.deploy_issued:
            return None
        mcv = next((u for u in obs.units if u.type == "mcv"), None)
        if mcv:
            self.deploy_issued = True
            self._log(f"Deploying MCV (actor {mcv.actor_id})")
            return CommandModel(action=ActionType.DEPLOY, actor_id=mcv.actor_id)
        return None

    def _handle_placement(self, obs: OpenRAObservation) -> List[CommandModel]:
        """Place any completed buildings waiting in the production queue."""
        commands = []
        cy = self._find_construction_yard(obs)
        if not cy:
            return commands

        for prod in obs.production:
            if prod.queue_type == "Building" and prod.progress >= 0.99:
                x, y = self._placement_offset(cy)
                self._log(f"Placing {prod.item} at ({x}, {y})")
                commands.append(CommandModel(
                    action=ActionType.PLACE_BUILDING,
                    item_type=prod.item,
                    target_x=x,
                    target_y=y,
                ))
                self.placement_count += 1
        return commands

    def _handle_production(self, obs: OpenRAObservation) -> List[CommandModel]:
        """Queue building construction or unit training."""
        commands = []

        # Check if anything is already being built
        building_in_progress = any(
            p.queue_type == "Building" and p.progress < 0.99
            for p in obs.production
        )

        # Build next building from build order
        if not building_in_progress and self.build_index < len(self.BUILD_ORDER):
            item_type, _ = self.BUILD_ORDER[self.build_index]
            # Check if we already have this building
            if any(b.type == item_type for b in obs.buildings):
                self.build_index += 1
            elif item_type in obs.available_production:
                self._log(f"Building {item_type} (#{self.build_index + 1}/{len(self.BUILD_ORDER)})")
                commands.append(CommandModel(
                    action=ActionType.BUILD,
                    item_type=item_type,
                ))
                self.build_index += 1

        # Train infantry if barracks exists
        has_barracks = any(b.type == "tent" for b in obs.buildings)
        infantry_training = any(
            p.queue_type == "Infantry" and p.progress < 0.99
            for p in obs.production
        )
        combat_units = [u for u in obs.units if u.type in self.COMBAT_UNIT_TYPES]

        if has_barracks and not infantry_training and len(combat_units) < self.TRAIN_TARGET:
            if "e1" in obs.available_production and obs.economy.cash >= 100:
                self._log(f"Training e1 ({len(combat_units)}/{self.TRAIN_TARGET})")
                commands.append(CommandModel(
                    action=ActionType.TRAIN,
                    item_type="e1",
                ))

        return commands

    def _handle_combat(self, obs: OpenRAObservation) -> List[CommandModel]:
        """Send idle combat units to attack the enemy."""
        commands = []
        idle_fighters = [
            u for u in obs.units
            if u.type in self.COMBAT_UNIT_TYPES and u.is_idle
        ]

        if len(idle_fighters) < 3:
            return commands

        # Find attack target: nearest visible enemy, or map center
        target_x, target_y = self._find_attack_target(obs)

        for unit in idle_fighters:
            commands.append(CommandModel(
                action=ActionType.ATTACK_MOVE,
                actor_id=unit.actor_id,
                target_x=target_x,
                target_y=target_y,
            ))

        if commands:
            self._log(f"Attacking with {len(idle_fighters)} units toward ({target_x}, {target_y})")
        return commands

    def _find_construction_yard(self, obs: OpenRAObservation) -> Optional[BuildingInfoModel]:
        """Find the Construction Yard building."""
        return next((b for b in obs.buildings if b.type == "fact"), None)

    def _placement_offset(self, cy: BuildingInfoModel) -> Tuple[int, int]:
        """Calculate where to place the next building relative to CY.

        Uses cell coordinates. Buildings are ~2-3 cells wide, so offset by 3.
        CY pos_x/pos_y are world coords; we need cell coords.
        World coords in OpenRA are 1024 units per cell.
        """
        cx = cy.pos_x // 1024
        cy_y = cy.pos_y // 1024
        offsets = [(3, 0), (6, 0), (0, 3), (3, 3), (-3, 0), (0, -3)]
        idx = self.placement_count % len(offsets)
        dx, dy = offsets[idx]
        return cx + dx, cy_y + dy

    def _find_attack_target(self, obs: OpenRAObservation) -> Tuple[int, int]:
        """Find the best position to attack-move toward."""
        if obs.visible_enemies:
            # Attack nearest visible enemy
            enemy = obs.visible_enemies[0]
            return enemy.cell_x, enemy.cell_y

        # Fallback: move toward map center (likely toward enemy base)
        if obs.map_info.width > 0:
            return obs.map_info.width // 2, obs.map_info.height // 2
        return 64, 64  # reasonable default

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Bot] {msg}")


def print_status(step: int, obs: OpenRAObservation, bot: ScriptedBot):
    """Print a status line for the current step."""
    combat = len([u for u in obs.units if u.type in bot.COMBAT_UNIT_TYPES])
    buildings = ", ".join(sorted(set(b.type for b in obs.buildings))) or "none"
    print(
        f"Step {step:4d} | Tick {obs.tick:5d} | "
        f"${obs.economy.cash:5d} | "
        f"Units: {len(obs.units)} (combat: {combat}) | "
        f"Buildings: [{buildings}] | "
        f"Phase: {bot.phase}"
    )


async def run_bot(url: str, max_steps: int, verbose: bool):
    """Connect to the OpenRA-RL server and play one full game."""
    print(f"Connecting to {url}...")
    bot = ScriptedBot(verbose=verbose)

    async with OpenRAEnv(base_url=url, message_timeout_s=300.0) as env:
        print("Resetting environment...")
        result = await env.reset()
        print(f"Game started! Map: {result.observation.map_info.map_name}")
        print_status(0, result.observation, bot)

        step = 0
        total_reward = 0.0

        while not result.done and step < max_steps:
            action = bot.decide(result.observation)
            result = await env.step(action)
            step += 1
            total_reward += result.reward or 0.0

            if step % 100 == 0:
                print_status(step, result.observation, bot)

        # Final status
        print()
        print("=" * 60)
        obs = result.observation
        if obs.done:
            print(f"Game over: {obs.result} after {step} steps (tick {obs.tick})")
        else:
            print(f"Reached max steps ({max_steps}) at tick {obs.tick}")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Final economy: ${obs.economy.cash}")
        print(f"Units killed: {obs.military.units_killed}")
        print(f"Units lost: {obs.military.units_lost}")
        print(f"Buildings: {len(obs.buildings)}")
        print("=" * 60)


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
        help="Print detailed bot decisions each tick",
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
