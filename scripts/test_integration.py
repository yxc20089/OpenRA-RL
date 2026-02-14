#!/usr/bin/env python3
"""End-to-end integration test for OpenRA-RL.

Tests the full reset → step × N → done cycle against a live OpenRA instance.

Prerequisites:
  - OpenRA built with ExternalBotBridge trait
  - OPENRA_PATH environment variable pointing to OpenRA installation
  - .NET runtime installed

Usage:
  $ python scripts/test_integration.py
  $ OPENRA_PATH=/path/to/openra python scripts/test_integration.py
  $ python scripts/test_integration.py --steps 50 --port 9999
"""

import argparse
import asyncio
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openra_env.models import ActionType, CommandModel, OpenRAAction, OpenRAObservation
from openra_env.reward import OpenRARewardFunction
from openra_env.server.bridge_client import BridgeClient, commands_to_proto, observation_to_dict
from openra_env.server.openra_process import OpenRAConfig, OpenRAProcessManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("integration-test")


async def test_bridge_connection(port: int) -> bool:
    """Test 1: Can we connect to the gRPC bridge?"""
    logger.info("--- Test 1: Bridge Connection ---")
    bridge = BridgeClient(port=port)
    try:
        ready = await bridge.wait_for_ready(max_retries=30, retry_interval=1.0)
        if ready:
            logger.info("PASS: Bridge connection established")
            state = await bridge.get_state()
            logger.info(f"  Game phase: {state.phase}, tick: {state.tick}")
            return True
        else:
            logger.error("FAIL: Bridge not ready after 30 attempts")
            return False
    finally:
        await bridge.close()


async def test_session_start(port: int) -> bool:
    """Test 2: Can we start a streaming session and get an initial observation?"""
    logger.info("--- Test 2: Session Start ---")
    bridge = BridgeClient(port=port)
    try:
        await bridge.connect()
        obs = await bridge.start_session()
        obs_dict = observation_to_dict(obs)

        logger.info(f"  Initial tick: {obs_dict['tick']}")
        logger.info(f"  Economy cash: {obs_dict['economy']['cash']}")
        logger.info(f"  Units: {len(obs_dict['units'])}")
        logger.info(f"  Buildings: {len(obs_dict['buildings'])}")
        logger.info(f"  Map: {obs_dict['map_info']['width']}x{obs_dict['map_info']['height']}")
        logger.info("PASS: Session started, initial observation received")
        return True
    except Exception as e:
        logger.error(f"FAIL: Session start failed: {e}")
        return False
    finally:
        await bridge.close()


async def test_step_cycle(port: int, num_steps: int) -> bool:
    """Test 3: Can we run a full step cycle (send actions, receive observations)?"""
    logger.info(f"--- Test 3: Step Cycle ({num_steps} steps) ---")
    bridge = BridgeClient(port=port)
    reward_fn = OpenRARewardFunction()

    try:
        await bridge.connect()
        obs = await bridge.start_session()
        obs_dict = observation_to_dict(obs)
        reward_fn.reset()

        total_reward = 0.0
        game_done = False

        for step in range(num_steps):
            # Build a simple action: no-op or move a random unit
            commands = []
            if obs_dict["units"]:
                # Move the first idle unit to a nearby cell
                for unit in obs_dict["units"]:
                    if unit["is_idle"]:
                        commands.append({
                            "action": "move",
                            "actor_id": unit["actor_id"],
                            "target_x": unit["cell_x"] + 1,
                            "target_y": unit["cell_y"],
                        })
                        break

            if not commands:
                commands.append({"action": "no_op"})

            proto_action = commands_to_proto(commands)
            obs = await bridge.step(proto_action)
            obs_dict = observation_to_dict(obs)

            reward = reward_fn.compute(obs_dict)
            total_reward += reward

            if step % 10 == 0 or obs_dict["done"]:
                logger.info(
                    f"  Step {step}: tick={obs_dict['tick']}, "
                    f"cash={obs_dict['economy']['cash']}, "
                    f"units={len(obs_dict['units'])}, "
                    f"enemies={len(obs_dict['visible_enemies'])}, "
                    f"reward={reward:.4f}"
                )

            if obs_dict["done"]:
                game_done = True
                logger.info(f"  Game ended: result={obs_dict['result']}")
                break

        logger.info(f"  Total reward after {step + 1} steps: {total_reward:.4f}")
        if game_done:
            logger.info("PASS: Full game episode completed")
        else:
            logger.info(f"PASS: {num_steps} steps executed successfully (game still running)")
        return True

    except Exception as e:
        logger.error(f"FAIL: Step cycle failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await bridge.close()


async def test_observation_model_parsing(port: int) -> bool:
    """Test 4: Can observation dicts be parsed into Pydantic models?"""
    logger.info("--- Test 4: Observation Model Parsing ---")
    bridge = BridgeClient(port=port)

    try:
        await bridge.connect()
        obs = await bridge.start_session()
        obs_dict = observation_to_dict(obs)

        from openra_env.server.openra_environment import OpenRAEnvironment

        env = OpenRAEnvironment.__new__(OpenRAEnvironment)
        env._reward_fn = OpenRARewardFunction()
        parsed = env._build_observation(obs_dict, 0.0)

        assert isinstance(parsed, OpenRAObservation)
        assert parsed.tick == obs_dict["tick"]
        assert parsed.economy.cash == obs_dict["economy"]["cash"]
        assert len(parsed.units) == len(obs_dict["units"])

        logger.info(f"  Parsed observation at tick {parsed.tick}")
        logger.info(f"  Economy: cash={parsed.economy.cash}, power={parsed.economy.power_provided}")
        logger.info(f"  Military: kills={parsed.military.units_killed}, losses={parsed.military.units_lost}")
        logger.info("PASS: Observation correctly parsed into Pydantic model")
        return True

    except Exception as e:
        logger.error(f"FAIL: Model parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await bridge.close()


def main():
    parser = argparse.ArgumentParser(description="OpenRA-RL Integration Test")
    parser.add_argument("--openra-path", default=os.environ.get("OPENRA_PATH", "/opt/openra"),
                        help="Path to OpenRA installation")
    parser.add_argument("--port", type=int, default=9999, help="gRPC port")
    parser.add_argument("--steps", type=int, default=30, help="Number of steps to run")
    parser.add_argument("--skip-launch", action="store_true",
                        help="Skip launching OpenRA (connect to existing instance)")
    parser.add_argument("--mod", default="ra", help="Game mod to use")
    parser.add_argument("--map", default="", help="Map to use")
    args = parser.parse_args()

    process = None
    results = {}

    try:
        # Launch OpenRA if not skipping
        if not args.skip_launch:
            logger.info("=== Launching OpenRA ===")
            config = OpenRAConfig(
                openra_path=args.openra_path,
                mod=args.mod,
                map_name=args.map,
                grpc_port=args.port,
            )
            process = OpenRAProcessManager(config)
            pid = process.launch()
            logger.info(f"OpenRA launched with PID {pid}")
            time.sleep(2)  # Brief wait for process startup
        else:
            logger.info("=== Skipping OpenRA launch (--skip-launch) ===")

        # Run tests
        loop = asyncio.new_event_loop()
        try:
            results["bridge_connection"] = loop.run_until_complete(
                test_bridge_connection(args.port)
            )

            if results["bridge_connection"]:
                results["session_start"] = loop.run_until_complete(
                    test_session_start(args.port)
                )

                results["observation_parsing"] = loop.run_until_complete(
                    test_observation_model_parsing(args.port)
                )

                results["step_cycle"] = loop.run_until_complete(
                    test_step_cycle(args.port, args.steps)
                )
        finally:
            loop.close()

    finally:
        # Clean up
        if process is not None:
            logger.info("=== Shutting down OpenRA ===")
            process.kill()

    # Summary
    print("\n" + "=" * 50)
    print("Integration Test Results")
    print("=" * 50)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if not results:
        print("  No tests were run!")
        all_passed = False

    print("=" * 50)
    if all_passed:
        print("All tests PASSED")
        sys.exit(0)
    else:
        print("Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
