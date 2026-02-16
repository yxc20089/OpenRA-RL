#!/usr/bin/env python3
"""LLM agent that plays Red Alert using OpenRouter models via MCP tools.

The agent connects to the OpenRA-RL server, discovers MCP tools, converts
them to OpenAI function calling format, and uses an LLM (via OpenRouter)
to decide which tools to call each turn.

Supports any OpenAI-compatible model via OpenRouter:
  - anthropic/claude-sonnet-4-20250514 (default, good at tool use)
  - openai/gpt-4o
  - google/gemini-2.0-flash-001
  - meta-llama/llama-3.1-70b-instruct

Usage:
    # Start the game server
    docker run -p 8000:8000 openra-rl

    # Run the agent
    export OPENROUTER_API_KEY=sk-or-...
    python examples/llm_agent.py --verbose

    # Or with a specific model
    python examples/llm_agent.py --model openai/gpt-4o --verbose
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Any, Optional

import httpx
from openra_env.mcp_ws_client import OpenRAMCPClient

# Line-buffered stdout so output is observable in real time
sys.stdout.reconfigure(line_buffering=True)

logger = logging.getLogger("llm_agent")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514"

SYSTEM_PROMPT = """\
You are an RTS AI playing Command & Conquer: Red Alert (Soviet faction). \
You control one side against a Normal AI opponent. The game runs in REAL \
TIME (~25 ticks/sec). Every second you spend thinking, the enemy is building.

## How to Act — Think in Plans

Use execute_plan() to submit multi-step temporal plans. Each step can have \
a time delay (wait_ticks) and conditions. The game executes your plan over \
many ticks automatically, then returns results.

CRITICAL RULES:
- Do NOT call get_game_state/get_economy/get_production/get_buildings — \
  a TURN BRIEFING with full state, unit IDs, and alerts is injected \
  automatically before each turn
- Use execute_plan() for multi-step sequences instead of individual tool calls
- React to ALERTS immediately (especially UNDER ATTACK)
- Use special selectors: "all_combat", "all_idle" instead of listing IDs
- You can still use individual tools when appropriate (single quick action)

## Game Mechanics You Should Leverage
- Production queues: build_unit queues training — it continues while you plan
- Auto-attack: set stance to "attack_anything" so units fight on their own
- Auto-pathfinding: one attack_move order sends units across the entire map
- Economy: game rejects orders you can't afford — just queue what you want
- Unit groups: assign_group("army", [id1,id2,...]) then use "army" as unit selector
  in execute_plan or command_group("army", "attack_move", target_x=50, target_y=30)

## Phase 1: Deploy & Build (turns 1-3)

WORKFLOW — Deploy MCV and build:
1. deploy_unit(<mcv_id>), advance(50) to get CY
2. build_structure("powr"), advance(120) to build it
3. place_building("powr") — auto-finds best position near your base
4. Repeat for barr, proc, weap

You can combine steps in execute_plan:
execute_plan([
  {"actions": [{"tool": "deploy_unit", "unit_id": <mcv_id>}], "wait_ticks": 50},
  {"actions": [{"tool": "build_structure", "building_type": "powr"}], "wait_ticks": 120},
  {"condition": "building_ready",
   "actions": [{"tool": "place_building", "building_type": "powr"}]},
  {"actions": [
    {"tool": "build_structure", "building_type": "barr"},
    {"tool": "build_unit", "unit_type": "e1", "count": 2},
    {"tool": "set_stance", "unit_ids": "all_combat", "stance": "attack_anything"}
  ], "wait_ticks": 120, "abort_on": "under_attack"}
])

Build order: powr → barr → proc → weap → 2nd powr if low power
PLACEMENT: Just call place_building("building_type") — coordinates are optional.
The engine auto-finds the best valid position near your base.

## Phase 2: Economy & Military (turns 3-8)

- Always queue units: build_unit("e1", count=3) or build_unit("1tnk")
- Build extra harvester when cash > $2500: build_unit("harv")
- Set ALL combat units to "attack_anything" stance
- Send 1 scout (e1) to map center, then opposite corner

EXAMPLE — Produce + scout:
execute_plan([
  {"actions": [
    {"tool": "build_unit", "unit_type": "e1", "count": 3},
    {"tool": "set_stance", "unit_ids": "all_combat", "stance": "attack_anything"}
  ], "wait_ticks": 50},
  {"actions": [{"tool": "attack_move", "unit_ids": [<scout_id>],
                "target_x": <map_w/2>, "target_y": <map_h/2>}],
   "wait_ticks": 200},
  {"condition": "enemies_visible",
   "actions": [{"tool": "attack_move", "unit_ids": "all_combat",
                "target_x": <map_w/2>, "target_y": <map_h/2>}]}
])

## Phase 3: Attack (5+ combat units)

EXAMPLE — All-in push:
execute_plan([
  {"actions": [
    {"tool": "set_stance", "unit_ids": "all_combat", "stance": "attack_anything"},
    {"tool": "attack_move", "unit_ids": "all_combat",
     "target_x": <enemy_x>, "target_y": <enemy_y>}
  ], "wait_ticks": 200, "abort_on": "under_attack"},
  {"actions": [
    {"tool": "build_unit", "unit_type": "1tnk", "count": 2},
    {"tool": "build_unit", "unit_type": "e1", "count": 3}
  ]}
])

## Phase 4: Defense (UNDER ATTACK alert)

execute_plan([
  {"actions": [
    {"tool": "set_stance", "unit_ids": "all_combat", "stance": "attack_anything"},
    {"tool": "attack_move", "unit_ids": "all_idle",
     "target_x": <attack_x>, "target_y": <attack_y>}
  ]},
  {"actions": [
    {"tool": "build_unit", "unit_type": "e1", "count": 3},
    {"tool": "repair_building", "building_id": <damaged_id>}
  ]}
])

## Economy Rules
- IDLE CASH (> $2000): build_unit("harv") or build_structure("proc")
- LOW POWER: build_structure("powr") immediately
- IDLE PRODUCTION: always queue something — idle production loses games
"""


def format_state_briefing(state: dict) -> str:
    """Format game state (from get_game_state tool) into a compact turn briefing with IDs."""
    if not isinstance(state, dict) or "tick" not in state:
        return ""

    eco = state.get("economy", {})
    tick = state["tick"]

    parts = [
        f"--- TURN BRIEFING (tick {tick}) ---",
        f"Cash: ${eco.get('cash', 0)} | Power: {state.get('power_balance', 0):+d} | Harvesters: {eco.get('harvester_count', 0)}",
    ]

    # Compact unit summary grouped by type, with IDs
    units = state.get("units_summary", [])
    if units:
        by_type = defaultdict(list)
        idle_ids = []
        for u in units:
            by_type[u["type"]].append(u["id"])
            if u.get("idle") and u.get("can_attack"):
                idle_ids.append(u["id"])
        unit_parts = []
        for utype, ids in by_type.items():
            id_str = ",".join(str(i) for i in ids)
            unit_parts.append(f"{len(ids)}x{utype}[{id_str}]")
        line = f"Units: {' '.join(unit_parts)}"
        if idle_ids:
            line += f" | Idle: [{','.join(str(i) for i in idle_ids)}]"
        parts.append(line)
    else:
        parts.append(f"Units: {state.get('own_units', '?')}")

    # Compact building summary with IDs
    buildings = state.get("buildings_summary", [])
    if buildings:
        bldg_parts = [f"{b['type']}({b['id']})" for b in buildings]
        parts.append(f"Buildings: {' '.join(bldg_parts)}")
    else:
        parts.append(f"Buildings: {state.get('own_buildings', '?')} ({', '.join(state.get('building_types', []))})")

    # Enemy summary with IDs and positions
    enemies = state.get("enemy_summary", [])
    if enemies:
        from collections import defaultdict
        eby_type = defaultdict(list)
        for e in enemies:
            eby_type[e["type"]].append(e)
        enemy_parts = []
        for etype, es in eby_type.items():
            id_str = ",".join(str(e["id"]) for e in es)
            enemy_parts.append(f"{len(es)}x{etype}[{id_str}]")
        # Average position
        avg_x = sum(e["cell_x"] for e in enemies) // len(enemies)
        avg_y = sum(e["cell_y"] for e in enemies) // len(enemies)
        parts.append(f"Enemies: {' '.join(enemy_parts)} near ({avg_x},{avg_y})")
    else:
        n_enemy = state.get("visible_enemy_units", 0)
        parts.append(f"Enemies: {'none visible' if n_enemy == 0 else f'{n_enemy} visible'}")

    prod = state.get("production_items", [])
    parts.append(f"Production: {', '.join(prod) if prod else 'IDLE'}")

    alerts = state.get("alerts", [])
    if alerts:
        parts.append("ALERTS:")
        for a in alerts:
            parts.append(f"  ** {a}")

    parts.append("---")
    parts.append("Use execute_plan() for multi-step actions. Do NOT call get_game_state.")

    if state.get("done"):
        parts.append(f"GAME OVER: {state.get('result', '?')}")

    return "\n".join(parts)


def mcp_tools_to_openai(tools: list) -> list[dict]:
    """Convert MCP Tool schemas to OpenAI function calling format."""
    result = []
    for tool in tools:
        schema = tool.input_schema if hasattr(tool, 'input_schema') else {}
        # Clean up schema — remove 'title' which confuses some models
        params = dict(schema) if schema else {}
        params.pop("title", None)
        if "properties" not in params:
            params["properties"] = {}
            params["type"] = "object"

        result.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": params,
            },
        })
    return result


async def chat_completion(
    messages: list[dict],
    tools: list[dict],
    api_key: str,
    model: str,
    verbose: bool = False,
) -> dict:
    """Call OpenRouter chat completions API."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }
    # Support provider routing (e.g., Cerebras for fast inference)
    provider = os.environ.get("OPENROUTER_PROVIDER", "")
    if provider:
        payload["provider"] = {"only": [provider]}

    async with httpx.AsyncClient() as client:
        if verbose:
            n_msgs = len(messages)
            print(f"  [LLM] Sending {n_msgs} messages to {model}...")

        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/openra-rl",
                "X-Title": "OpenRA-RL Agent",
            },
            json=payload,
            timeout=120.0,
        )

        if response.status_code != 200:
            error_text = response.text[:500]
            raise RuntimeError(f"OpenRouter API error {response.status_code}: {error_text}")

        data = response.json()

        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {data['error']}")

        if verbose:
            usage = data.get("usage", {})
            print(
                f"  [LLM] Response: {usage.get('prompt_tokens', '?')} prompt + "
                f"{usage.get('completion_tokens', '?')} completion tokens"
            )

        return data


def compress_history(messages: list[dict], keep_last: int = 40) -> list[dict]:
    """Compress message history to stay within context limits.

    Keeps the system prompt and the last N messages, replacing
    earlier messages with a state-aware summary that preserves
    critical game context (buildings, economy, errors).

    Ensures tool call/result pairs are never split — the kept
    window always starts with a user or assistant message (not tool).
    """
    if len(messages) <= keep_last + 1:
        return messages

    system = messages[0]
    # Find a clean cut point: recent must not start with tool role
    cut = len(messages) - keep_last
    while cut < len(messages) and messages[cut].get("role") == "tool":
        cut += 1  # move cut forward to skip orphaned tool results
    if cut >= len(messages) - 2:
        return messages  # can't compress safely

    old_messages = messages[1:cut]
    recent = messages[cut:]

    # Extract game state context from old messages
    last_state = {}
    building_types = set()
    errors = []

    for msg in old_messages:
        if msg.get("role") != "tool":
            continue
        try:
            content = json.loads(msg["content"]) if isinstance(msg["content"], str) else msg["content"]
            if not isinstance(content, dict):
                continue

            # Track latest state snapshot
            if "tick" in content and "economy" in content:
                last_state = content

            # Track buildings built
            for bt in content.get("building_types", []):
                building_types.add(bt)

            # Track placement failures and errors
            if content.get("placement_failed"):
                errors.append("placement failed")
            elif "error" in content and isinstance(content["error"], str):
                err = content["error"]
                if len(err) < 80:
                    errors.append(err)
        except (json.JSONDecodeError, TypeError):
            pass

    # Build summary
    parts = [f"[History: {len(old_messages)} earlier messages removed]"]

    if last_state:
        eco = last_state.get("economy", {})
        parts.append(
            f"Last state at tick {last_state.get('tick', '?')}: "
            f"${eco.get('cash', '?')} cash, "
            f"{last_state.get('own_units', '?')} units, "
            f"{last_state.get('own_buildings', '?')} buildings"
        )

    if building_types:
        parts.append(f"Buildings built so far: {', '.join(sorted(building_types))}")

    if errors:
        unique = list(dict.fromkeys(errors))[-3:]
        parts.append(f"Recent issues: {'; '.join(unique)}")

    parts.append("Continue from current state. Check TURN BRIEFING for latest info.")

    return [
        system,
        {"role": "user", "content": "\n".join(parts)},
        *recent,
    ]


async def run_agent(
    url: str,
    api_key: str,
    model: str,
    max_turns: int,
    verbose: bool,
):
    """Connect to OpenRA-RL and play a game using an LLM agent."""
    print(f"Connecting to {url}...")
    print(f"Model: {model}")

    async with OpenRAMCPClient(base_url=url, message_timeout_s=300.0) as env:
        print("Resetting environment (launching OpenRA)...")
        await env.reset()

        # Discover and convert tools
        mcp_tools = await env.list_tools()
        openai_tools = mcp_tools_to_openai(mcp_tools)
        print(f"Discovered {len(mcp_tools)} MCP tools")

        if verbose:
            for t in mcp_tools:
                print(f"  - {t.name}: {t.description[:60]}...")

        # Initialize conversation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Get initial state
        state = await env.call_tool("get_game_state")
        messages.append({
            "role": "user",
            "content": (
                f"Game started! Here's the initial state:\n"
                f"```json\n{json.dumps(state, indent=2)}\n```\n\n"
                f"Deploy your MCV and start building. What's your first move?"
            ),
        })

        total_tool_calls = 0
        total_api_calls = 0
        start_time = time.time()
        game_done = False
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 3

        for turn in range(max_turns):
            # Compress history periodically
            messages = compress_history(messages, keep_last=40)

            # Inject state briefing before LLM thinks (skip first turn — initial state already provided)
            if total_api_calls > 0:
                try:
                    briefing_state = await env.call_tool("get_game_state")
                    briefing = format_state_briefing(briefing_state)
                    if briefing:
                        messages.append({"role": "user", "content": briefing})
                        if verbose:
                            # Print just the alerts
                            for a in briefing_state.get("alerts", []):
                                print(f"  [ALERT] {a}")
                    # Check game over from briefing
                    if isinstance(briefing_state, dict) and briefing_state.get("done"):
                        game_done = True
                        print(f"\n  GAME OVER: {briefing_state.get('result', '?').upper()} at tick {briefing_state.get('tick', '?')}")
                        break
                except Exception:
                    pass

            # Call LLM
            try:
                response = await chat_completion(messages, openai_tools, api_key, model, verbose)
            except RuntimeError as e:
                print(f"\n  [ERROR] API call failed: {e}")
                # Wait and retry once
                await asyncio.sleep(5)
                try:
                    response = await chat_completion(messages, openai_tools, api_key, model, verbose)
                except RuntimeError as e2:
                    print(f"  [ERROR] Retry failed: {e2}")
                    break

            total_api_calls += 1
            choice = response["choices"][0]
            assistant_msg = choice["message"]

            # Add assistant response to history
            messages.append(assistant_msg)

            # Print assistant's reasoning
            if assistant_msg.get("content") and verbose:
                print(f"\n  [LLM thinks] {assistant_msg['content'][:200]}")

            # Handle tool calls
            tool_calls = assistant_msg.get("tool_calls", [])
            if not tool_calls:
                # No tool calls — prompt to act
                if verbose:
                    content = assistant_msg.get("content", "(no content)")
                    print(f"  [LLM] No tool calls. Response: {content[:100]}")
                messages.append({
                    "role": "user",
                    "content": "Please use the game tools to take action. Call get_game_state to see current state, or an action tool to do something.",
                })
                continue

            # Execute each tool call
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"].get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                total_tool_calls += 1

                if verbose:
                    args_str = json.dumps(fn_args)
                    if len(args_str) > 80:
                        args_str = args_str[:80] + "..."
                    print(f"  [Tool] {fn_name}({args_str})")

                try:
                    result = await env.call_tool(fn_name, **fn_args)
                    consecutive_errors = 0
                except Exception as e:
                    result = {"error": str(e)}

                # Detect game connection lost
                if isinstance(result, dict) and "connection lost" in str(result.get("error", "")).lower():
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        print(f"\n  GAME CRASHED: {consecutive_errors} consecutive connection errors. Stopping.")
                        game_done = True

                # Format result for message
                result_str = json.dumps(result) if not isinstance(result, str) else result

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

                # Check for game over
                if isinstance(result, dict) and result.get("done"):
                    game_done = True
                    print(f"\n  GAME OVER: {result.get('result', '?').upper()} at tick {result.get('tick', '?')}")

                if verbose and isinstance(result, dict):
                    result_preview = json.dumps(result)
                    if len(result_preview) > 120:
                        result_preview = result_preview[:120] + "..."
                    print(f"  [Result] {result_preview}")

            # Status update
            if total_api_calls % 5 == 0 or game_done:
                elapsed = time.time() - start_time
                print(
                    f"  Turn {turn + 1}/{max_turns} | "
                    f"API calls: {total_api_calls} | "
                    f"Tool calls: {total_tool_calls} | "
                    f"Time: {elapsed:.0f}s"
                )

            if game_done:
                break

            # Check finish reason
            if choice.get("finish_reason") == "stop" and not tool_calls:
                messages.append({
                    "role": "user",
                    "content": "Continue playing. Use game tools to check state and take actions.",
                })

        # Final report
        elapsed = time.time() - start_time
        print()
        print("=" * 70)
        print(f"Agent finished after {total_api_calls} API calls, {total_tool_calls} tool calls")
        print(f"Time: {elapsed:.1f}s ({elapsed / max(total_api_calls, 1):.1f}s per API call)")

        # Get final state
        try:
            final = await env.call_tool("get_game_state")
            print(f"Final tick: {final.get('tick', '?')}")
            print(f"Result: {final.get('result', 'ongoing')}")
            eco = final.get("economy", {})
            print(f"Cash: ${eco.get('cash', '?')}")
            print(f"Units: {final.get('own_units', '?')} own, {final.get('visible_enemies', '?')} enemy")
            print(f"Buildings: {final.get('own_buildings', '?')}")
        except Exception:
            pass

        # Get replay
        try:
            replay = await env.call_tool("get_replay_path")
            if replay.get("path"):
                print(f"Replay: {replay['path']}")
        except Exception:
            pass

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="LLM agent that plays Red Alert via OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --api-key sk-or-... --verbose\n"
            "  %(prog)s --model openai/gpt-4o --max-turns 100\n"
            "  OPENROUTER_API_KEY=sk-or-... %(prog)s\n"
        ),
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("OPENRA_URL", "http://localhost:8000"),
        help="OpenRA-RL server URL (default: $OPENRA_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
        help=f"OpenRouter model ID (default: $OPENROUTER_MODEL or {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (default: $OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=200,
        help="Maximum LLM turns (default: 200)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed LLM reasoning and tool calls",
    )
    parser.add_argument(
        "--log-file",
        default=os.environ.get("LLM_AGENT_LOG", ""),
        help="Write all output to this log file in addition to stdout (default: $LLM_AGENT_LOG)",
    )
    args = parser.parse_args()

    # Set up logging to file if requested — tee all print() to both stdout and file
    if args.log_file:
        import builtins
        _builtin_print = builtins.print
        _log_fh = open(args.log_file, "w")

        def _tee_print(*pargs, **kwargs):
            _builtin_print(*pargs, **kwargs)
            kwargs.pop("file", None)
            _builtin_print(*pargs, file=_log_fh, **kwargs)
            _log_fh.flush()

        builtins.print = _tee_print

    if not args.api_key:
        print("Error: OpenRouter API key required.")
        print("  Set OPENROUTER_API_KEY environment variable or use --api-key")
        print("  Get a key at: https://openrouter.ai/keys")
        sys.exit(1)

    try:
        asyncio.run(run_agent(args.url, args.api_key, args.model, args.max_turns, args.verbose))
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
