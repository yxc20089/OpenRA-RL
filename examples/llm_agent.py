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
from typing import Any, Optional

import httpx
from openra_env.mcp_ws_client import OpenRAMCPClient

# Line-buffered stdout so output is observable in real time
sys.stdout.reconfigure(line_buffering=True)

logger = logging.getLogger("llm_agent")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514"

SYSTEM_PROMPT = """\
You are playing Command & Conquer: Red Alert via game tools. You control \
one faction against an AI opponent on a randomly generated map.

## Win Condition
Destroy all enemy buildings and units, or outlast the opponent.

## Build Order (follow this!)
1. Deploy MCV immediately (deploy_unit)
2. Build Power Plant (build_structure "powr"), then place it (place_building)
3. Build Barracks (build_structure "tent" or "barr"), place it
4. Build Ore Refinery (build_structure "proc"), place it
5. Build War Factory (build_structure "weap"), place it
6. Build a second Power Plant if power balance < 20

## Economy
- Keep cash flowing: protect your harvester
- Build extra refineries if you can afford them ($2000 each)
- Keep power balance positive (build power plants as needed)

## Military
- Train infantry from barracks: "e1" (Rifle, $100), "e3" (Rocket, $300)
- Train vehicles from war factory: "1tnk" (Light Tank, $700), "2tnk" (Medium Tank, $800)
- Set rally points on production buildings so units gather near your base
- Set stance to "attack_anything" on combat units

## Combat
- Scout first with cheap units (e1) to find the enemy base
- Attack with groups of 4+ units using attack_move
- Focus fire on enemy production buildings (fact, tent/barr, weap)

## Tools
- Use get_game_state to get an overview (economy, unit counts, enemy visibility)
- Use get_economy to check cash before building
- Use get_production to check what's being built and what's available
- Use lookup_tech_tree to plan your tech progression
- Use advance() between decisions to let the game progress (100+ ticks)
- Check production progress with get_production before placing buildings

## Important — Real-Time Game
- The game runs in REAL TIME at ~25 ticks/sec, regardless of your actions
- Time passes while you think! Check get_game_state to see the current tick
- Use advance(ticks=100) to wait for production/building to complete
- Place buildings at offsets from your Construction Yard (cell coordinates)
- Building placement: use cell_x, cell_y = (CY_cell_x + offset, CY_cell_y + offset)
  Offsets like (3,0), (-3,0), (0,3), (0,-3) work well
- Act quickly — every second of deliberation is a second the enemy is building
"""


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
    earlier messages with a summary.
    """
    if len(messages) <= keep_last + 1:
        return messages

    system = messages[0]
    old_messages = messages[1:-keep_last]
    recent = messages[-keep_last:]

    # Count tool calls and game events in old messages
    tool_calls = 0
    game_events = []
    for msg in old_messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_calls += len(msg["tool_calls"])
        if msg.get("role") == "tool":
            try:
                content = json.loads(msg["content"]) if isinstance(msg["content"], str) else msg["content"]
                if isinstance(content, dict):
                    if content.get("done"):
                        game_events.append(f"Game ended: {content.get('result', '?')}")
                    elif content.get("tick"):
                        game_events.append(f"Tick {content['tick']}")
            except (json.JSONDecodeError, TypeError):
                pass

    summary = (
        f"[History compressed: {len(old_messages)} messages, "
        f"{tool_calls} tool calls. "
    )
    if game_events:
        summary += f"Key events: {', '.join(game_events[-5:])}. "
    summary += "Continue playing from current state.]"

    return [
        system,
        {"role": "user", "content": summary},
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
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
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
