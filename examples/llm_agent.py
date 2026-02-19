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

from dotenv import load_dotenv
load_dotenv()
from collections import defaultdict
from typing import Any, Optional

import httpx
from openra_env.game_data import get_building_stats, get_faction_info, get_tech_tree, get_unit_stats
from openra_env.mcp_ws_client import OpenRAMCPClient

# Line-buffered stdout so output is observable in real time
sys.stdout.reconfigure(line_buffering=True)

logger = logging.getLogger("llm_agent")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-coder-next"

SYSTEM_PROMPT = """\
You are playing Command & Conquer: Red Alert as one faction against an AI opponent.

## How the Game Works
The game runs in real time at ~25 ticks/sec. You interact through tool calls. \
Between your turns, a TURN BRIEFING is injected showing current state: \
economy, units, buildings, enemies, production queue, and available builds.

A STRATEGIC BRIEFING at game start provides map size, base position, \
enemy spawn estimate, faction, tech tree, and unit/building stats.

## Available Tools

**advance(ticks)** — Let game time pass. Nothing happens without this. \
Production, movement, and combat all require game time to progress. \
250 ticks ≈ 10 seconds. Typical build times: power plant ~300 ticks, \
barracks ~500 ticks, war factory ~750 ticks.

**batch(actions)** — Send multiple commands simultaneously in one game tick.
Example: batch([
  {"tool": "build_unit", "unit_type": "e1", "count": 3},
  {"tool": "attack_move", "unit_ids": "all_combat", "target_x": 50, "target_y": 50}
])

**plan(steps)** — Execute steps sequentially with state refresh between each.
Example: plan([
  {"actions": [{"tool": "deploy_unit", "unit_id": 120}]},
  {"actions": [{"tool": "build_and_place", "building_type": "powr"}]}
])

**build_and_place(building_type)** — Queue a building and auto-place when done.

**build_unit(unit_type, count)** — Queue unit training. Requires the right \
production building (barracks for infantry, war factory for vehicles).

**attack_move(unit_ids, target_x, target_y)** — Move units, engaging enemies en route.

Unit selectors: comma-separated IDs (e.g. "145,146"), "all_combat", "all_idle", or a group name.

## Game Knowledge Tools
Use these to look up unit stats, building stats, and tech trees at any time:
- **get_faction_briefing()** — Get ALL units and buildings for your faction with full stats in one call. Best for planning.
- **get_map_analysis()** — Get strategic map summary: resource patches, water, terrain, quadrant breakdown.
- **batch_lookup(queries)** — Look up multiple items at once: [{"type":"unit","name":"3tnk"}, {"type":"building","name":"weap"}]
- **lookup_unit(unit_type)** — Get stats for a single unit (e.g. "3tnk", "e1")
- **lookup_building(building_type)** — Get stats for a single building (e.g. "weap", "proc")
- **lookup_tech_tree(faction)** — Get the full build order and tech tree for "allied" or "soviet"
- **lookup_faction(faction)** — Get all available units and buildings for a faction

## Game Mechanics

**Economy**: Funds = cash + ore. Harvesters collect ore from the map. \
Ore refineries (proc) come with one free harvester. More harvesters = faster income. \
CRITICAL: Construction costs are paid incrementally — if you hit $0, ALL production \
pauses until income resumes. Never let funds reach zero.

**Power**: Buildings require power. Power plants (powr) provide +100. \
When power demand exceeds supply, ALL production slows to 1/3 speed — \
this is devastating. Always build a power plant BEFORE any building \
that drains power. Check your power balance in every briefing.

**Production**: Buildings produce units — barracks/tent → infantry, \
war factory (weap) → vehicles. Multiple production buildings of the same \
type speed up production. Queue items and advance() to let them finish.

**Tech tree**: Higher-tier buildings unlock stronger units. \
A war factory requires an ore refinery. Tanks require a Repair Facility (fix) \
which requires a war factory. Build order for tanks: powr → barracks → proc → weap → fix → tanks. \
The "Can build:" line in each briefing shows what is currently available to produce.

**Unit strength**: Vehicles (tanks) are much stronger than infantry. \
e1 costs $100, a heavy tank (3tnk, Soviet) costs $950 but is far more powerful. \
Allied uses light tanks (1tnk, $600). Only build units listed in your available_production.

**Building placement**: build_and_place() handles placement automatically. \
Buildings that need water (spen, syrd) will fail on land maps.

**Defense**: Build turrets to protect your base. Gun turrets (gun/ftur) \
are cheap ground defense. SAM sites (sam/agun) defend against air. \
If auto-placement fails, use get_valid_placements(building_type) to find \
positions, then place_building(building_type, cell_x, cell_y) to place manually.

**Rally points**: Use set_rally_point(building_id, x, y) after building a barracks \
or war factory to auto-send new units to a staging area away from the building.

**Air power**: Build a radar dome (dome), then an airfield (afld) or helipad \
(hpad) to produce aircraft. MiGs and Hinds are powerful strike units. \
Air units bypass ground defenses and can hit the enemy base directly.

**Scouting**: Send a cheap unit (e1 or dog) to explore the map early. \
Knowing the enemy's base location and army composition is critical. \
Don't wait — scout within the first 500 ticks. Scout toward the opposite \
corner from your base — enemies usually spawn there.

**Expansion**: Build multiple ore refineries (proc) for faster income. \
Each comes with a free harvester. 2-3 refineries is ideal. Ore patches \
deplete over time, so expand to new areas. If ore storage is near capacity, \
build a silo ($150) to avoid wasting harvester income.

**Unit variety**: Don't just build one unit type. Mix tanks for armor, \
rocket soldiers (e3) for anti-air, and engineers (e6) to capture enemy buildings. \
Soviet players can build attack dogs (fast, cheap scouts) from a kennel.

## Strategy Priorities
1. Deploy MCV immediately, then power plant
2. Build barracks, then scout with a cheap unit toward the OPPOSITE corner
3. Build ore refinery for economy — NEVER let cash hit $0
4. Build second refinery BEFORE war factory (economy sustains everything)
5. War factory → Repair Facility (fix) → tanks
6. Build 1-2 defense turrets at your base entrance
7. Radar dome → airfield for air strikes (mid-game)
8. Attack when you have 4+ tanks — don't hoard units at base
8. Expand: build a second ore refinery when funds allow

## Pre-Game Planning Phase
If planning is enabled, you get a PLANNING PHASE before gameplay begins. \
During planning you receive map intel, faction info, and an opponent scouting \
report with their behavioral tendencies, win rate, and recommended counters. \
BE EFFICIENT with your planning turns — use bulk tools: \
1. Call get_faction_briefing() for ALL your units and buildings with full stats. \
2. Call get_map_analysis() for terrain, resource locations, and strategic notes. \
3. Review the opponent intel provided by start_planning_phase. \
4. Call end_planning_phase(strategy="your detailed strategy here") to begin gameplay. \
Do NOT look up units one at a time — use get_faction_briefing() or batch_lookup() instead. \
Planning has a turn limit — aim to finish in 3-4 turns max.

## Briefing Format
Each turn briefing includes:
- Funds, power balance, harvester count
- Your units with IDs and positions
- Your buildings with IDs and positions
- Visible enemies with IDs and positions
- Current production queue and available builds
- ALERTS for events needing attention (attacks, low power, idle production)
"""


def compose_pregame_briefing(state: dict) -> str:
    """Compose a strategic briefing from initial game state + static game data.

    Sent once at game start so the LLM knows map, base position, faction, tech tree,
    and available units/buildings without needing extra tool calls.
    """
    map_info = state.get("map", {})
    map_w = map_info.get("width", 0)
    map_h = map_info.get("height", 0)
    map_name = map_info.get("map_name", "?")

    # Determine base position from buildings/units
    buildings = state.get("buildings_summary", [])
    units = state.get("units_summary", [])
    all_positions = [(b["cell_x"], b["cell_y"]) for b in buildings] + \
                    [(u["cell_x"], u["cell_y"]) for u in units]
    if all_positions:
        base_x = sum(p[0] for p in all_positions) // len(all_positions)
        base_y = sum(p[1] for p in all_positions) // len(all_positions)
    else:
        base_x, base_y = map_w // 2, map_h // 2

    # Estimate enemy spawn — opposite side of map
    enemy_x = max(2, min(map_w - 2, map_w - base_x))
    enemy_y = max(2, min(map_h - 2, map_h - base_y))

    # Determine faction and side
    faction = state.get("faction", "")
    allied_factions = {"england", "france", "germany"}
    soviet_factions = {"russia", "ukraine"}
    if faction in allied_factions:
        side = "Allied"
        barracks = "tent"
    elif faction in soviet_factions:
        side = "Soviet"
        barracks = "barr"
    else:
        # Infer from available production or buildings
        avail = state.get("available_production", [])
        bldg_types = state.get("building_types", [])
        if "tent" in avail or "tent" in bldg_types:
            side, barracks = "Allied", "tent"
        else:
            side, barracks = "Soviet", "barr"

    # Get tech tree — returns {side: [order]} dict
    tech = get_tech_tree(side.lower())
    tech_order = tech.get(side.lower(), tech.get("build_order", []))

    # Get faction info for available units/buildings
    faction_info = get_faction_info(faction) if faction else get_faction_info(side.lower())
    avail_units = faction_info.get("available_units", []) if faction_info else []
    avail_buildings = faction_info.get("available_buildings", []) if faction_info else []

    # Format key units with costs
    unit_lines = []
    for utype in avail_units[:12]:  # Cap at 12 to keep concise
        stats = get_unit_stats(utype)
        if stats:
            unit_lines.append(f"  {utype}: {stats['name']} — ${stats['cost']}, {stats.get('category', '?')}")

    # Format key buildings with costs and power
    bldg_lines = []
    for btype in avail_buildings[:10]:
        stats = get_building_stats(btype)
        if stats:
            power = stats.get("power", 0)
            power_str = f", {power:+d} power" if power else ""
            bldg_lines.append(f"  {btype}: {stats['name']} — ${stats['cost']}{power_str}")

    parts = [
        f"## Strategic Briefing",
        f"Map: {map_name} ({map_w}x{map_h})",
        f"Your faction: {faction or side} ({side})",
        f"Your base: ({base_x}, {base_y})",
        f"Enemy likely near: ({enemy_x}, {enemy_y}) — SCOUT to confirm!",
        f"",
        f"Tech tree: {' → '.join(tech_order[:8])}{'...' if len(tech_order) > 8 else ''}",
        f"Barracks type: {barracks}",
        f"",
        f"Available units:",
        *unit_lines,
        f"",
        f"Available buildings:",
        *bldg_lines,
    ]
    return "\n".join(parts)


def format_state_briefing(state: dict) -> str:
    """Format game state (from get_game_state tool) into a compact turn briefing with positions."""
    if not isinstance(state, dict) or "tick" not in state:
        return ""

    eco = state.get("economy", {})
    tick = state["tick"]
    cash = eco.get("cash", 0)
    ore = eco.get("ore", 0)
    funds = cash + ore

    parts = [
        f"--- TURN BRIEFING (tick {tick}, ~{tick // 25}s game time) ---",
        f"Funds: ${funds} (cash=${cash} + ore=${ore}) | Power: {state.get('power_balance', 0):+d} | Harvesters: {eco.get('harvester_count', 0)}",
    ]

    # Base center from buildings
    buildings = state.get("buildings_summary", [])
    if buildings:
        base_x = sum(b["cell_x"] for b in buildings) // len(buildings)
        base_y = sum(b["cell_y"] for b in buildings) // len(buildings)
        parts.append(f"Base center: ({base_x},{base_y})")

    # Compact unit summary grouped by type, with IDs and positions
    units = state.get("units_summary", [])
    if units:
        by_type = defaultdict(list)
        idle_ids = []
        for u in units:
            by_type[u["type"]].append(u)
            if u.get("idle") and u.get("can_attack"):
                idle_ids.append(u["id"])
        unit_parts = []
        for utype, us in by_type.items():
            entries = ",".join(f"{u['id']}@({u['cell_x']},{u['cell_y']})" for u in us)
            unit_parts.append(f"{len(us)}x{utype}[{entries}]")
        line = f"Units: {' '.join(unit_parts)}"
        if idle_ids:
            line += f" | Idle: [{','.join(str(i) for i in idle_ids)}]"
        parts.append(line)
    else:
        parts.append(f"Units: {state.get('own_units', '?')}")

    # Compact building summary with IDs, positions, and production category
    _BLDG_CATEGORY = {"tent": "infantry", "barr": "infantry", "weap": "vehicle",
                       "hpad": "aircraft", "afld": "aircraft", "syrd": "ship", "spen": "ship",
                       "gun": "defense", "ftur": "defense", "tsla": "defense",
                       "sam": "defense", "agun": "defense", "pbox": "defense", "hbox": "defense"}
    if buildings:
        bldg_parts = []
        for b in buildings:
            cat = _BLDG_CATEGORY.get(b["type"], "")
            cat_str = f"[{cat}]" if cat else ""
            bldg_parts.append(f"{b['type']}({b['id']})@({b['cell_x']},{b['cell_y']}){cat_str}")
        parts.append(f"Buildings: {' '.join(bldg_parts)}")
    else:
        parts.append(f"Buildings: {state.get('own_buildings', '?')} ({', '.join(state.get('building_types', []))})")

    # Enemy summary with IDs and positions
    enemies = state.get("enemy_summary", [])
    if enemies:
        eby_type = defaultdict(list)
        for e in enemies:
            eby_type[e["type"]].append(e)
        enemy_parts = []
        for etype, es in eby_type.items():
            entries = ",".join(f"{e['id']}@({e['cell_x']},{e['cell_y']})" for e in es)
            enemy_parts.append(f"{len(es)}x{etype}[{entries}]")
        # Average position
        avg_x = sum(e["cell_x"] for e in enemies) // len(enemies)
        avg_y = sum(e["cell_y"] for e in enemies) // len(enemies)
        parts.append(f"Enemies: {' '.join(enemy_parts)} center ({avg_x},{avg_y})")
    else:
        n_enemy = state.get("visible_enemy_units", 0)
        parts.append(f"Enemies: {'none visible' if n_enemy == 0 else f'{n_enemy} visible'}")

    prod = state.get("production_items", [])
    parts.append(f"Production: {', '.join(prod) if prod else 'IDLE'}")

    available = state.get("available_production", [])
    if available:
        parts.append(f"Can build: {', '.join(available)}")

    alerts = state.get("alerts", [])
    if alerts:
        parts.append("ALERTS:")
        for a in alerts:
            parts.append(f"  ** {a}")

    parts.append("---")

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
        "max_tokens": 1500,  # Cap output — enough for tool calls, kills long thinking
    }
    # Let OpenRouter auto-route to the best available provider

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

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"OpenRouter API error 502: invalid JSON response ({e})")

        if "error" in data:
            raise RuntimeError(f"OpenRouter error 500: {data['error']}")

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
    max_time: int,
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

        # ─── Pre-Game Planning Phase ──────────────────────────────────
        planning_strategy = ""
        planning_status = await env.call_tool("get_planning_status")

        if planning_status.get("planning_enabled", True) is not False:
            print("Starting pre-game planning phase...")
            planning_data = await env.call_tool("start_planning_phase")

            if planning_data.get("planning_active"):
                max_planning_turns = planning_data.get("max_turns", 10)
                opponent_summary = planning_data.get("opponent_summary", "")

                planning_prompt = (
                    f"## PRE-GAME PLANNING PHASE\n"
                    f"You have {max_planning_turns} turns to plan. Be efficient!\n\n"
                    f"### Map Intel\n"
                    f"Map: {planning_data.get('map', {}).get('map_name', '?')} "
                    f"({planning_data.get('map', {}).get('width', '?')}x"
                    f"{planning_data.get('map', {}).get('height', '?')})\n"
                    f"Your base: ({planning_data.get('base_position', {}).get('x', '?')}, "
                    f"{planning_data.get('base_position', {}).get('y', '?')})\n"
                    f"Enemy estimated: ({planning_data.get('enemy_estimated_position', {}).get('x', '?')}, "
                    f"{planning_data.get('enemy_estimated_position', {}).get('y', '?')})\n"
                    f"Your faction: {planning_data.get('your_faction', '?')} ({planning_data.get('your_side', '?')})\n\n"
                    f"### Opponent Intelligence\n{opponent_summary}\n\n"
                    f"### How to Plan Efficiently\n"
                    f"1. Call get_faction_briefing() — returns ALL your units and buildings with full stats\n"
                    f"2. Call get_map_analysis() — returns terrain, resource locations, strategic notes\n"
                    f"3. Review the opponent intel above and the key units/buildings data below\n"
                    f"4. Call end_planning_phase(strategy='your detailed strategy') to begin gameplay\n\n"
                    f"Do NOT look up units or buildings one at a time. "
                    f"get_faction_briefing() gives you everything in one call. "
                    f"You can also use batch_lookup() for targeted multi-item queries.\n\n"
                    f"Think about: build order, unit composition, timing of attacks, "
                    f"defense priorities, and how to counter the opponent's tendencies."
                )
                messages.append({"role": "user", "content": planning_prompt})

                # Planning loop (bounded by max_planning_turns + margin)
                planning_done = False
                for planning_turn in range(max_planning_turns + 2):
                    try:
                        response = await chat_completion(messages, openai_tools, api_key, model, verbose)
                    except (RuntimeError, httpx.ReadTimeout, httpx.ConnectTimeout):
                        print("  [Planning] API error, ending planning phase.")
                        break
                    if response is None:
                        break

                    choice = response["choices"][0]
                    assistant_msg = choice["message"]
                    messages.append(assistant_msg)

                    if verbose and assistant_msg.get("content"):
                        print(f"  [Planning] {assistant_msg['content'][:200]}")

                    tool_calls = assistant_msg.get("tool_calls", [])
                    if not tool_calls:
                        messages.append({
                            "role": "user",
                            "content": (
                                "Use game knowledge tools to research, then call "
                                "end_planning_phase(strategy='...') when ready."
                            ),
                        })
                        continue

                    for tc in tool_calls:
                        fn_name = tc["function"]["name"]
                        try:
                            fn_args = json.loads(tc["function"].get("arguments", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            fn_args = {}

                        if verbose:
                            args_str = json.dumps(fn_args)
                            if len(args_str) > 80:
                                args_str = args_str[:80] + "..."
                            print(f"  [Planning Tool] {fn_name}({args_str})")

                        try:
                            result = await env.call_tool(fn_name, **fn_args)
                        except Exception as e:
                            result = {"error": str(e)}

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps(result) if not isinstance(result, str) else result,
                        })

                        # Check if planning ended
                        if isinstance(result, dict):
                            if result.get("planning_complete"):
                                planning_strategy = result.get("strategy", "")
                                planning_done = True
                                if verbose:
                                    print(f"  [Planning] Strategy: {planning_strategy[:150]}...")
                            elif result.get("planning_expired"):
                                planning_strategy = result.get("strategy", "")
                                planning_done = True
                                print(f"  [Planning] Expired: {result.get('reason', '?')}")

                    if planning_done:
                        break

                if not planning_done:
                    # Force end planning
                    try:
                        result = await env.call_tool(
                            "end_planning_phase",
                            strategy="(planning timed out, no explicit strategy)"
                        )
                        planning_strategy = result.get("strategy", "")
                    except Exception:
                        pass
                    print("  Planning phase timed out, proceeding to gameplay.")

                print(f"Planning phase complete. Strategy recorded: {bool(planning_strategy)}")
            else:
                if verbose:
                    print(f"  Planning: {planning_data.get('message', 'skipped')}")

        # ─── Game Start ───────────────────────────────────────────────
        state = await env.call_tool("get_game_state")
        briefing = compose_pregame_briefing(state)

        strategy_section = ""
        if planning_strategy:
            strategy_section = f"\n\n## Your Pre-Game Strategy\n{planning_strategy}\n"

        messages.append({
            "role": "user",
            "content": (
                f"Game started!{strategy_section}\n\n{briefing}\n\n"
                f"## Current State\n```json\n{json.dumps(state, indent=2)}\n```\n\n"
                f"Deploy your MCV and start building. Execute your strategy!"
            ),
        })

        total_tool_calls = 0
        total_api_calls = 0
        start_time = time.time()
        game_done = False
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 3

        turn = 0
        while True:
            # Check limits
            elapsed = time.time() - start_time
            if max_time and elapsed >= max_time:
                print(f"\n  TIME LIMIT reached ({max_time}s). Stopping.")
                break
            if max_turns and turn >= max_turns:
                break
            turn += 1

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

            # Call LLM with retry for rate limits
            response = None
            for attempt in range(4):
                try:
                    response = await chat_completion(messages, openai_tools, api_key, model, verbose)
                    break
                except (RuntimeError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                    err_str = str(e)
                    retriable = isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout)) or \
                        any(code in err_str for code in ("429", "500", "502", "503", "504"))
                    if retriable and attempt < 3:
                        wait = 10 * (attempt + 1)
                        print(f"\n  [RETRY] Provider error, waiting {wait}s ({attempt + 1}/4)...")
                        await asyncio.sleep(wait)
                    else:
                        print(f"\n  [ERROR] API call failed: {e}")
                        break
            if response is None:
                print("  [ERROR] All retries exhausted, stopping.")
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
                    if len(result_preview) > 500:
                        result_preview = result_preview[:500] + "..."
                    print(f"  [Result] {result_preview}")

            # Status update
            if total_api_calls % 5 == 0 or game_done:
                elapsed = time.time() - start_time
                limit_str = f"/{max_turns}" if max_turns else ""
                time_str = f"{elapsed:.0f}/{max_time}s" if max_time else f"{elapsed:.0f}s"
                print(
                    f"  Turn {turn}{limit_str} | "
                    f"API calls: {total_api_calls} | "
                    f"Tool calls: {total_tool_calls} | "
                    f"Time: {time_str}"
                )

            if game_done:
                break

            # Check finish reason
            if choice.get("finish_reason") == "stop" and not tool_calls:
                messages.append({
                    "role": "user",
                    "content": "Continue playing. Use game tools to check state and take actions.",
                })

        # Surrender so the replay has a proper ending
        if not game_done:
            try:
                await env.call_tool("surrender")
                print("\n  Surrendered (replay will have proper ending)")
            except Exception:
                pass

        # Final report
        elapsed = time.time() - start_time
        print()
        print("=" * 70)
        print(f"Agent finished after {total_api_calls} API calls, {total_tool_calls} tool calls")
        print(f"Time: {elapsed:.1f}s ({elapsed / max(total_api_calls, 1):.1f}s per API call)")

        # Get final state and scorecard
        try:
            final = await env.call_tool("get_game_state")
            mil = final.get("military", {})
            eco = final.get("economy", {})
            print(f"Result: {final.get('result', 'ongoing').upper()}")
            print()
            print("--- SCORECARD ---")
            print(f"  Planning:         {'ON — ' + planning_strategy[:100] if planning_strategy else 'OFF'}")
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
            print(f"  Own units:        {final.get('own_units', '?')}")
            print(f"  Own buildings:    {final.get('own_buildings', '?')}")
            print()
        except Exception as e:
            print(f"  (could not get final state: {e})")

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
        default=0,
        help="Maximum LLM turns, 0 = unlimited (default: 0)",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=int(os.environ.get("MAX_TIME", "1800")),
        help="Maximum wall-clock time in seconds (default: $MAX_TIME or 1800 = 30min)",
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
        asyncio.run(run_agent(args.url, args.api_key, args.model, args.max_turns, args.max_time, args.verbose))
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
