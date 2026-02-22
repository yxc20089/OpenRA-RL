"""LLM agent that plays Red Alert using any OpenAI-compatible model.

Supports OpenRouter, Ollama, LM Studio, or any local/remote endpoint
that implements the OpenAI Chat Completions API with tool calling.
"""

import asyncio
import json
import logging
import time

from collections import defaultdict

import httpx
from openra_env.config import LLMConfig
from openra_env.game_data import get_building_stats, get_faction_info, get_tech_tree, get_unit_stats
from openra_env.mcp_ws_client import OpenRAMCPClient

logger = logging.getLogger("llm_agent")

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

Unit selectors: comma-separated IDs (e.g. "145,146"), "all_combat", "all_idle", \
"type:e1" (all units of a type), "all_infantry", "all_vehicles", "all_aircraft", "all_ships", or a group name.

## Game Knowledge Tools
Use these to look up unit stats, building stats, and tech trees at any time:
- **get_faction_briefing()** — Get ALL units and buildings for your faction with full stats in one call. Best for planning.
- **get_map_analysis()** — Get strategic map summary: resource patches, water, terrain, quadrant breakdown, and exploration %.
- **get_exploration_status()** — Get fog-of-war data: overall/per-quadrant explored %, enemy found, idle combat/infantry counts.
- **batch_lookup(queries)** — Look up multiple items at once: [{"type":"unit","name":"3tnk"}, {"type":"building","name":"weap"}]
- **lookup_unit(unit_type)** — Get stats for a single unit (e.g. "3tnk", "e1")
- **lookup_building(building_type)** — Get stats for a single building (e.g. "weap", "proc")
- **lookup_tech_tree(faction)** — Get the full build order and tech tree for "allied" or "soviet"
- **lookup_faction(faction)** — Get all available units and buildings for a faction

## Game Mechanics

**Economy (TOP PRIORITY)**: Funds = cash + ore. Harvesters collect ore and bring it \
to refineries. Each ore refinery (proc) comes with one FREE harvester — so every \
refinery is both storage and income. More refineries = more harvesters = faster income. \
CRITICAL: Construction costs are paid incrementally — if you hit $0, ALL production \
pauses until income resumes. NEVER let funds reach zero. \
Build 2-3 ore refineries as early as possible. This is more important than military. \
If ore storage is near max, build a silo ($150) to avoid wasting harvester income. \
Protect your harvesters — if they die, your economy collapses.

**Power**: Buildings require power. Power plants (powr) provide +100. \
When power demand exceeds supply, ALL production slows to 1/3 speed — \
this is devastating. Always build a power plant BEFORE any building \
that drains power. Check your power balance in every briefing. \
Defense turrets consume power too — plan ahead and build extra power plants.

**Production**: Buildings produce units — barracks/tent → infantry, \
war factory (weap) → vehicles. Multiple production buildings of the same \
type speed up production. Queue items and advance() to let them finish.

**Tech tree**: Higher-tier buildings unlock stronger units. \
A war factory requires an ore refinery. Build order: powr → barracks → proc → weap. \
The "Can build:" line in each briefing shows what is currently available to produce.

**Unit strength**: Vehicles (tanks) are VASTLY stronger than infantry. \
A single heavy tank (3tnk, $950) can kill 10+ infantry (e1, $100 each). \
Allied uses medium tanks (2tnk, $800) or light tanks (1tnk, $600). \
CRITICAL LESSON: Once you have a war factory, STOP building infantry for combat. \
Switch IMMEDIATELY to tanks — they are your main fighting force. \
Infantry dies fast to tanks, turrets, and splash damage. Only build infantry \
for scouting (e1, dog) or anti-air (e3 rocket soldiers). \
Your army should be 70-80% tanks, with a few e3 mixed in for anti-air. \
Only build units listed in your available_production.

**Building placement**: build_and_place() handles placement automatically. \
For precise placement, use get_valid_placements(building_type) to see positions, \
then place_building(building_type, cell_x, cell_y). \
Buildings that need water (spen, syrd) will fail on land maps.

**Defense (HIGH PRIORITY)**: Defense turrets protect your base while you grow your economy. \
They are far more cost-effective than units for base defense.

Available turrets by faction:
- Allied early: Pillbox (pbox, $400, no power drain, needs barracks) — anti-infantry
- Allied early: Camo Pillbox (hbox, $600, no power drain, needs barracks) — hidden defense
- Allied mid: Gun Turret (gun, $600, -20 power, needs war factory) — anti-armor, best all-round
- Soviet early: Flame Tower (ftur, $600, -20 power, needs barracks) — anti-infantry
- Soviet mid: Tesla Coil (tsla, $1500, -75 power, needs war factory) — devastating but power-hungry

PLACEMENT IS CRITICAL — place turrets between your base and where the enemy will attack from:
1. The STRATEGIC BRIEFING tells you the enemy estimated position
2. Calculate the APPROACH DIRECTION: if enemy is NE, defenses go on the NE side of your base
3. Use get_valid_placements(building_type) to find candidate positions
4. Pick positions that are TOWARD the enemy direction — place_building(type, cell_x, cell_y)
5. Build 2 cheap turrets early (pbox/ftur) right after your first refinery
6. Add stronger turrets (gun/tsla) once you have a war factory
7. Also protect ore refineries — they are high-value targets enemies will target
8. Cover flanks as you expand — don't cluster all turrets in one spot

Example: Your base at (20,60), enemy estimated at (80,20). Enemy approaches from NE. \
Place turrets at (25,55), (28,52) — NE of your construction yard, between base and enemy.

TURRET QUANTITY: 2 turrets is NOT enough. Aim for 4-6 turrets minimum. \
As you expand, keep building turrets — they are cheap insurance. \
A well-turreted base can hold off attacks while you mass tanks.

ANTI-AIR IS ESSENTIAL: The enemy WILL build helicopters or aircraft mid-game. \
Build SAM sites (sam, Soviet, $750) or AA Guns (agun, Allied, $600) once you have \
a radar dome. Also keep 2-3 rocket soldiers (e3) near your base as mobile anti-air. \
Without anti-air, a single helicopter can destroy your entire base.

**Rally points**: Use set_rally_point(building_id, x, y) after building a barracks \
or war factory to auto-send new units to a staging area near your defenses.

**Scouting**: Send a cheap unit (e1 or dog) to explore the map early. \
Knowing the enemy's base location and army composition is critical. \
Use get_exploration_status() to see how much of the map is explored and which \
quadrants are still unexplored. Use "type:e1" or "all_infantry" selectors to \
move scout units. Once you find the enemy, you know where to aim your defenses.

**Expansion**: Every ore refinery comes with a free harvester. \
Build refinery #2 as soon as possible, refinery #3 when ore starts depleting. \
Place refineries near ore patches for faster collection. \
Always guard expansion refineries with at least 1 turret.

**Army composition**: Your army should be mostly TANKS once the war factory is up. \
Mix in 2-3 rocket soldiers (e3) for anti-air coverage. Engineers (e6) can capture \
enemy buildings for a surprise swing. Stop building e1 riflemen once you have tanks — \
they are cannon fodder against armored units.

**MASS BEFORE ATTACKING**: This is the #1 rule of RTS combat. \
NEVER send units to attack one by one — they will die one by one. \
Wait until you have 5+ tanks gathered in one spot, THEN attack_move together. \
Keep new units near your base defenses until the attack group is ready. \
Set rally points on your war factory to a staging area behind your turrets. \
A single attack with 8 tanks wins. Eight attacks with 1 tank each = 8 dead tanks.

## Strategy Priorities (ECONOMY & DEFENSE FIRST)
ACT FAST — every second of idle construction is wasted. Start building immediately!
1. Deploy MCV the INSTANT the game starts — don't wait even one tick
2. Build power plant IMMEDIATELY after MCV deploys — you need power for everything
3. Build barracks as soon as power plant starts, scout with a cheap unit toward OPPOSITE corner
4. Build ore refinery #1 — your economy starts here. NEVER let cash hit $0
5. Build 2 defense turrets (pbox/ftur) facing the enemy approach direction
6. Build ore refinery #2 immediately — double income is essential for sustained production
7. Build power plant #2 (defense turrets + production need power headroom)
8. War factory → IMMEDIATELY switch to building TANKS, not infantry
9. Build ore refinery #3 — triple income sustains tank production
10. Add 2-3 more turrets (gun/tsla) at base entrance and near refineries (aim for 4-6 total)
11. Radar dome → SAM/AA Gun for anti-air defense (enemy WILL build aircraft)
12. Keep 2-3 e3 rocket soldiers at base for mobile anti-air backup
13. MASS tanks behind your turret line — do NOT send them one by one
14. Attack only when you have 5+ tanks grouped together AND a strong economy
15. Never stop: more refineries, more turrets, more tanks — economy and defense win games

TEMPO: Use plan() and batch() to issue multiple build orders per turn. \
Don't build one thing at a time — queue the power plant, then immediately \
queue the barracks while power is building. Overlap production!

CRITICAL MISTAKES TO AVOID:
- Building infantry (e1) for combat after you have a war factory — TANKS are 10x better
- Sending units to attack one at a time — they die for nothing. MASS FIRST.
- Having only 1-2 turrets — build 4-6 minimum, more as you expand
- Ignoring anti-air — one helicopter can destroy your whole base. Build SAM/AA + keep e3 at base
- Letting production sit idle — always be building something (tanks, turrets, refineries)

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
Planning has a turn limit — aim to finish in 3-4 turns max. \
\
Your strategy MUST be a detailed 3-phase game plan adapted to the opponent and map. \
Study the opponent intel (aggressiveness, tendencies, weaknesses) and map analysis \
(size, resource locations, terrain chokepoints, water) before writing your plan. \
\
**EARLY GAME (ticks 0-5000, ~0-3 min):** \
- Exact build order: what buildings in what sequence, adapted to map size \
  (small maps = faster enemy contact, rush defenses; large maps = more economy time) \
- Scouting plan: which direction to scout based on map layout and enemy spawn estimate \
- Economy target: how many refineries by tick 5000 (aim for 2-3) \
- Defense placement: how many turrets and WHERE based on enemy approach direction \
- If opponent is aggressive (check intel): prioritize defenses and turrets earlier \
- If opponent is passive/slow (check intel): prioritize economy, delay defenses slightly \
\
**MID GAME (ticks 5000-15000, ~3-10 min):** \
- Tank production plan: which tank types to build based on your faction and opponent counters \
- Anti-air plan: when to build radar dome + SAM/AA, how many e3 to keep at base \
  (enemy AI WILL build aircraft — plan for it before it happens) \
- Turret expansion: add stronger turrets (gun/tsla) and cover flanks + refineries \
- Army composition target: e.g. "6 heavy tanks + 3 e3 rocket soldiers" \
- Massing point: where to stage your army before attacking (behind turret line) \
- If map has chokepoints: turret those chokepoints for massive defensive advantage \
- If map is open: need more mobile defense (tanks on patrol) \
\
**LATE GAME (ticks 15000+, ~10+ min):** \
- Attack plan: target priority (enemy harvesters → production buildings → CY) \
- Reinforcement pipeline: keep war factory producing while army attacks \
- Expansion plan: where to build additional refineries as ore depletes \
- Tech upgrades: radar dome → airfield/helipad for air strikes if economy allows \
- Win condition: how to finish the opponent (overwhelming tank push vs attrition) \
- If opponent is defensive (check intel): expand economy, build overwhelming force \
- If opponent rushes early (check intel): survive with turrets, then counter-attack when they're spent

## Briefing Format
Each turn briefing includes:
- Funds, power balance, harvester count — monitor harvester count closely!
- Your units with IDs and positions
- Your buildings with IDs and positions — check turret coverage
- Visible enemies with IDs and positions — note their approach direction
- Current production queue and available builds
- ALERTS for events needing attention (attacks, low power, idle production)
- Base center position — compare with enemy center to know defense direction

KEY CHECK EVERY TURN: Do you have enough refineries (3+)? Enough turrets (3+)? \
Is there a turret between each refinery and the enemy?
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

    # Calculate defense direction
    dx = enemy_x - base_x
    dy = enemy_y - base_y
    dir_parts = []
    if dy < -map_h // 6:
        dir_parts.append("North")
    elif dy > map_h // 6:
        dir_parts.append("South")
    if dx > map_w // 6:
        dir_parts.append("East")
    elif dx < -map_w // 6:
        dir_parts.append("West")
    defense_direction = "".join(dir_parts) if dir_parts else "Center"

    # Suggest turret placement offset (toward enemy, 3-5 cells from base center)
    norm = max(abs(dx), abs(dy), 1)
    turret_x = base_x + (dx * 5) // norm
    turret_y = base_y + (dy * 5) // norm

    parts = [
        "## Strategic Briefing",
        f"Map: {map_name} ({map_w}x{map_h})",
        f"Your faction: {faction or side} ({side})",
        f"Your base: ({base_x}, {base_y})",
        f"Enemy likely near: ({enemy_x}, {enemy_y}) — SCOUT to confirm!",
        f"Enemy approach direction: {defense_direction}",
        f">> PLACE DEFENSE TURRETS toward ({turret_x}, {turret_y}) — {defense_direction} side of base <<",
        "",
        f"Tech tree: {' → '.join(tech_order[:8])}{'...' if len(tech_order) > 8 else ''}",
        f"Barracks type: {barracks}",
        "",
        "Available units:",
        *unit_lines,
        "",
        "Available buildings:",
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


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Merge consecutive same-role messages for strict-alternation models (e.g. Mistral).

    Some models require strict user/assistant alternation and reject sequences
    like ``user → user`` or ``tool → user``.  This helper:
    1. Merges consecutive ``user`` messages by joining their content with newlines.
    2. Inserts a bridge ``assistant`` message when a ``tool`` result is followed
       by a ``user`` message (Mistral requires tool → assistant → user).
    """
    if not messages:
        return messages

    merged: list[dict] = [dict(messages[0])]
    for msg in messages[1:]:
        prev = merged[-1]
        # Merge consecutive user messages
        if msg["role"] == "user" and prev["role"] == "user":
            merged[-1] = {**prev, "content": prev["content"] + "\n\n" + msg["content"]}
            continue
        # Bridge: tool → user needs an assistant message in between
        if msg["role"] == "user" and prev["role"] == "tool":
            merged.append({"role": "assistant", "content": "Acknowledged. Continuing."})
        merged.append(msg)
    return merged


async def chat_completion(
    messages: list[dict],
    tools: list[dict],
    llm_config: LLMConfig,
    verbose: bool = False,
) -> dict:
    """Call an OpenAI-compatible chat completions API.

    Works with OpenRouter, Ollama, LM Studio, or any endpoint
    implementing the OpenAI Chat Completions spec with tool calling.
    """
    clean_messages = _sanitize_messages(messages)
    payload = {
        "model": llm_config.model,
        "messages": clean_messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": llm_config.max_tokens,
    }
    if llm_config.temperature is not None:
        payload["temperature"] = llm_config.temperature
    if llm_config.top_p is not None:
        payload["top_p"] = llm_config.top_p

    headers = dict(llm_config.extra_headers)
    if llm_config.api_key:
        headers["Authorization"] = f"Bearer {llm_config.api_key}"

    async with httpx.AsyncClient() as client:
        if verbose:
            n_msgs = len(clean_messages)
            roles = [m.get("role", "?") for m in clean_messages]
            print(f"  [LLM] Sending {n_msgs} messages to {llm_config.model}...")
            print(f"  [LLM] Roles: {' → '.join(roles)}")

        response = await client.post(
            llm_config.base_url,
            headers=headers,
            json=payload,
            timeout=llm_config.request_timeout_s,
        )

        if response.status_code != 200:
            error_text = response.text[:500]
            if response.status_code in (401, 403):
                raise RuntimeError(
                    f"Authentication failed ({response.status_code}). "
                    f"Check your API key: openra-rl config"
                )
            if response.status_code == 400 and "model" in error_text.lower():
                raise RuntimeError(
                    f"Invalid model ID '{llm_config.model}'. "
                    f"Update with: openra-rl config"
                )
            if response.status_code == 429:
                raise RuntimeError(
                    f"Rate limited by LLM provider. Wait a minute and retry."
                )
            raise RuntimeError(f"LLM API error {response.status_code}: {error_text}")

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"LLM API error 502: invalid JSON response ({e})")

        if "error" in data:
            raise RuntimeError(f"LLM API error 500: {data['error']}")

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


async def run_agent(config, verbose: bool = False):
    """Connect to OpenRA-RL and play a game using an LLM agent."""
    url = config.agent.server_url
    llm_config = config.llm
    max_turns = config.agent.max_turns
    max_time = config.agent.max_time_s

    print(f"Connecting to {url}...")
    print(f"Model: {llm_config.model} @ {llm_config.base_url}")

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
                        response = await chat_completion(messages, openai_tools, llm_config, verbose)
                    except (RuntimeError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                        print(f"  [Planning] API error: {e}")
                        print("  Skipping planning phase.")
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
        # Reset messages to just system prompt — planning context is captured
        # in the strategy text below. This avoids tool/user role alternation
        # issues with models that enforce strict message ordering (e.g. Mistral).
        messages = [messages[0]]  # keep only system prompt

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
                f"ACT NOW! Deploy your MCV immediately, then start building power plant + barracks. "
                f"Expand fast — every idle second costs you. Use plan() to chain: "
                f"deploy MCV → build power plant → build barracks → build refinery. "
                f"Then focus on economy (3+ refineries) and defense turrets toward the enemy."
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
            messages = compress_history(messages, keep_last=llm_config.keep_last_messages)

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
            max_retries = llm_config.max_retries
            for attempt in range(max_retries):
                try:
                    response = await chat_completion(messages, openai_tools, llm_config, verbose)
                    break
                except (RuntimeError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                    err_str = str(e)
                    retriable = isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout)) or \
                        any(code in err_str for code in ("429", "500", "502", "503", "504"))
                    if retriable and attempt < max_retries - 1:
                        wait = llm_config.retry_backoff_s * (attempt + 1)
                        print(f"\n  [RETRY] Provider error, waiting {wait}s ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait)
                    else:
                        print(f"\n  [ERROR] API call failed: {e}")
                        break
            if response is None:
                print("  [ERROR] All retries exhausted. Run with --verbose for details.")
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
