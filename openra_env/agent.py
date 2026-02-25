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

def _load_default_prompt() -> str:
    """Load the default system prompt shipped with the package."""
    from openra_env.prompts import load_default_prompt
    return load_default_prompt()


# Public constant for backward compatibility (lazy-loaded on first access)
SYSTEM_PROMPT = _load_default_prompt()


def load_system_prompt(config) -> str:
    """Resolve system prompt from config: inline > file > default.

    Priority:
      1. config.prompts.system_prompt (inline string)
      2. config.prompts.system_prompt_file (path to .txt file)
      3. config.agent.system_prompt (deprecated, backward compat)
      4. config.agent.system_prompt_file (deprecated, backward compat)
      5. Built-in default (openra_env/prompts/default.txt)
    """
    from pathlib import Path

    # Check prompts.* first (canonical location)
    prompts_cfg = getattr(config, "prompts", None)
    if prompts_cfg:
        if getattr(prompts_cfg, "system_prompt", ""):
            return prompts_cfg.system_prompt
        prompt_file = getattr(prompts_cfg, "system_prompt_file", "")
        if prompt_file:
            p = Path(prompt_file).expanduser()
            if p.is_file():
                return p.read_text(encoding="utf-8").strip()
            raise FileNotFoundError(f"system_prompt_file not found: {p}")

    # Backward compat: check agent.* (deprecated)
    agent_cfg = config.agent if hasattr(config, "agent") else config
    if getattr(agent_cfg, "system_prompt", ""):
        return agent_cfg.system_prompt
    prompt_file = getattr(agent_cfg, "system_prompt_file", "")
    if prompt_file:
        p = Path(prompt_file).expanduser()
        if p.is_file():
            return p.read_text(encoding="utf-8").strip()
        raise FileNotFoundError(f"system_prompt_file not found: {p}")

    # Default
    return SYSTEM_PROMPT


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

    parts = [
        "## Strategic Briefing",
        f"Map: {map_name} ({map_w}x{map_h})",
        f"Your faction: {faction or side} ({side})",
        f"Your base: ({base_x}, {base_y})",
        f"Enemy likely near: ({enemy_x}, {enemy_y})",
        f"Enemy approach direction: {defense_direction}",
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
        f"Funds: ${funds} (cash=${cash} + ore=${ore}) | Power: {state.get('power_balance', 0):+d} | Harvesters: {eco.get('harvester_count', 0)} | Explored: {state.get('explored_percent', 0)}%",
    ]

    # Minimap (ASCII spatial overview)
    minimap = state.get("minimap", "")
    if minimap:
        parts.append(minimap)

    # Base center from buildings
    buildings = state.get("buildings_summary", [])
    if buildings:
        base_x = sum(b["cell_x"] for b in buildings) // len(buildings)
        base_y = sum(b["cell_y"] for b in buildings) // len(buildings)
        parts.append(f"Base center: ({base_x},{base_y})")

    # Compact unit summary grouped by type, with IDs, positions, and activity
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
            entries = []
            for u in us:
                pos = f"{u['id']}@({u['cell_x']},{u['cell_y']})"
                if u.get("target_x") is not None:
                    pos += f"→({u['target_x']},{u['target_y']})"
                elif not u.get("idle"):
                    # Show short activity tag for non-idle units without tracked target
                    act = u.get("activity", "")
                    if act and act not in ("Idle", "Unknown", "Wait"):
                        tag = act[:3].lower()
                        pos += f"→{tag}"
                entries.append(pos)
            unit_parts.append(f"{len(us)}x{utype}[{','.join(entries)}]")
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

    # Enemy summary with IDs and positions (units + buildings)
    enemies = state.get("enemy_summary", [])
    enemy_bldgs = state.get("enemy_buildings_summary", [])
    if enemies or enemy_bldgs:
        enemy_parts = []
        if enemies:
            eby_type = defaultdict(list)
            for e in enemies:
                eby_type[e["type"]].append(e)
            for etype, es in eby_type.items():
                entries = ",".join(f"{e['id']}@({e['cell_x']},{e['cell_y']})" for e in es)
                enemy_parts.append(f"{len(es)}x{etype}[{entries}]")
        if enemy_bldgs:
            ebby_type = defaultdict(list)
            for b in enemy_bldgs:
                ebby_type[b["type"]].append(b)
            for btype, bs in ebby_type.items():
                entries = ",".join(f"{b['id']}@({b['cell_x']},{b['cell_y']})" for b in bs)
                enemy_parts.append(f"{len(bs)}x{btype}[{entries}]")
        # Average position of all visible enemies
        all_enemy_pos = (
            [(e["cell_x"], e["cell_y"]) for e in enemies]
            + [(b["cell_x"], b["cell_y"]) for b in enemy_bldgs]
        )
        avg_x = sum(p[0] for p in all_enemy_pos) // len(all_enemy_pos)
        avg_y = sum(p[1] for p in all_enemy_pos) // len(all_enemy_pos)
        parts.append(f"Enemies: {' '.join(enemy_parts)} center ({avg_x},{avg_y})")
    else:
        n_enemy = state.get("visible_enemy_units", 0)
        parts.append(f"Enemies: {'none visible' if n_enemy == 0 else f'{n_enemy} visible'}")

    prod = state.get("production_items", [])
    if prod:
        active = [p for p in prod if "@100%" not in p]
        ready = [p.split("@")[0] for p in prod if "@100%" in p]
        parts_prod = []
        if active:
            parts_prod.append(", ".join(active))
        if ready:
            parts_prod.append(f"READY TO PLACE: {', '.join(ready)}")
        parts.append(f"Production: {' | '.join(parts_prod)}")
    else:
        parts.append("Production: IDLE")

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


def _sanitize_messages(messages: list[dict], prompts=None) -> list[dict]:
    """Merge consecutive same-role messages for strict-alternation models (e.g. Mistral).

    Some models require strict user/assistant alternation and reject sequences
    like ``user → user`` or ``tool → user``.  This helper:
    1. Merges consecutive ``user`` messages by joining their content with newlines.
    2. Inserts a bridge ``assistant`` message when a ``tool`` result is followed
       by a ``user`` message (Mistral requires tool → assistant → user).
    """
    if not messages:
        return messages

    bridge = prompts.sanitize_bridge if prompts else "Acknowledged. Continuing."
    merged: list[dict] = [dict(messages[0])]
    for msg in messages[1:]:
        prev = merged[-1]
        # Merge consecutive user messages
        if msg["role"] == "user" and prev["role"] == "user":
            merged[-1] = {**prev, "content": prev["content"] + "\n\n" + msg["content"]}
            continue
        # Bridge: tool → user needs an assistant message in between
        if msg["role"] == "user" and prev["role"] == "tool":
            merged.append({"role": "assistant", "content": bridge})
        merged.append(msg)
    return merged


async def chat_completion(
    messages: list[dict],
    tools: list[dict],
    llm_config: LLMConfig,
    verbose: bool = False,
    prompts=None,
) -> dict:
    """Call an OpenAI-compatible chat completions API.

    Works with OpenRouter, Ollama, LM Studio, or any endpoint
    implementing the OpenAI Chat Completions spec with tool calling.
    """
    clean_messages = _sanitize_messages(messages, prompts=prompts)
    payload = {
        "model": llm_config.model,
        "messages": clean_messages,
        "max_tokens": llm_config.max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if llm_config.temperature is not None:
        payload["temperature"] = llm_config.temperature
    if llm_config.top_p is not None:
        payload["top_p"] = llm_config.top_p
    if llm_config.reasoning_effort is not None:
        payload["reasoning"] = {"effort": llm_config.reasoning_effort}

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
            error_text = response.text[:2000]
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
                    "Rate limited by LLM provider. Wait a minute and retry."
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


def compress_history(messages: list[dict], keep_last: int = 40,
                     trigger: int = 0, prompts=None, compression=None) -> list[dict]:
    """Compress message history to stay within context limits.

    Keeps the system prompt and the last ``keep_last`` messages, replacing
    earlier messages with a state-aware summary that preserves critical
    game context (buildings, economy, strategy, military, errors).

    Args:
        keep_last: Number of recent messages to keep after compression.
        trigger: Compress when total messages exceed this threshold.
            0 (default) means ``keep_last * 2``.
        prompts: PromptsConfig for customizable text.
        compression: CompressionConfig controlling what to include in summary.
    """
    threshold = trigger if trigger > 0 else keep_last * 2
    if len(messages) <= threshold:
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

    # Compression config defaults
    inc_strategy = compression.include_strategy if compression else True
    inc_military = compression.include_military if compression else True
    inc_production = compression.include_production if compression else True

    # Extract game state context from old messages
    last_state = {}
    building_types = set()
    unit_types_produced = set()
    strategy_text = ""
    errors = []

    for msg in old_messages:
        # Extract planning strategy from early user messages
        if inc_strategy and msg.get("role") == "user" and not strategy_text:
            content_str = msg.get("content", "")
            if isinstance(content_str, str):
                for line in content_str.split("\n"):
                    if line.strip().startswith("Strategy:"):
                        strategy_text = line.strip()
                        break

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

            # Track units produced (from build_unit notes)
            if inc_production and "note" in content:
                note = content["note"]
                if isinstance(note, str) and "queued" in note:
                    # Extract unit/building name from "'name' ... queued"
                    import re
                    m = re.search(r"'(\w+)'.*queued", note)
                    if m:
                        name = m.group(1)
                        # Distinguish units from buildings
                        if "per unit" in note or "each" in note:
                            unit_types_produced.add(name)
                        else:
                            building_types.add(name)

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

    if inc_strategy and strategy_text:
        parts.append(strategy_text)

    if building_types:
        parts.append(f"Buildings built: {', '.join(sorted(building_types))}")

    if inc_production and unit_types_produced:
        parts.append(f"Units produced: {', '.join(sorted(unit_types_produced))}")

    if inc_military and last_state:
        mil = last_state.get("military", {})
        if mil:
            parts.append(
                f"Military: {mil.get('units_killed', 0)} kills, "
                f"{mil.get('units_lost', 0)} losses"
            )

    if errors:
        unique = list(dict.fromkeys(errors))[-3:]
        parts.append(f"Recent issues: {'; '.join(unique)}")

    suffix = prompts.compression_suffix if prompts else "Game continues from current state."
    parts.append(suffix)

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

    # Auto-increase timeout for local models (they're slower than cloud APIs)
    is_local = any(h in llm_config.base_url for h in ("localhost", "127.0.0.1"))
    if is_local and llm_config.request_timeout_s <= 120.0:
        llm_config = llm_config.model_copy(update={"request_timeout_s": 300.0})

    print(f"Connecting to {url}...")
    print(f"Model: {llm_config.model} @ {llm_config.base_url}")
    if is_local:
        print(f"Timeout: {int(llm_config.request_timeout_s)}s (local model)")

    async with OpenRAMCPClient(base_url=url, message_timeout_s=300.0) as env:
        print("Resetting environment (launching OpenRA)...")
        await env.reset()

        # Discover and convert tools
        mcp_tools = await env.list_tools()
        openai_tools = mcp_tools_to_openai(mcp_tools)
        tool_names = {t["function"]["name"] for t in openai_tools}
        print(f"Discovered {len(mcp_tools)} MCP tools")

        if verbose:
            for t in mcp_tools:
                print(f"  - {t.name}: {t.description[:60]}...")

        # Initialize conversation
        system_prompt = load_system_prompt(config)
        messages = [{"role": "system", "content": system_prompt}]

        # ─── Pre-Game Planning Phase ──────────────────────────────────
        planning_strategy = ""
        planning_status = await env.call_tool("get_planning_status")

        if planning_status.get("planning_enabled", True) is not False:
            print("Starting pre-game planning phase...")
            planning_data = await env.call_tool("start_planning_phase")

            if planning_data.get("planning_active"):
                max_planning_turns = planning_data.get("max_turns", 10)
                opponent_summary = planning_data.get("opponent_summary", "")

                prompts = config.prompts
                planning_prompt = prompts.planning_prompt.format(
                    max_turns=max_planning_turns,
                    map_name=planning_data.get("map", {}).get("map_name", "?"),
                    map_width=planning_data.get("map", {}).get("width", "?"),
                    map_height=planning_data.get("map", {}).get("height", "?"),
                    base_x=planning_data.get("base_position", {}).get("x", "?"),
                    base_y=planning_data.get("base_position", {}).get("y", "?"),
                    enemy_x=planning_data.get("enemy_estimated_position", {}).get("x", "?"),
                    enemy_y=planning_data.get("enemy_estimated_position", {}).get("y", "?"),
                    faction=planning_data.get("your_faction", "?"),
                    side=planning_data.get("your_side", "?"),
                    opponent_summary=opponent_summary,
                    planning_nudge=prompts.planning_nudge,
                )
                messages.append({"role": "user", "content": planning_prompt})

                # Planning loop (bounded by max_planning_turns + margin)
                planning_done = False
                for planning_turn in range(max_planning_turns + 2):
                    try:
                        response = await chat_completion(messages, openai_tools, llm_config, verbose, prompts=config.prompts)
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
                            "content": prompts.planning_nudge,
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

        # Find MCV unit ID and barracks type for context
        mcv_id = None
        for u in state.get("units_summary", []):
            if u.get("type") == "mcv":
                mcv_id = u["id"]
                break
        faction = state.get("faction", "")
        barracks_type = "tent" if faction in {"england", "france", "germany"} else "barr"

        mcv_note = f" Your MCV is unit {mcv_id}." if mcv_id else ""

        game_start_prompts = config.prompts
        messages.append({
            "role": "user",
            "content": game_start_prompts.game_start.format(
                strategy_section=strategy_section,
                briefing=briefing,
                barracks_type=barracks_type,
                mcv_note=mcv_note,
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

            # Compress history periodically (unless disabled)
            if llm_config.compression_strategy != "none":
                messages = compress_history(
                    messages, keep_last=llm_config.keep_last_messages,
                    trigger=llm_config.compression_trigger,
                    prompts=config.prompts,
                    compression=config.prompts.compression)

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
            is_local = any(h in llm_config.base_url for h in ("localhost", "127.0.0.1"))
            for attempt in range(max_retries):
                try:
                    response = await chat_completion(messages, openai_tools, llm_config, verbose, prompts=config.prompts)
                    break
                except (httpx.ReadTimeout, httpx.ConnectTimeout):
                    timeout_s = int(llm_config.request_timeout_s)
                    print(f"\n  [ERROR] Request timed out after {timeout_s}s.")
                    if is_local:
                        print("  [HINT] Local models can be slow. Increase timeout in config.yaml:")
                        print(f"         llm.request_timeout_s: {timeout_s * 2}")
                    break
                except RuntimeError as e:
                    err_str = str(e)
                    retriable = any(code in err_str for code in ("429", "500", "502", "503", "504"))
                    if retriable and attempt < max_retries - 1:
                        wait = llm_config.retry_backoff_s * (attempt + 1)
                        print(f"\n  [RETRY] Provider error, waiting {wait}s ({attempt + 1}/{max_retries})...")
                        print(f"          {e}")
                        await asyncio.sleep(wait)
                    else:
                        print(f"\n  [ERROR] API call failed: {e}")
                        break
            if response is None:
                print("  [ERROR] Stopping agent.")
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
                    "content": config.prompts.no_tool_nudge,
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
                    # Suggest similar tools for unknown tool errors
                    if fn_name not in tool_names:
                        import difflib
                        close = difflib.get_close_matches(fn_name, tool_names, n=3, cutoff=0.4)
                        # Always include canonical build tools for build-related names
                        build_keywords = {"build", "place", "train", "produce", "construct"}
                        if any(kw in fn_name.lower() for kw in build_keywords):
                            for bt in ("build_unit", "build_structure", "build_and_place"):
                                if bt in tool_names and bt not in close:
                                    close.append(bt)
                        if close:
                            result["suggested_tools"] = close

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
                    "content": config.prompts.continue_nudge,
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
            print(f"  Explored:         {final.get('explored_percent', 0)}%")
            rv = final.get("reward_vector", {})
            if rv:
                print("  Reward vector:")
                for dim, val in rv.items():
                    print(f"    {dim:15s} {val:+.3f}")
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
