"""Standard MCP server for OpenRA-RL (stdio transport).

Exposes all game tools over the MCP protocol using FastMCP.
Connects to the game server WebSocket and proxies tool calls.

Usage:
    openra-rl mcp-server
    openra-rl mcp-server --server-url http://localhost:8000

Works with OpenClaw, Claude Desktop, and any MCP client.
"""

import json
import logging
from typing import Annotated, Any, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

logger = logging.getLogger("openra-rl-mcp")

# Lazy-initialized shared state
_client = None
_server_url = "http://localhost:8000"
_game_started = False
_UNSET = object()  # Sentinel: distinguishes "not loaded yet" from "loaded but disabled/failed"
_directives_manager = _UNSET

mcp = FastMCP(
    "openra-rl",
    instructions="Play Command & Conquer: Red Alert via AI tool calls",
)


async def _get_client():
    """Get or create the WebSocket client connection."""
    global _client
    if _client is not None:
        return _client
    from openra_env.mcp_ws_client import OpenRAMCPClient
    _client = OpenRAMCPClient(base_url=_server_url, message_timeout_s=300.0)
    await _client.connect()
    return _client


def _load_directives_manager():
    """Load directives manager from config file if available.

    Uses a sentinel (_UNSET) so that a disabled/failed load is cached and
    load_config() is not re-executed on every subsequent call.
    """
    global _directives_manager
    if _directives_manager is not _UNSET:
        return _directives_manager  # Already loaded (may be None if disabled/failed)

    try:
        from openra_env.config import load_config
        from openra_env.directives import DirectivesManager

        config = load_config()
        if getattr(config, "directives", None) and config.directives.enabled:
            _directives_manager = DirectivesManager(config.directives)
        else:
            _directives_manager = None
    except Exception:
        logger.warning("Failed to load directives config; directives will be unavailable", exc_info=True)
        _directives_manager = None

    return _directives_manager


async def _ensure_game() -> None:
    """Ensure game server is running and a game is started."""
    global _game_started
    if _game_started:
        return

    # Check if server is healthy
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.urlopen(f"{_server_url}/health", timeout=3)
        if req.status == 200:
            client = await _get_client()
            await client.reset()
            _game_started = True
            _load_directives_manager()  # Load directives on game start
            return
    except (urllib.error.URLError, OSError):
        pass

    # Try starting Docker container
    try:
        from openra_env.cli.docker_manager import (
            check_docker, is_running, start_server, wait_for_health,
        )
        if not is_running():
            if not check_docker():
                raise RuntimeError(
                    "Docker is not available. Start the game server manually: "
                    "docker run -p 8000:8000 ghcr.io/yxc20089/openra-rl:latest"
                )
            port = int(_server_url.split(":")[-1].split("/")[0]) if ":" in _server_url else 8000
            start_server(port=port)
            wait_for_health(port=port)
    except ImportError:
        raise RuntimeError(
            f"Game server not reachable at {_server_url}. "
            "Start it manually: docker run -p 8000:8000 ghcr.io/yxc20089/openra-rl:latest"
        )

    client = await _get_client()
    await client.reset()
    _game_started = True
    _load_directives_manager()  # Load directives on game start


async def _call(tool_name: str, **kwargs) -> Any:
    """Call a game tool and return the result."""
    await _ensure_game()
    client = await _get_client()
    return await client.call_tool(tool_name, **kwargs)


def _format(result: Any) -> str:
    """Format a tool result as a string."""
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=2, default=str)


# ── Game Lifecycle ─────────────────────────────────────────────────

@mcp.tool()
async def start_game() -> str:
    """Start a new Red Alert game. Returns initial game state."""
    global _game_started
    _game_started = False
    await _ensure_game()
    state = await _call("get_game_state")
    return _format(state)


@mcp.tool()
async def get_game_state() -> str:
    """Get current game state: economy, units, buildings, enemies, production."""
    return _format(await _call("get_game_state"))


@mcp.tool()
async def advance(
    ticks: Annotated[int, Field(description="Number of game ticks to advance (~25 ticks = 1 game-second). Range 1-5000. Use 100-200 for movement, 300-750 for building construction.")] = 50,
) -> str:
    """Advance the game by N ticks at accelerated speed (~25 ticks = 1 game-second).
    Production, movement, combat, and auto-placement all require game time.
    Also triggers auto-placement of buildings queued via build_and_place().
    Typical build times: power plant ~300 ticks, barracks ~500, war factory ~750."""
    return _format(await _call("advance", ticks=ticks))


# ── Economy & Info ─────────────────────────────────────────────────

@mcp.tool()
async def get_economy() -> str:
    """Get economy info: cash, ore, power, harvesters."""
    return _format(await _call("get_economy"))


@mcp.tool()
async def get_units() -> str:
    """Get list of your units with positions, health, type."""
    return _format(await _call("get_units"))


@mcp.tool()
async def get_buildings() -> str:
    """Get list of your buildings with positions, production, power."""
    return _format(await _call("get_buildings"))


@mcp.tool()
async def get_enemies() -> str:
    """Get visible enemy units and buildings."""
    return _format(await _call("get_enemies"))


@mcp.tool()
async def get_production() -> str:
    """Get current production queue and available builds."""
    return _format(await _call("get_production"))


@mcp.tool()
async def get_map_info() -> str:
    """Get map dimensions, name, and metadata."""
    return _format(await _call("get_map_info"))


@mcp.tool()
async def get_exploration_status() -> str:
    """Get fog-of-war data: explored %, quadrants, enemy found."""
    return _format(await _call("get_exploration_status"))


# ── Knowledge ──────────────────────────────────────────────────────

@mcp.tool()
async def lookup_unit(
    unit_type: Annotated[str, Field(description="Unit type code, e.g. 'e1' (rifle infantry), 'e2' (grenadier), '3tnk' (heavy tank), 'dog' (attack dog)")],
) -> str:
    """Look up stats for a unit type (e.g. 'e1', '3tnk')."""
    return _format(await _call("lookup_unit", unit_type=unit_type))


@mcp.tool()
async def lookup_building(
    building_type: Annotated[str, Field(description="Building type code, e.g. 'powr' (power plant), 'barr' (barracks), 'weap' (war factory), 'tent' (allied barracks)")],
) -> str:
    """Look up stats for a building type (e.g. 'powr', 'weap')."""
    return _format(await _call("lookup_building", building_type=building_type))


@mcp.tool()
async def lookup_tech_tree(
    faction: Annotated[str, Field(description="Faction name: 'soviet' or 'allied'")] = "soviet",
) -> str:
    """Get full tech tree and build order for a faction ('allied' or 'soviet')."""
    return _format(await _call("lookup_tech_tree", faction=faction))


@mcp.tool()
async def lookup_faction(
    faction: Annotated[str, Field(description="Faction name: 'soviet' or 'allied'")],
) -> str:
    """Get all available units and buildings for a faction."""
    return _format(await _call("lookup_faction", faction=faction))


@mcp.tool()
async def get_faction_briefing() -> str:
    """Get ALL units and buildings for your faction with full stats. Best for planning."""
    return _format(await _call("get_faction_briefing"))


@mcp.tool()
async def get_map_analysis() -> str:
    """Get strategic map analysis: resources, terrain, chokepoints, quadrants."""
    return _format(await _call("get_map_analysis"))


@mcp.tool()
async def batch_lookup(
    queries: Annotated[list[dict], Field(description="List of lookup queries, each with 'type' ('unit' or 'building') and 'name' (type code). Example: [{\"type\":\"unit\",\"name\":\"3tnk\"}, {\"type\":\"building\",\"name\":\"weap\"}]")],
) -> str:
    """Batch multiple lookups. Example: [{"type":"unit","name":"3tnk"}, {"type":"building","name":"weap"}]"""
    return _format(await _call("batch_lookup", queries=queries))


# ── Planning ───────────────────────────────────────────────────────

@mcp.tool()
async def get_opponent_intel() -> str:
    """Get intelligence on the AI opponent: difficulty, tendencies, counters."""
    return _format(await _call("get_opponent_intel"))


@mcp.tool()
async def start_planning_phase() -> str:
    """Start pre-game planning phase with map intel and opponent report."""
    return _format(await _call("start_planning_phase"))


@mcp.tool()
async def end_planning_phase(
    strategy: Annotated[str, Field(description="Your chosen strategy summary (free text). Logged for analysis.")] = "",
) -> str:
    """End planning phase with your strategy. Begins gameplay."""
    return _format(await _call("end_planning_phase", strategy=strategy))


@mcp.tool()
async def get_planning_status() -> str:
    """Check if planning phase is active and remaining turns."""
    return _format(await _call("get_planning_status"))


# ── Movement ───────────────────────────────────────────────────────

@mcp.tool()
async def move_units(
    unit_ids: Annotated[str, Field(description="Comma-separated unit IDs (e.g. 'u1,u2'), or selectors: 'all_combat', 'all_idle', 'all_infantry', 'all_vehicles', 'type:e1' (all units of type)")],
    target_x: Annotated[int, Field(description="Target cell X coordinate (0 to map_width)")],
    target_y: Annotated[int, Field(description="Target cell Y coordinate (0 to map_height)")],
    queued: Annotated[bool, Field(description="If true, queue after current orders; if false, replace current orders")] = False,
) -> str:
    """Move units to a position. Units will path-find to the target cell."""
    return _format(await _call("move_units", unit_ids=unit_ids, target_x=target_x, target_y=target_y, queued=queued))


@mcp.tool()
async def attack_move(
    unit_ids: Annotated[str, Field(description="Comma-separated unit IDs (e.g. 'u1,u2'), or selectors: 'all_combat', 'all_idle', 'all_infantry', 'all_vehicles', 'type:e1'")],
    target_x: Annotated[int, Field(description="Target cell X coordinate (0 to map_width)")],
    target_y: Annotated[int, Field(description="Target cell Y coordinate (0 to map_height)")],
    queued: Annotated[bool, Field(description="If true, queue after current orders; if false, replace current orders")] = False,
) -> str:
    """Move units toward a position, automatically engaging any enemies encountered en route."""
    return _format(await _call("attack_move", unit_ids=unit_ids, target_x=target_x, target_y=target_y, queued=queued))


@mcp.tool()
async def attack_target(
    unit_ids: Annotated[str, Field(description="Comma-separated unit IDs or selectors (e.g. 'all_combat', 'type:3tnk')")],
    target_actor_id: Annotated[int, Field(description="Actor ID of the enemy unit or building to attack (from get_enemies)")],
    queued: Annotated[bool, Field(description="If true, queue after current orders; if false, replace current orders")] = False,
) -> str:
    """Order units to attack a specific enemy by actor ID."""
    return _format(await _call("attack_target", unit_ids=unit_ids, target_actor_id=target_actor_id, queued=queued))


@mcp.tool()
async def stop_units(
    unit_ids: Annotated[str, Field(description="Comma-separated unit IDs or selectors: 'all_combat', 'all_idle', 'all_infantry', 'all_vehicles', 'type:e1'")],
) -> str:
    """Stop units from moving or attacking."""
    return _format(await _call("stop_units", unit_ids=unit_ids))


# ── Production ─────────────────────────────────────────────────────

@mcp.tool()
async def build_unit(
    unit_type: Annotated[str, Field(description="Unit type code to train, e.g. 'e1' (rifle), 'e2' (grenadier), '3tnk' (heavy tank), 'v2rl' (V2 launcher)")],
    count: Annotated[int, Field(description="Number of units to queue for production")] = 1,
) -> str:
    """Train units. Requires the right production building (barracks, war factory)."""
    return _format(await _call("build_unit", unit_type=unit_type, count=count))


@mcp.tool()
async def build_structure(
    building_type: Annotated[str, Field(description="Building type code, e.g. 'powr' (power plant), 'barr' (barracks), 'weap' (war factory), 'proc' (ore refinery)")],
) -> str:
    """Start constructing a building (manual placement workflow).
    Call advance(ticks) to let construction finish, then place_building() to place it.
    Prefer build_and_place() which handles placement automatically."""
    return _format(await _call("build_structure", building_type=building_type))


@mcp.tool()
async def build_and_place(
    building_type: Annotated[str, Field(description="Building type code, e.g. 'powr', 'barr', 'weap', 'proc'")],
    cell_x: Annotated[int, Field(description="Preferred placement X coordinate (0 = auto-find best position)")] = 0,
    cell_y: Annotated[int, Field(description="Preferred placement Y coordinate (0 = auto-find best position)")] = 0,
) -> str:
    """Build a structure and auto-place it when construction finishes.
    Call advance(ticks) after this to let construction complete — placement is automatic.
    Do NOT call place_building() on buildings queued this way."""
    return _format(await _call("build_and_place", building_type=building_type, cell_x=cell_x, cell_y=cell_y))


# ── Building/Unit Actions ─────────────────────────────────────────

@mcp.tool()
async def place_building(
    building_type: Annotated[str, Field(description="Building type code to place (must be fully constructed via build_structure)")],
    cell_x: Annotated[int, Field(description="Placement X coordinate (0 = auto-find best position)")] = 0,
    cell_y: Annotated[int, Field(description="Placement Y coordinate (0 = auto-find best position)")] = 0,
) -> str:
    """Place a completed building on the map (only for build_structure workflow).
    Do NOT use on buildings queued via build_and_place() — those auto-place via advance().
    Cell coordinates are optional — engine auto-finds position if omitted."""
    return _format(await _call("place_building", building_type=building_type, cell_x=cell_x, cell_y=cell_y))


@mcp.tool()
async def cancel_production(
    item_type: Annotated[str, Field(description="Type code of the unit or building to cancel (e.g. 'e1', 'powr')")],
) -> str:
    """Cancel production of a unit or building type."""
    return _format(await _call("cancel_production", item_type=item_type))


@mcp.tool()
async def deploy_unit(
    unit_id: Annotated[int, Field(description="Actor ID of the unit to deploy (e.g. MCV deploys into Construction Yard)")],
) -> str:
    """Deploy a unit (e.g. MCV → Construction Yard)."""
    return _format(await _call("deploy_unit", unit_id=unit_id))


@mcp.tool()
async def sell_building(
    building_id: Annotated[int, Field(description="Actor ID of the building to sell (from get_buildings)")],
) -> str:
    """Sell a building for partial refund."""
    return _format(await _call("sell_building", building_id=building_id))


@mcp.tool()
async def repair_building(
    building_id: Annotated[int, Field(description="Actor ID of the building to repair (from get_buildings)")],
) -> str:
    """Toggle repair on a building."""
    return _format(await _call("repair_building", building_id=building_id))


@mcp.tool()
async def set_rally_point(
    building_id: Annotated[int, Field(description="Actor ID of the production building (from get_buildings)")],
    cell_x: Annotated[int, Field(description="Rally point X coordinate")],
    cell_y: Annotated[int, Field(description="Rally point Y coordinate")],
) -> str:
    """Set rally point for a production building. New units go here automatically."""
    return _format(await _call("set_rally_point", building_id=building_id, cell_x=cell_x, cell_y=cell_y))


@mcp.tool()
async def guard_target(
    unit_ids: Annotated[str, Field(description="Comma-separated unit IDs or selectors (e.g. 'all_combat', 'type:e1')")],
    target_actor_id: Annotated[int, Field(description="Actor ID of the unit or building to guard")],
    queued: Annotated[bool, Field(description="If true, queue after current orders; if false, replace current orders")] = False,
) -> str:
    """Order units to guard a specific actor."""
    return _format(await _call("guard_target", unit_ids=unit_ids, target_actor_id=target_actor_id, queued=queued))


@mcp.tool()
async def set_stance(
    unit_ids: Annotated[str, Field(description="Comma-separated unit IDs or selectors (e.g. 'all_combat', 'type:e1')")],
    stance: Annotated[str, Field(description="Stance mode: 'holdfire', 'returnfire', 'defend', or 'attackanything'")],
) -> str:
    """Set unit stance: 'holdfire', 'returnfire', 'defend', 'attackanything'."""
    return _format(await _call("set_stance", unit_ids=unit_ids, stance=stance))


@mcp.tool()
async def harvest(
    unit_id: Annotated[int, Field(description="Actor ID of the harvester unit")],
    cell_x: Annotated[int, Field(description="Harvest location X coordinate (0 = auto-find nearest ore)")] = 0,
    cell_y: Annotated[int, Field(description="Harvest location Y coordinate (0 = auto-find nearest ore)")] = 0,
) -> str:
    """Send a harvester to harvest at a location."""
    return _format(await _call("harvest", unit_id=unit_id, cell_x=cell_x, cell_y=cell_y))


@mcp.tool()
async def power_down(
    building_id: Annotated[int, Field(description="Actor ID of the building to toggle power on/off")],
) -> str:
    """Toggle power on a building to save electricity."""
    return _format(await _call("power_down", building_id=building_id))


@mcp.tool()
async def set_primary(
    building_id: Annotated[int, Field(description="Actor ID of the building to set as primary production facility")],
) -> str:
    """Set a building as the primary production facility."""
    return _format(await _call("set_primary", building_id=building_id))


# ── Placement ──────────────────────────────────────────────────────

@mcp.tool()
async def get_valid_placements(
    building_type: Annotated[str, Field(description="Building type code to find placements for (e.g. 'powr', 'barr')")],
    max_results: Annotated[int, Field(description="Maximum number of valid positions to return")] = 8,
) -> str:
    """Get valid placement locations for a building type."""
    return _format(await _call("get_valid_placements", building_type=building_type, max_results=max_results))


# ── Unit Groups ────────────────────────────────────────────────────

@mcp.tool()
async def assign_group(
    group_name: Annotated[str, Field(description="Name for the unit group (e.g. 'scouts', 'main_army', 'defense')")],
    unit_ids: Annotated[list[int], Field(description="List of unit actor IDs to assign to this group")],
) -> str:
    """Create a named group of units."""
    return _format(await _call("assign_group", group_name=group_name, unit_ids=unit_ids))


@mcp.tool()
async def add_to_group(
    group_name: Annotated[str, Field(description="Name of an existing unit group")],
    unit_ids: Annotated[list[int], Field(description="List of unit actor IDs to add to the group")],
) -> str:
    """Add units to an existing group."""
    return _format(await _call("add_to_group", group_name=group_name, unit_ids=unit_ids))


@mcp.tool()
async def get_groups() -> str:
    """List all unit groups and their members."""
    return _format(await _call("get_groups"))


@mcp.tool()
async def command_group(
    group_name: Annotated[str, Field(description="Name of the unit group to command")],
    command_type: Annotated[str, Field(description="Command: 'move', 'attack_move', 'attack', 'stop', or 'guard'")],
    target_x: Annotated[int, Field(description="Target X coordinate (for move/attack_move)")] = 0,
    target_y: Annotated[int, Field(description="Target Y coordinate (for move/attack_move)")] = 0,
    target_actor_id: Annotated[int, Field(description="Target actor ID (for attack/guard commands)")] = 0,
    queued: Annotated[bool, Field(description="If true, queue after current orders")] = False,
) -> str:
    """Issue a command to a unit group. command_type: move, attack_move, attack, stop, guard."""
    kwargs = dict(
        group_name=group_name, command_type=command_type,
        target_x=target_x, target_y=target_y,
        target_actor_id=target_actor_id, queued=queued,
    )
    return _format(await _call("command_group", **kwargs))


# ── Compound ───────────────────────────────────────────────────────

@mcp.tool()
async def batch(
    actions: Annotated[list[dict], Field(description="List of action dicts, each with 'tool' and tool-specific params. Example: [{\"tool\":\"build_unit\",\"unit_type\":\"e1\"}, {\"tool\":\"move_units\",\"unit_ids\":\"all_combat\",\"target_x\":50,\"target_y\":30}]")],
) -> str:
    """Execute multiple actions simultaneously in one tick. Does NOT advance game time.
    Cannot contain advance() or query tools."""
    return _format(await _call("batch", actions=actions))


@mcp.tool()
async def plan(
    steps: Annotated[list[dict], Field(description="List of step dicts, each with 'tool' and tool-specific params. Executed sequentially with state refresh between each.")],
) -> str:
    """Execute steps sequentially with state refresh between each.
    Does NOT advance game time between steps — use advance() standalone for that."""
    return _format(await _call("plan", steps=steps))


# ── Utility ────────────────────────────────────────────────────────

@mcp.tool()
async def get_replay_path() -> str:
    """Get the path to the current game's replay file."""
    return _format(await _call("get_replay_path"))


@mcp.tool()
async def surrender() -> str:
    """Surrender the current game."""
    return _format(await _call("surrender"))


# ── Terrain ────────────────────────────────────────────────────────

@mcp.tool()
async def get_terrain_at(
    cell_x: Annotated[int, Field(description="Cell X coordinate to query")],
    cell_y: Annotated[int, Field(description="Cell Y coordinate to query")],
) -> str:
    """Get terrain type at a specific cell."""
    return _format(await _call("get_terrain_at", cell_x=cell_x, cell_y=cell_y))


# ── Strategic Directives ───────────────────────────────────────────

@mcp.tool()
async def check_directives() -> str:
    """Get all active strategic directives from high command.
    Shows pregame strategy, standing orders, and tactical adjustments.
    Use this to review your orders or when you need strategic guidance."""
    manager = _load_directives_manager()
    if manager is None:
        return "No directives configured. Set directives.enabled=true in config.yaml."
    return manager.format_for_mcp_tool()


@mcp.tool()
async def acknowledge_directive(
    directive_id: Annotated[int, Field(description="The ID of the directive to acknowledge")],
) -> str:
    """Acknowledge that you have received and understood a directive.
    Confirms to high command that orders are being followed."""
    manager = _load_directives_manager()
    if manager is None:
        return "No directives configured. Set directives.enabled=true in config.yaml."

    success = manager.acknowledge_directive(directive_id)
    if success:
        directive = manager.get_directive_by_id(directive_id)
        return f"Acknowledged directive {directive_id}: {directive.text}"
    else:
        return f"Error: Directive {directive_id} not found"


@mcp.tool()
async def get_directives_status() -> str:
    """Get status of all directives (total, acknowledged, unacknowledged).
    Shows which directives have been acknowledged and which need attention."""
    manager = _load_directives_manager()
    if manager is None:
        return "No directives configured. Set directives.enabled=true in config.yaml."
    return _format(manager.get_status_summary())


# ── Entry Point ────────────────────────────────────────────────────

def main(server_url: Optional[str] = None) -> None:
    """Run the MCP stdio server."""
    global _server_url
    if server_url:
        _server_url = server_url
    mcp.run(transport="stdio")
