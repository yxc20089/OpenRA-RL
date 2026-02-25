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
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("openra-rl-mcp")

# Lazy-initialized shared state
_client = None
_server_url = "http://localhost:8000"
_game_started = False

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
async def start_game(difficulty: str = "normal") -> str:
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
async def advance(ticks: int = 50) -> str:
    """Advance the game by N ticks (~25 ticks = 1 second).
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
async def lookup_unit(unit_type: str) -> str:
    """Look up stats for a unit type (e.g. 'e1', '3tnk')."""
    return _format(await _call("lookup_unit", unit_type=unit_type))


@mcp.tool()
async def lookup_building(building_type: str) -> str:
    """Look up stats for a building type (e.g. 'powr', 'weap')."""
    return _format(await _call("lookup_building", building_type=building_type))


@mcp.tool()
async def lookup_tech_tree(faction: str = "soviet") -> str:
    """Get full tech tree and build order for a faction ('allied' or 'soviet')."""
    return _format(await _call("lookup_tech_tree", faction=faction))


@mcp.tool()
async def lookup_faction(faction: str) -> str:
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
async def batch_lookup(queries: list[dict]) -> str:
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
async def end_planning_phase(strategy: str = "") -> str:
    """End planning phase with your strategy. Begins gameplay."""
    return _format(await _call("end_planning_phase", strategy=strategy))


@mcp.tool()
async def get_planning_status() -> str:
    """Check if planning phase is active and remaining turns."""
    return _format(await _call("get_planning_status"))


# ── Movement ───────────────────────────────────────────────────────

@mcp.tool()
async def move_units(unit_ids: str, target_x: int, target_y: int, queued: bool = False) -> str:
    """Move units to a position. unit_ids: comma-separated IDs, 'all_combat', 'type:e1', etc."""
    return _format(await _call("move_units", unit_ids=unit_ids, target_x=target_x, target_y=target_y, queued=queued))


@mcp.tool()
async def attack_move(unit_ids: str, target_x: int, target_y: int, queued: bool = False) -> str:
    """Move units, engaging enemies en route. Best for advancing your army."""
    return _format(await _call("attack_move", unit_ids=unit_ids, target_x=target_x, target_y=target_y, queued=queued))


@mcp.tool()
async def attack_target(unit_ids: str, target_actor_id: int, queued: bool = False) -> str:
    """Order units to attack a specific enemy by actor ID."""
    return _format(await _call("attack_target", unit_ids=unit_ids, target_actor_id=target_actor_id, queued=queued))


@mcp.tool()
async def stop_units(unit_ids: str) -> str:
    """Stop units from moving or attacking."""
    return _format(await _call("stop_units", unit_ids=unit_ids))


# ── Production ─────────────────────────────────────────────────────

@mcp.tool()
async def build_unit(unit_type: str, count: int = 1) -> str:
    """Train units. Requires the right production building (barracks, war factory)."""
    return _format(await _call("build_unit", unit_type=unit_type, count=count))


@mcp.tool()
async def build_structure(building_type: str) -> str:
    """Start constructing a building (manual placement workflow).
    Call advance(ticks) to let construction finish, then place_building() to place it.
    Prefer build_and_place() which handles placement automatically."""
    return _format(await _call("build_structure", building_type=building_type))


@mcp.tool()
async def build_and_place(building_type: str, cell_x: int = 0, cell_y: int = 0) -> str:
    """Build a structure and auto-place it when construction finishes.
    Call advance(ticks) after this to let construction complete — placement is automatic.
    Do NOT call place_building() on buildings queued this way."""
    return _format(await _call("build_and_place", building_type=building_type, cell_x=cell_x, cell_y=cell_y))


# ── Building/Unit Actions ─────────────────────────────────────────

@mcp.tool()
async def place_building(building_type: str, cell_x: int = 0, cell_y: int = 0) -> str:
    """Place a completed building on the map (only for build_structure workflow).
    Do NOT use on buildings queued via build_and_place() — those auto-place via advance().
    Cell coordinates are optional — engine auto-finds position if omitted."""
    return _format(await _call("place_building", building_type=building_type, cell_x=cell_x, cell_y=cell_y))


@mcp.tool()
async def cancel_production(item_type: str) -> str:
    """Cancel production of a unit or building type."""
    return _format(await _call("cancel_production", item_type=item_type))


@mcp.tool()
async def deploy_unit(unit_id: int) -> str:
    """Deploy a unit (e.g. MCV → Construction Yard)."""
    return _format(await _call("deploy_unit", unit_id=unit_id))


@mcp.tool()
async def sell_building(building_id: int) -> str:
    """Sell a building for partial refund."""
    return _format(await _call("sell_building", building_id=building_id))


@mcp.tool()
async def repair_building(building_id: int) -> str:
    """Toggle repair on a building."""
    return _format(await _call("repair_building", building_id=building_id))


@mcp.tool()
async def set_rally_point(building_id: int, cell_x: int, cell_y: int) -> str:
    """Set rally point for a production building. New units go here automatically."""
    return _format(await _call("set_rally_point", building_id=building_id, cell_x=cell_x, cell_y=cell_y))


@mcp.tool()
async def guard_target(unit_ids: str, target_actor_id: int, queued: bool = False) -> str:
    """Order units to guard a specific actor."""
    return _format(await _call("guard_target", unit_ids=unit_ids, target_actor_id=target_actor_id, queued=queued))


@mcp.tool()
async def set_stance(unit_ids: str, stance: str) -> str:
    """Set unit stance: 'holdfire', 'returnfire', 'defend', 'attackanything'."""
    return _format(await _call("set_stance", unit_ids=unit_ids, stance=stance))


@mcp.tool()
async def harvest(unit_id: int, cell_x: int = 0, cell_y: int = 0) -> str:
    """Send a harvester to harvest at a location."""
    return _format(await _call("harvest", unit_id=unit_id, cell_x=cell_x, cell_y=cell_y))


@mcp.tool()
async def power_down(building_id: int) -> str:
    """Toggle power on a building to save electricity."""
    return _format(await _call("power_down", building_id=building_id))


@mcp.tool()
async def set_primary(building_id: int) -> str:
    """Set a building as the primary production facility."""
    return _format(await _call("set_primary", building_id=building_id))


# ── Placement ──────────────────────────────────────────────────────

@mcp.tool()
async def get_valid_placements(building_type: str, max_results: int = 8) -> str:
    """Get valid placement locations for a building type."""
    return _format(await _call("get_valid_placements", building_type=building_type, max_results=max_results))


# ── Unit Groups ────────────────────────────────────────────────────

@mcp.tool()
async def assign_group(group_name: str, unit_ids: list[int]) -> str:
    """Create a named group of units."""
    return _format(await _call("assign_group", group_name=group_name, unit_ids=unit_ids))


@mcp.tool()
async def add_to_group(group_name: str, unit_ids: list[int]) -> str:
    """Add units to an existing group."""
    return _format(await _call("add_to_group", group_name=group_name, unit_ids=unit_ids))


@mcp.tool()
async def get_groups() -> str:
    """List all unit groups and their members."""
    return _format(await _call("get_groups"))


@mcp.tool()
async def command_group(
    group_name: str,
    command_type: str,
    target_x: int = 0,
    target_y: int = 0,
    target_actor_id: int = 0,
    queued: bool = False,
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
async def batch(actions: list[dict]) -> str:
    """Execute multiple actions simultaneously in one tick. Does NOT advance game time.
    Cannot contain advance() or query tools. Example: [{"tool":"build_unit","unit_type":"e1"}]"""
    return _format(await _call("batch", actions=actions))


@mcp.tool()
async def plan(steps: list[dict]) -> str:
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
async def get_terrain_at(cell_x: int, cell_y: int) -> str:
    """Get terrain type at a specific cell."""
    return _format(await _call("get_terrain_at", cell_x=cell_x, cell_y=cell_y))


# ── Entry Point ────────────────────────────────────────────────────

def main(server_url: Optional[str] = None) -> None:
    """Run the MCP stdio server."""
    global _server_url
    if server_url:
        _server_url = server_url
    mcp.run(transport="stdio")
