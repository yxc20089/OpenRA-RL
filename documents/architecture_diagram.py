"""
OpenRA-RL Architecture Diagram
From the LLM Agent's perspective, showing MCP servers, tools, and game engine internals.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Style constants ──────────────────────────────────────────────
C = {
    "bg": "#0d1117",
    "box_border": "#30363d",
    "text": "#e6edf3",
    "text_dim": "#8b949e",
    "text_sub": "#b0b8c4",
    "agent": "#bc8cff",
    "mcp_client": "#58a6ff",
    "mcp_server": "#2b5b84",
    "env": "#2b5b84",
    "grpc_client": "#E84D31",
    "process": "#58a6ff",
    "proto": "#F5A623",
    "grpc_port": "#E84D31",
    "csharp": "#178600",
    "csharp_dark": "#2d6a2d",
    "world": "#2d4a2d",
    "channels": "#d29922",
    "docker": "#2496ED",
    "reward": "#3fb950",
    "arrow_obs": "#58a6ff",
    "arrow_action": "#f0883e",
    "arrow_ctrl": "#8b949e",
    "arrow_mcp": "#bc8cff",
    "game_loop": "#00d4aa",
    "grpc": "#F5A623",
    "layer_agent": "#1f1a30",
    "layer_mcp": "#1a2235",
    "layer_backend": "#1a2235",
    "layer_csharp": "#1a2a1a",
}

BH = 1.0  # standard box height (compact)

W, H = 24, 30
fig, ax = plt.subplots(1, 1, figsize=(W, H), facecolor=C["bg"])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)


def box(x, y, w, h, label, color, sub=None, fs=11, bold=True, tc=None, alpha=0.85, cr=0.25):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={cr}",
                        facecolor=color, edgecolor="white", linewidth=1.2, alpha=alpha, zorder=3)
    ax.add_patch(p)
    t = tc or C["text"]
    wt = "bold" if bold else "normal"
    if sub:
        ax.text(x + w/2, y + h/2 + 0.13, label, ha="center", va="center",
                fontsize=fs, fontweight=wt, color=t, zorder=4)
        sc = "#1a1a1a" if tc == "#1a1a1a" else C["text_sub"]
        ax.text(x + w/2, y + h/2 - 0.15, sub, ha="center", va="center",
                fontsize=max(fs - 3, 7), color=sc, zorder=4, style="italic")
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fs, fontweight=wt, color=t, zorder=4)


def layer(x, y, w, h, color, label, lx=None, ly=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.4",
                        facecolor=color, edgecolor=C["box_border"], linewidth=1.5, alpha=0.5, zorder=1)
    ax.add_patch(p)
    ax.text(lx or x + 0.5, ly or y + h - 0.3, label, ha="left", va="top",
            fontsize=11, fontweight="bold", color=C["text_dim"], zorder=2, fontstyle="italic")


def arrow(x1, y1, x2, y2, color, label=None, lw=2.0, lo=(0, 0), cs="arc3,rad=0", fs=9, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color, lw=lw,
                         connectionstyle=cs, zorder=5, mutation_scale=18)
    ax.add_patch(a)
    if label:
        mx, my = (x1+x2)/2 + lo[0], (y1+y2)/2 + lo[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs, color=color,
                zorder=6, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=C["bg"], edgecolor=color,
                          alpha=0.9, linewidth=0.8))


def tool_list_box(x, y_top, w, title, items, title_color, bg_color):
    """Compact tool box with tight, centered 2-column layout."""
    cols = 2
    n = len(items)
    rows = (n + cols - 1) // cols
    line_h = 0.25
    pad_top = 0.50
    pad_bot = 0.10
    h = pad_top + rows * line_h + pad_bot
    y = y_top - h
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.2",
                        facecolor=bg_color, edgecolor=title_color, linewidth=1.5, alpha=0.6, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y_top - 0.25, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color=title_color, zorder=4)
    char_w = 0.125
    col0_items = items[:rows]
    col1_items = items[rows:]
    max_col0 = max(len(s) for s in col0_items) * char_w if col0_items else 0
    max_col1 = max(len(s) for s in col1_items) * char_w if col1_items else 0
    gap = 0.3
    total_content_w = max_col0 + gap + max_col1
    left_pad = (w - total_content_w) / 2
    first_item_y = y_top - pad_top
    for i, item in enumerate(items):
        col = i // rows
        row = i % rows
        ix = x + left_pad + (0 if col == 0 else max_col0 + gap)
        iy = first_item_y - row * line_h
        ax.text(ix, iy, item, ha="left", va="center",
                fontsize=11, color=C["text_sub"], zorder=4, fontfamily="monospace")
    return y, y_top


# ══════════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════════

ax.text(W/2, H - 0.5, "OpenRA-RL System Architecture", ha="center", va="center",
        fontsize=24, fontweight="bold", color=C["text"], zorder=10)
ax.text(W/2, H - 1.0, "From the LLM Agent's Perspective", ha="center", va="center",
        fontsize=12, color=C["text_dim"], zorder=10)

# ══════════════════════════════════════════════════════════════════
#  LAYER BACKGROUNDS
# ══════════════════════════════════════════════════════════════════

# Agent layer (OUTSIDE Docker)
layer(0.5, 25.5, W - 1, 3.2, C["layer_agent"], "LLM AGENT")

# Docker container outline
docker_top = 24.8
docker_box = FancyBboxPatch((0.3, 0.3), W - 0.6, docker_top - 0.3,
                             boxstyle="round,pad=0,rounding_size=0.5",
                             facecolor="none", edgecolor=C["docker"], linewidth=2.5,
                             linestyle=(0, (8, 4)), alpha=0.7, zorder=0)
ax.add_patch(docker_box)
ax.text(W - 1.0, docker_top + 0.15, "Docker Container", ha="right", va="bottom",
        fontsize=11, fontweight="bold", color=C["docker"], zorder=2,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=C["bg"], edgecolor=C["docker"],
                  alpha=0.9, linewidth=1.0))

# Inner layers
layer(0.6, 17.0, W - 1.2, 7.5, C["layer_mcp"], "MCP SERVER (FastMCP + OpenEnv)")
layer(0.6, 10.0, W - 1.2, 6.5, C["layer_backend"], "PYTHON BACKEND")
layer(0.6, 0.6, W - 1.2, 8.5, C["layer_csharp"], "C# GAME ENGINE (OpenRA / .NET 8)")

# ══════════════════════════════════════════════════════════════════
#  AGENT LAYER  (y: 25.5 – 28.7)
# ══════════════════════════════════════════════════════════════════

box(3.0, 27.5, 5.5, BH, "LLM Agent", C["agent"],
    sub="examples/llm_agent.py", fs=13)

box(10.5, 27.5, 4.0, BH, "LLM", "#6e40aa",
    sub="Claude / GPT via OpenRouter", fs=12)

arrow(8.5, 28.0, 10.5, 28.0, C["text_dim"], label="send prompts", lw=1.8, lo=(0, 0.32), fs=8)

box(3.0, 26.0, 5.5, BH, "MCP Client", C["mcp_client"],
    sub="WebSocket ws://localhost:8000", fs=10)

arrow(5.75, 27.5, 5.75, 27.0, C["arrow_mcp"], label="call tools", lw=1.5, lo=(1.3, 0), fs=8)

# ══════════════════════════════════════════════════════════════════
#  MCP SERVER LAYER  (y: 17.0 – 24.5)
# ══════════════════════════════════════════════════════════════════

box(4.0, 23.0, 6.0, BH, "OpenEnv Server", C["mcp_server"],
    sub="app.py (FastMCP + Uvicorn)", fs=12)

# MCP Client -> OpenEnv Server
arrow(5.75, 26.0, 7.0, 24.0, C["arrow_mcp"], label="send JSON-RPC", lw=2.0, lo=(-1.3, 0), fs=9)

# Tool boxes
obs_tools = ["get_game_state", "get_economy", "get_units", "get_buildings",
             "get_enemies", "get_production", "get_map_info", "get_terrain_at"]

act_tools = ["move_units", "attack_move", "attack_target", "stop_units",
             "build_unit", "build_structure", "build_and_place", "place_building",
             "deploy_unit", "sell_building", "repair_building", "set_rally_point",
             "guard_target", "set_stance", "harvest", "cancel_production",
             "power_down", "set_primary", "surrender", "advance"]

kc_tools = ["lookup_unit", "lookup_building", "lookup_tech_tree", "lookup_faction",
            "batch", "plan", "assign_group", "add_to_group",
            "get_groups", "command_group", "get_valid_placements", "get_replay_path"]

TOOLS_TOP = 21.8
obs_bot, _ = tool_list_box(1.5, TOOLS_TOP, 5.5, "Observation Tools (8)", obs_tools, "#58a6ff", "#0d1f3a")
act_bot, _ = tool_list_box(7.5, TOOLS_TOP, 6.5, "Action Tools (20)", act_tools, "#f0883e", "#2a1a0d")
kc_bot, _  = tool_list_box(14.5, TOOLS_TOP, 6.5, "Knowledge & Composite (13)", kc_tools, "#3fb950", "#0d2a14")

# OpenEnv Server -> tool boxes
arrow(5.5, 23.0, 4.25, TOOLS_TOP, "#58a6ff", lw=1.3, cs="arc3,rad=0.05",
      label="query state", lo=(-1.0, 0), fs=7)
arrow(7.0, 23.0, 10.75, TOOLS_TOP, "#f0883e", lw=1.3, cs="arc3,rad=-0.05",
      label="run actions", lo=(1.0, 0), fs=7)
arrow(10.0, 23.5, 17.75, TOOLS_TOP, "#3fb950", lw=1.3, cs="arc3,rad=-0.2",
      label="lookup data", lo=(0.8, 0.3), fs=7)

# ══════════════════════════════════════════════════════════════════
#  PYTHON BACKEND LAYER  (y: 10.0 – 16.5)
# ══════════════════════════════════════════════════════════════════

box(3.5, 14.5, 6.0, BH, "OpenRAEnvironment", C["env"],
    sub="MCPEnvironment impl", fs=12)
box(12.5, 14.5, 5.0, BH, "Game Data", C["channels"],
    sub="game_data.py (static RA stats)", fs=10, alpha=0.7)

box(1.5, 11.0, 5.0, BH, "BridgeClient", C["grpc_client"],
    sub="bridge_client.py (gRPC :9999)", fs=11)
box(8.0, 11.0, 5.0, BH, "ProcessManager", C["process"],
    sub="openra_process.py", fs=11)

# Tools -> OpenRAEnvironment
env_tx = 6.5
env_ty = 15.5
arrow(4.25, obs_bot, env_tx, env_ty, "#58a6ff", lw=1.5, cs="arc3,rad=0.05",
      label="read state", lo=(-1.0, 0), fs=7)
arrow(10.75, act_bot, env_tx, env_ty, "#f0883e", lw=1.5, cs="arc3,rad=-0.05",
      label="apply actions", lo=(1.0, 0), fs=7)
arrow(17.75, kc_bot, 15.0, env_ty, "#3fb950", label="read stats", lw=1.2, lo=(0.8, 0.3), fs=8,
      cs="arc3,rad=-0.1")

# OpenRAEnvironment -> backend row
env_bx = 6.5
env_by = 14.5
arrow(env_bx, env_by, 4.0, 12.0, C["arrow_obs"], label="read game state",
      lw=1.8, lo=(-1.3, 0), fs=8, cs="arc3,rad=0.08")
arrow(env_bx, env_by, 10.5, 12.0, C["arrow_ctrl"], label="launch/kill game",
      lw=1.5, lo=(0.8, 0.3), fs=8)

# ══════════════════════════════════════════════════════════════════
#  BRIDGE: Python Backend -> C# Game Engine
# ══════════════════════════════════════════════════════════════════

# BridgeClient -> ExternalBotBridge (single gRPC arrow, obs + actions)
arrow(4.0, 11.0, 2.5, 7.5, C["grpc"], label="stream obs & actions",
      lw=2.5, lo=(-1.5, 0), fs=9)

# ProcessManager -> ExternalBotBridge (launches game that hosts gRPC server)
arrow(10.5, 11.0, 4.5, 7.5, C["arrow_ctrl"], label="start game",
      lw=1.5, lo=(1.5, 0.3), fs=8)

# ══════════════════════════════════════════════════════════════════
#  C# GAME ENGINE LAYER  (y: 0.6 – 9.1)
# ══════════════════════════════════════════════════════════════════

# Top row (y=6.5)
box(1.5, 6.5, 4.0, BH, "ExternalBotBridge", C["csharp"],
    sub="IBot + ITick + Kestrel", fs=10)

box(6.5, 6.5, 4.0, BH, "RLBridgeService", C["csharp"],
    sub="gRPC (decoupled loops)", fs=10)

box(11.5, 6.5, 4.0, BH, "ActionHandler", C["csharp"],
    sub="Proto -> OpenRA Orders", fs=10)

box(16.5, 6.5, 3.5, BH, "Spatial Map", C["csharp_dark"],
    sub="9ch H x W x 9", fs=9)

# Bottom row (y=3.5)
box(1.5, 3.5, 3.5, BH, "Channels", C["channels"],
    sub="DropOldest (obs=1)", fs=9, alpha=0.8)

box(6.0, 3.5, 4.5, BH, "ObservationSerializer", C["csharp_dark"],
    sub="World state -> protobuf", fs=9)

box(11.5, 3.5, 4.0, BH, "Game World", C["world"],
    sub="Actors, Map, Rules, Fog", fs=9)

box(16.5, 3.5, 3.5, BH, "Game Loop", C["csharp"],
    sub="~25 ticks/sec", fs=9)

# ── C# internal arrows ──────────────────────────────────────────

# ExternalBotBridge -> RLBridgeService
arrow(5.5, 7.0, 6.5, 7.0, C["reward"], lw=1.5,
      label="await tasks", lo=(0, 0.30), fs=7)

# RLBridgeService -> ActionHandler
arrow(10.5, 7.0, 11.5, 7.0, C["arrow_action"], lw=1.5, label="send commands", lo=(0, 0.28), fs=7)

# ExternalBotBridge -> Channels
arrow(3.5, 6.5, 3.25, 4.5, C["channels"], lw=1.5, label="enqueue obs", lo=(0.8, 0), fs=7)

# Channels -> ObservationSerializer
arrow(5.0, 4.0, 6.0, 4.0, C["arrow_obs"], lw=1.5, label="read buffer", lo=(0, 0.28), fs=7)

# Game World -> ObservationSerializer
arrow(11.5, 4.0, 10.5, 4.0, C["arrow_obs"], lw=1.5, label="read world", lo=(0, 0.28), fs=7)

# ActionHandler -> Game World
arrow(13.5, 6.5, 13.5, 4.5, C["arrow_action"], lw=1.5, label="issue orders", lo=(0.7, 0), fs=7)

# Game World -> Spatial Map (diagonal up)
arrow(15.5, 4.2, 16.5, 6.5, C["arrow_obs"], lw=1.2, cs="arc3,rad=-0.2",
      label="build spatial", lo=(0.7, 0), fs=7)

# Game Loop -> Game World
arrow(16.5, 4.0, 15.5, 4.0, C["game_loop"], lw=1.5, label="tick world", lo=(0, 0.28), fs=7)

# Game Loop ticks ExternalBotBridge (route around bottom)
arrow(18.25, 3.5, 18.25, 2.3, C["game_loop"], lw=1.5)
ax.annotate("", xy=(1.0, 2.3), xytext=(18.25, 2.3),
            arrowprops=dict(arrowstyle="-", color=C["game_loop"], lw=1.5), zorder=5)
ax.annotate("", xy=(1.0, 7.0), xytext=(1.0, 2.3),
            arrowprops=dict(arrowstyle="-", color=C["game_loop"], lw=1.5), zorder=5)
arrow(1.0, 7.0, 1.5, 7.0, C["game_loop"], lw=1.5)
ax.text(9.5, 2.0, "ITick (~25/sec)", ha="center", va="center", fontsize=9,
        color=C["game_loop"], zorder=6, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor=C["bg"], edgecolor=C["game_loop"],
                  alpha=0.9, linewidth=1.0))

# ══════════════════════════════════════════════════════════════════
#  LEGEND (top-right)
# ══════════════════════════════════════════════════════════════════

lx, ly = 17.5, 28.5
legend_bg = FancyBboxPatch((lx - 0.3, ly - 2.9), 6.5, 3.2,
    boxstyle="round,pad=0,rounding_size=0.2",
    facecolor=C["bg"], edgecolor=C["box_border"], linewidth=1.2, alpha=1.0, zorder=9)
ax.add_patch(legend_bg)
ax.text(lx, ly, "Legend:", fontsize=10, fontweight="bold", color=C["text_dim"], zorder=10)

for i, (color, label, ls) in enumerate([
    (C["arrow_mcp"], "MCP (agent <-> server)", "-"),
    (C["grpc"], "gRPC (Python <-> C#)", "-"),
    (C["arrow_obs"], "Observations (internal)", "-"),
    (C["arrow_action"], "Actions (internal)", "-"),
    (C["arrow_ctrl"], "Process / Control", "-"),
    (C["game_loop"], "Game Loop (ITick)", "-"),
    (C["docker"], "Docker Container", (0, (4, 2))),
]):
    yy = ly - 0.36 * (i + 1)
    ax.plot([lx, lx + 0.7], [yy, yy], color=color, lw=2.5, linestyle=ls, zorder=10)
    ax.text(lx + 0.95, yy, label, fontsize=8, color=C["text_dim"], va="center", zorder=10)

# ── Save ─────────────────────────────────────────────────────────
out = "/Users/berta/Projects/OpenRA-RL/documents/architecture_diagram.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=C["bg"], edgecolor="none")
plt.close()
print(f"Architecture diagram saved to: {out}")
