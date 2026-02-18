"""
OpenRA-RL Game State Diagram
Shows the full game lifecycle from IDLE to GAME OVER, plus replay playback flow.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Style constants (matching architecture diagram) ─────────────
C = {
    "bg": "#0d1117",
    "text": "#e6edf3",
    "text_dim": "#8b949e",
    "text_sub": "#d8dee4",
    "box_border": "#30363d",
    # State colors
    "python": "#2b5b84",       # Python-side states
    "engine": "#178600",       # C# game engine states
    "grpc": "#F5A623",         # gRPC bridge states
    "replay": "#00d4aa",       # Replay states
    "error": "#f0883e",        # Error states
    "idle": "#6e40aa",         # IDLE (neutral/start)
    "cleanup": "#8b949e",      # CLEANUP (grey)
    # Arrow colors
    "arrow_main": "#58a6ff",   # Main flow
    "arrow_replay": "#00d4aa", # Replay flow
    "arrow_error": "#f0883e",  # Error flow
    "arrow_loop": "#bc8cff",   # Loop-back arrows
}

W, H = 19, 26
fig, ax = plt.subplots(1, 1, figsize=(W, H), facecolor=C["bg"])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)

BW = 4.0   # box width
BH = 0.9   # box height


def state_box(x, y, label, color, sub=None, w=BW, h=BH, fs=12, badge=None):
    """Draw a state box centered at (x, y)."""
    bx = x - w / 2
    by = y - h / 2
    p = FancyBboxPatch((bx, by), w, h, boxstyle="round,pad=0,rounding_size=0.2",
                        facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.85, zorder=3)
    ax.add_patch(p)
    if sub:
        ax.text(x, y + 0.12, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=C["text"], zorder=4)
        ax.text(x, y - 0.14, sub, ha="center", va="center",
                fontsize=max(fs - 3, 7), color=C["text_sub"], zorder=4, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=C["text"], zorder=4)
    if badge:
        bx_r = x + w / 2 + 0.15
        by_c = y
        bp = FancyBboxPatch((bx_r, by_c - 0.18), len(badge) * 0.12 + 0.3, 0.36,
                             boxstyle="round,pad=0,rounding_size=0.1",
                             facecolor=C["bg"], edgecolor=color, linewidth=1.0, alpha=0.9, zorder=5)
        ax.add_patch(bp)
        ax.text(bx_r + (len(badge) * 0.12 + 0.3) / 2, by_c, badge, ha="center", va="center",
                fontsize=7, fontweight="bold", color=color, zorder=6)


def arrow(x1, y1, x2, y2, color, label=None, lw=2.0, lo=(0, 0), fs=9,
          cs="arc3,rad=0", style="-|>", ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color, lw=lw,
                         connectionstyle=cs, zorder=5, mutation_scale=16, linestyle=ls)
    ax.add_patch(a)
    if label:
        mx, my = (x1 + x2) / 2 + lo[0], (y1 + y2) / 2 + lo[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs, color=color,
                zorder=6, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", facecolor=C["bg"], edgecolor=color,
                          alpha=0.9, linewidth=0.8))


# ══════════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════════

ax.text(W / 2, H - 0.5, "OpenRA-RL Game State Diagram", ha="center", va="center",
        fontsize=22, fontweight="bold", color=C["text"], zorder=10)
ax.text(W / 2, H - 1.0, "Lifecycle: Launch → Play → Game Over  |  Replay Playback",
        ha="center", va="center", fontsize=10, color=C["text_dim"], zorder=10)

# ══════════════════════════════════════════════════════════════════
#  COLUMN HEADERS
# ══════════════════════════════════════════════════════════════════

LIVE_X = 6.0     # center of live game column
REPLAY_X = 13.5  # center of replay column

ax.text(LIVE_X, 24.0, "Live Game", ha="center", va="center",
        fontsize=13, fontweight="bold", color=C["arrow_main"], zorder=10)
ax.text(REPLAY_X, 24.0, "Replay Playback", ha="center", va="center",
        fontsize=13, fontweight="bold", color=C["arrow_replay"], zorder=10)

# Subtle column divider
ax.plot([9.75, 9.75], [1.5, 24.3], color=C["box_border"], lw=1.0, ls="--", alpha=0.4, zorder=1)

# ══════════════════════════════════════════════════════════════════
#  IDLE STATE (shared, top center)
# ══════════════════════════════════════════════════════════════════

IDLE_Y = 23.0
state_box(W / 2, IDLE_Y, "IDLE", C["idle"], sub="environment constructed, no game running")

# ══════════════════════════════════════════════════════════════════
#  LIVE GAME COLUMN (left)
# ══════════════════════════════════════════════════════════════════

LAUNCH_Y = 20.8
LOAD_Y = 18.8
CONNECT_Y = 16.8
STREAM_Y = 14.8
PLAY_Y = 12.5
OVER_Y = 10.0
CLEAN_Y = 7.5

state_box(LIVE_X, LAUNCH_Y, "LAUNCHING", C["python"],
          sub="dotnet OpenRA.dll subprocess")
state_box(LIVE_X, LOAD_Y, "LOADING", C["engine"],
          sub="map, rules, traits, gRPC server")
state_box(LIVE_X, CONNECT_Y, "CONNECTING", C["grpc"],
          sub="BridgeClient retries GetState() RPC")
state_box(LIVE_X, STREAM_Y, "STREAMING", C["grpc"],
          sub="GameSession RPC, bg obs reader")
state_box(LIVE_X, PLAY_Y, "PLAYING", C["engine"],
          sub="step() loop · recording .orarep")
state_box(LIVE_X, OVER_Y, "GAME OVER", C["engine"],
          sub="done=True, result: win / lose / draw", w=4.5)
state_box(LIVE_X, CLEAN_Y, "CLEANUP", C["cleanup"],
          sub="close bridge, kill process")

# ── Live game arrows ────────────────────────────────────────────

# IDLE -> LAUNCHING
arrow(W / 2 - 1.0, IDLE_Y - BH / 2, LIVE_X, LAUNCH_Y + BH / 2,
      C["arrow_main"], label="call reset()", lo=(-0.8, 0.1), fs=8)

# LAUNCHING -> LOADING
arrow(LIVE_X, LAUNCH_Y - BH / 2, LIVE_X, LOAD_Y + BH / 2,
      C["arrow_main"], label="spawn process", lo=(1.2, 0), fs=8)

# LOADING -> CONNECTING
arrow(LIVE_X, LOAD_Y - BH / 2, LIVE_X, CONNECT_Y + BH / 2,
      C["arrow_main"], label="start gRPC server", lo=(1.3, 0), fs=8)

# CONNECTING -> STREAMING
arrow(LIVE_X, CONNECT_Y - BH / 2, LIVE_X, STREAM_Y + BH / 2,
      C["arrow_main"], label="establish session", lo=(1.3, 0), fs=8)

# STREAMING -> PLAYING
arrow(LIVE_X, STREAM_Y - BH / 2, LIVE_X, PLAY_Y + BH / 2,
      C["arrow_main"], label="receive first obs", lo=(1.3, 0), fs=8)

# PLAYING -> GAME OVER
arrow(LIVE_X, PLAY_Y - BH / 2, LIVE_X, OVER_Y + BH / 2,
      C["arrow_main"], label="detect game end", lo=(1.3, 0), fs=8)

# GAME OVER -> CLEANUP
arrow(LIVE_X, OVER_Y - BH / 2, LIVE_X, CLEAN_Y + BH / 2,
      C["arrow_main"], label="close streams", lo=(1.2, 0), fs=8)

# CLEANUP -> IDLE (loop back, left side)
arrow(LIVE_X - BW / 2, CLEAN_Y, 0.8, CLEAN_Y,
      C["arrow_loop"], lw=1.5, style="-")
ax.annotate("", xy=(0.8, IDLE_Y), xytext=(0.8, CLEAN_Y),
            arrowprops=dict(arrowstyle="-", color=C["arrow_loop"], lw=1.5), zorder=5)
arrow(0.8, IDLE_Y, W / 2 - BW / 2, IDLE_Y,
      C["arrow_loop"], lw=1.5)
ax.text(0.5, (CLEAN_Y + IDLE_Y) / 2, "next episode", ha="center", va="center",
        fontsize=8, fontweight="bold", color=C["arrow_loop"], zorder=6, rotation=90,
        bbox=dict(boxstyle="round,pad=0.12", facecolor=C["bg"], edgecolor=C["arrow_loop"],
                  alpha=0.9, linewidth=0.8))

# ── PLAYING self-loop (agent step cycle) ────────────────────────

arrow(LIVE_X + BW / 2, PLAY_Y + 0.15, LIVE_X + BW / 2 + 0.6, PLAY_Y + 0.15,
      C["engine"], lw=1.5, style="-")
ax.annotate("", xy=(LIVE_X + BW / 2 + 0.6, PLAY_Y - 0.15),
            xytext=(LIVE_X + BW / 2 + 0.6, PLAY_Y + 0.15),
            arrowprops=dict(arrowstyle="-", color=C["engine"], lw=1.5), zorder=5)
arrow(LIVE_X + BW / 2 + 0.6, PLAY_Y - 0.15, LIVE_X + BW / 2, PLAY_Y - 0.15,
      C["engine"], lw=1.5)
ax.text(LIVE_X + BW / 2 + 1.5, PLAY_Y, "step()", ha="center", va="center",
        fontsize=8, fontweight="bold", color=C["engine"], zorder=6,
        bbox=dict(boxstyle="round,pad=0.12", facecolor=C["bg"], edgecolor=C["engine"],
                  alpha=0.9, linewidth=0.8))

# ══════════════════════════════════════════════════════════════════
#  REPLAY PLAYBACK COLUMN (right)
# ══════════════════════════════════════════════════════════════════

RLOAD_Y = 20.8
RPLAY_Y = 16.8
REND_Y = 12.5

state_box(REPLAY_X, RLOAD_Y, "LOADING REPLAY", C["replay"],
          sub="parse .orarep, extract metadata")
state_box(REPLAY_X, RPLAY_Y, "REPLAYING", C["replay"],
          sub="ReplayConnection feeds packets", w=4.5)
state_box(REPLAY_X, REND_Y, "REPLAY ENDED", C["replay"],
          sub="all packets consumed")

# ── Replay arrows ───────────────────────────────────────────────

# IDLE -> LOADING REPLAY
arrow(W / 2 + 1.0, IDLE_Y - BH / 2, REPLAY_X, RLOAD_Y + BH / 2,
      C["arrow_replay"], label="load .orarep", lo=(0.8, 0.1), fs=8)

# LOADING REPLAY -> REPLAYING
arrow(REPLAY_X, RLOAD_Y - BH / 2, REPLAY_X, RPLAY_Y + BH / 2,
      C["arrow_replay"], label="start playback", lo=(1.3, 0), fs=8)

# REPLAYING -> REPLAY ENDED
arrow(REPLAY_X, RPLAY_Y - BH / 2, REPLAY_X, REND_Y + BH / 2,
      C["arrow_replay"], label="consume all frames", lo=(1.4, 0), fs=8)

# REPLAY ENDED -> IDLE (loop back, right side)
arrow(REPLAY_X + BW / 2 + 0.25, REND_Y, W - 0.8, REND_Y,
      C["arrow_loop"], lw=1.5, style="-")
ax.annotate("", xy=(W - 0.8, IDLE_Y), xytext=(W - 0.8, REND_Y),
            arrowprops=dict(arrowstyle="-", color=C["arrow_loop"], lw=1.5), zorder=5)
arrow(W - 0.8, IDLE_Y, W / 2 + BW / 2, IDLE_Y,
      C["arrow_loop"], lw=1.5)

# ── REPLAYING self-loop (frame advance, right side) ──────────────

arrow(REPLAY_X + BW / 2 + 0.25, RPLAY_Y + 0.15, REPLAY_X + BW / 2 + 0.85, RPLAY_Y + 0.15,
      C["replay"], lw=1.5, style="-")
ax.annotate("", xy=(REPLAY_X + BW / 2 + 0.85, RPLAY_Y - 0.15),
            xytext=(REPLAY_X + BW / 2 + 0.85, RPLAY_Y + 0.15),
            arrowprops=dict(arrowstyle="-", color=C["replay"], lw=1.5), zorder=5)
arrow(REPLAY_X + BW / 2 + 0.85, RPLAY_Y - 0.15, REPLAY_X + BW / 2 + 0.25, RPLAY_Y - 0.15,
      C["replay"], lw=1.5)
ax.text(REPLAY_X + BW / 2 + 1.7, RPLAY_Y, "next frame", ha="center", va="center",
        fontsize=8, fontweight="bold", color=C["replay"], zorder=6,
        bbox=dict(boxstyle="round,pad=0.12", facecolor=C["bg"], edgecolor=C["replay"],
                  alpha=0.9, linewidth=0.8))

# ══════════════════════════════════════════════════════════════════
#  CROSS-LINK: GAME OVER saves replay
# ══════════════════════════════════════════════════════════════════

# Note: .orarep saved after game ends
ax.text(LIVE_X + BW / 2 + 0.3, OVER_Y + 0.4, ".orarep saved", ha="left", va="center",
        fontsize=7, fontweight="bold", color=C["text_dim"], zorder=6,
        bbox=dict(boxstyle="round,pad=0.1", facecolor=C["bg"], edgecolor=C["text_dim"],
                  alpha=0.9, linewidth=0.8))

# ══════════════════════════════════════════════════════════════════
#  ERROR PATHS (dashed, orange)
# ══════════════════════════════════════════════════════════════════

# CONNECTING -> TIMEOUT -> CLEANUP
ERR_X = 2.2
TIMEOUT_Y = 13.5
state_box(ERR_X, TIMEOUT_Y, "TIMEOUT", C["error"],
          sub="120 retries exhausted", w=2.8, h=0.8, fs=10)

arrow(LIVE_X - BW / 2, CONNECT_Y - 0.2, ERR_X + 1.4, TIMEOUT_Y + 0.4,
      C["arrow_error"], lw=1.5, ls="--", label="fail connect", lo=(-0.2, 0.4), fs=7)
arrow(ERR_X, TIMEOUT_Y - 0.4, LIVE_X - BW / 2, CLEAN_Y + 0.2,
      C["arrow_error"], lw=1.5, ls="--", label="abort", lo=(-0.3, 0), fs=7)

# PLAYING -> CONNECTION LOST -> CLEANUP
LOST_Y = 10.5
state_box(ERR_X, LOST_Y, "CONN LOST", C["error"],
          sub="stream broken", w=2.8, h=0.8, fs=10)

arrow(LIVE_X - BW / 2, PLAY_Y - 0.2, ERR_X + 1.4, LOST_Y + 0.4,
      C["arrow_error"], lw=1.5, ls="--", label="lose stream", lo=(-0.2, 0.4), fs=7)
arrow(ERR_X, LOST_Y - 0.4, LIVE_X - BW / 2, CLEAN_Y - 0.2,
      C["arrow_error"], lw=1.5, ls="--", label="abort", lo=(-0.3, 0), fs=7)

# ══════════════════════════════════════════════════════════════════
#  LEGEND
# ══════════════════════════════════════════════════════════════════

lx, ly = 12.0, 7.0
legend_bg = FancyBboxPatch((lx - 0.3, ly - 2.8), 5.8, 3.2,
    boxstyle="round,pad=0,rounding_size=0.2",
    facecolor=C["bg"], edgecolor=C["box_border"], linewidth=1.2, alpha=1.0, zorder=9)
ax.add_patch(legend_bg)
ax.text(lx, ly, "Legend:", fontsize=10, fontweight="bold", color=C["text_dim"], zorder=10)

for i, (color, label, ls) in enumerate([
    (C["python"], "Python-side state", "-"),
    (C["engine"], "C# Game Engine state", "-"),
    (C["grpc"], "gRPC Bridge state", "-"),
    (C["replay"], "Replay state", "-"),
    (C["arrow_error"], "Error path", (0, (4, 2))),
    (C["arrow_loop"], "Episode loop-back", "-"),
]):
    yy = ly - 0.38 * (i + 1)
    ax.plot([lx, lx + 0.7], [yy, yy], color=color, lw=2.5, linestyle=ls, zorder=10)
    ax.text(lx + 0.95, yy, label, fontsize=8, color=C["text_dim"], va="center", zorder=10)

# ── Save ─────────────────────────────────────────────────────────
out = "/Users/berta/Projects/OpenRA-RL/documents/game_state_diagram.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=C["bg"], edgecolor="none")
plt.close()
print(f"Game state diagram saved to: {out}")
