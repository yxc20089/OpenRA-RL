---
name: openra-rl
description: Play Red Alert RTS against AI opponents
version: 1.0.0
homepage: https://github.com/yxc20089/OpenRA-RL
metadata:
  openclaw:
    requires:
      bins: ["docker"]
    install:
      - type: uv
        package: openra-rl
---

# OpenRA-RL: Play Red Alert

You can play Command & Conquer: Red Alert against AI opponents.

## Setup

The game server runs in Docker. Make sure Docker Desktop is running.

## MCP Server Configuration

Add to your `~/.openclaw/openclaw.json`:

```json
{
  "agents": {
    "main": {
      "mcpServers": {
        "openra-rl": {
          "command": "openra-rl",
          "args": ["mcp-server"]
        }
      }
    }
  }
}
```

## Available Tools

- **start_game** — Start a new Red Alert match
- **get_game_state** — View economy, units, buildings, visible enemies
- **advance** — Let the game run for N ticks
- **build_unit / build_structure** — Train units or construct buildings
- **move_units / attack_move / attack_target** — Command your army
- **get_faction_briefing** — Get all unit/building stats for your faction
- **get_map_analysis** — Strategic map breakdown with resources and terrain
- **deploy_unit** — Deploy MCV to start building
- **set_rally_point** — Auto-send new units to a staging area
- **batch / plan** — Execute multiple commands efficiently
- ... and 35+ more tools for full game control

## Example

"Start a game of Red Alert on easy difficulty, build a base, train an army, and defeat the enemy."
