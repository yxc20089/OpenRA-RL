---
name: openra-rl
description: Play Command & Conquer Red Alert RTS — build bases, train armies, and defeat AI opponents using 48 MCP tools.
version: 1.1.0
metadata:
  openclaw:
    emoji: "🎮"
    homepage: https://github.com/yxc20089/OpenRA-RL
    requires:
      bins:
        - docker
      env: []
    install:
      - kind: uv
        package: openra-rl
        bins: [openra-rl]
    os: ["macos", "linux"]
---

# OpenRA-RL: Play Command & Conquer Red Alert

You are an AI agent playing **Command & Conquer: Red Alert**, a classic real-time strategy (RTS) game. You control one faction (Allied or Soviet) and must build a base, gather resources, train an army, and destroy the enemy.

The game runs in a Docker container. You interact through MCP tools that let you observe the battlefield, issue orders, and advance game time.

## Quick Start

### 1. Install

```bash
pip install openra-rl
```

### 2. Start the game server

```bash
openra-rl server start
```

This pulls the Docker image and starts the game server on port 8000. Verify with `openra-rl server status`.

### 3. Configure MCP

Add to your OpenClaw config (`~/.openclaw/openclaw.json`):

```json
{
  "mcpServers": {
    "openra-rl": {
      "command": "openra-rl",
      "args": ["mcp-server"]
    }
  }
}
```

### 4. Play

Tell your agent: *"Start a game of Red Alert and try to win."*

The agent will use the MCP tools listed below to observe and command.

---

## How the Game Works

- **Real-time**: The game runs continuously at ~25 ticks/second. Call `advance(ticks)` to let time pass.
- **Fog of war**: You can only see areas near your units/buildings. Scout to find the enemy.
- **Resources**: Harvest ore to earn credits. Credits buy buildings and units.
- **Power**: Buildings need power. Build Power Plants (`powr`) to stay powered. Low power slows production.
- **Tech tree**: Advanced buildings require prerequisites (e.g., War Factory needs Ore Refinery).

---

## MCP Tools Reference

### Observation (read the battlefield)

| Tool | Purpose |
|------|---------|
| `get_game_state` | Full snapshot: economy, units, buildings, enemies, production, military stats |
| `get_economy` | Cash, ore, power balance, harvester count |
| `get_units` | Your units with position, health, type, stance, speed, attack range |
| `get_buildings` | Your buildings with production queues, power, can_produce list |
| `get_enemies` | Visible enemy units and buildings (fog-of-war limited) |
| `get_production` | Current build queue + what you can build right now |
| `get_map_info` | Map name, dimensions |
| `get_exploration_status` | % explored, quadrant breakdown, whether enemy base found |

### Knowledge (learn the game)

| Tool | Purpose |
|------|---------|
| `lookup_unit(unit_type)` | Stats for a unit (e.g., `lookup_unit("e1")` → Rifle Infantry) |
| `lookup_building(building_type)` | Stats for a building (e.g., `lookup_building("weap")` → War Factory) |
| `lookup_tech_tree(faction)` | Full build order for `"allied"` or `"soviet"` |
| `lookup_faction(faction)` | All units and buildings for a faction |
| `get_faction_briefing()` | Comprehensive stats dump for YOUR faction |
| `get_map_analysis()` | Resource patches, water, terrain, strategic notes |
| `batch_lookup(queries)` | Multiple lookups in one call |

### Game Control

| Tool | Purpose |
|------|---------|
| `advance(ticks)` | **Critical** — advances the game by N ticks. Nothing happens without this. Use 25 ticks ≈ 1 second, 250 ticks ≈ 10 seconds. |

### Movement & Combat

| Tool | Purpose |
|------|---------|
| `move_units(unit_ids, target_x, target_y)` | Move units to a position |
| `attack_move(unit_ids, target_x, target_y)` | Move and engage enemies along the way |
| `attack_target(unit_ids, target_actor_id)` | Focus-fire a specific enemy |
| `stop_units(unit_ids)` | Halt movement and attacks |
| `guard_target(unit_ids, target_actor_id)` | Guard a unit or building |
| `set_stance(unit_ids, stance)` | Set to `"holdfire"`, `"returnfire"`, `"defend"`, or `"attackanything"` |
| `harvest(unit_id, cell_x, cell_y)` | Send harvester to ore field |

### Production

| Tool | Purpose |
|------|---------|
| `build_unit(unit_type, count)` | Train units (e.g., `build_unit("e1", 5)` → 5 Rifle Infantry) |
| `build_structure(building_type)` | Start constructing a building (needs manual placement) |
| `build_and_place(building_type, cell_x, cell_y)` | Build + auto-place when done (preferred) |
| `place_building(building_type, cell_x, cell_y)` | Place a completed building |
| `cancel_production(item_type)` | Cancel queued production |
| `get_valid_placements(building_type)` | Get valid locations to place a building |

### Building Management

| Tool | Purpose |
|------|---------|
| `deploy_unit(unit_id)` | Deploy MCV into Construction Yard |
| `sell_building(building_id)` | Sell for partial refund |
| `repair_building(building_id)` | Toggle auto-repair |
| `set_rally_point(building_id, cell_x, cell_y)` | New units go here |
| `power_down(building_id)` | Toggle power to save electricity |
| `set_primary(building_id)` | Set as primary production building |

### Unit Groups

| Tool | Purpose |
|------|---------|
| `assign_group(group_name, unit_ids)` | Create a named group |
| `add_to_group(group_name, unit_ids)` | Add units to existing group |
| `get_groups()` | List all groups |
| `command_group(group_name, command_type, ...)` | Command entire group |

### Compound Actions

| Tool | Purpose |
|------|---------|
| `batch(actions)` | Execute multiple actions in ONE tick (no time advance) |
| `plan(steps)` | Execute steps sequentially with state refresh between each |

### Utility

| Tool | Purpose |
|------|---------|
| `surrender()` | Give up the current game |
| `get_replay_path()` | Path to the replay file |
| `get_terrain_at(cell_x, cell_y)` | Terrain type at a cell |

### Planning Phase (optional)

| Tool | Purpose |
|------|---------|
| `start_planning_phase()` | Begin pre-game strategy planning |
| `get_opponent_intel()` | AI opponent profile and counters |
| `end_planning_phase(strategy)` | Commit strategy and start playing |
| `get_planning_status()` | Check planning state |

---

## How to Play (Strategy Guide)

### Step 1: Deploy your MCV

At game start you have a Mobile Construction Vehicle (MCV). Deploy it to create your Construction Yard:

```
1. Call get_units() to find your MCV (type "mcv")
2. Call deploy_unit(mcv_actor_id)
3. Call advance(50) to let it deploy
```

### Step 2: Build your base

Follow this build order:

| Order | Building | Type Code | Cost | Why |
|-------|----------|-----------|------|-----|
| 1 | Power Plant | `powr` | $300 | Powers everything |
| 2 | Barracks | `tent` (Allied) or `barr` (Soviet) | $300 | Infantry production |
| 3 | Ore Refinery | `proc` | $2000 | Income + free harvester |
| 4 | War Factory | `weap` | $2000 | Vehicle production (requires Refinery) |
| 5 | More Power | `powr` | $300 | Keep power positive |

Use `build_and_place()` — it auto-places when construction finishes:

```
1. Call get_valid_placements("powr") to find a good spot
2. Call build_and_place("powr", cell_x, cell_y)
3. Call advance(250) to let it build (~10 seconds)
4. Check get_production() to confirm completion
5. Repeat for next building
```

**Important**: Your faction may be Allied OR Soviet. Check `get_game_state()` → `faction` field. Barracks type depends on faction.

### Step 3: Train your army

```
1. Call build_unit("e1", 5) for 5 Rifle Infantry ($100 each)
2. Call advance(100) to let them train
3. Once War Factory is ready: build_unit("3tnk", 3) for Medium Tanks ($800 each)
4. Set rally point near base exit: set_rally_point(barracks_id, x, y)
```

**Key units by faction:**

| Unit | Code | Cost | Role |
|------|------|------|------|
| Rifle Infantry | `e1` | $100 | Cheap, fast |
| Rocket Soldier | `e3` | $300 | Anti-armor |
| Medium Tank | `3tnk` | $800 | Main battle tank |
| Heavy Tank | `4tnk` | $950 | Soviet heavy armor |
| Light Tank | `1tnk` | $700 | Fast flanker |
| Artillery | `arty` | $600 | Long range |
| V2 Launcher | `v2rl` | $700 | Soviet long range |

### Step 4: Scout the map

Send a cheap unit to explore:

```
1. Train one Rifle Infantry
2. Call attack_move([unit_id], far_x, far_y) toward unexplored areas
3. Call advance(500) to let it travel
4. Call get_enemies() to see what you've found
```

### Step 5: Attack the enemy

Once you have 8-10 combat units:

```
1. Call get_enemies() to find enemy buildings
2. Call attack_move(all_unit_ids, enemy_base_x, enemy_base_y)
3. Call advance(100), check get_game_state() for battle progress
4. If enemies visible: attack_target(unit_ids, enemy_id) to focus fire
5. Keep producing reinforcements while attacking
```

### Step 6: Macro (ongoing economy)

Throughout the game:
- Keep power positive (build Power Plants when needed)
- Keep producing units — never let production idle
- Build additional Ore Refineries for more income
- Replace lost harvesters

---

## Game Loop Pattern

A good agent loop looks like this:

```
1. get_game_state() → read the situation
2. Decide what to do based on:
   - Economy: enough cash? Power positive?
   - Production: anything building? Queue empty?
   - Military: under attack? Ready to attack?
   - Exploration: enemy found yet?
3. Issue orders (build, move, attack)
4. advance(50-250) → let time pass
5. Repeat until game is won or lost
```

Check `get_game_state()` → `done` field. When true, `result` will be `"win"` or `"loss"`.

---

## Tips

- **Always call `advance()`** after issuing orders. Orders don't execute until game time passes.
- **Use `batch()`** to issue multiple orders in one tick (e.g., build + move + set rally).
- **Check `available_production`** before building — it lists what you CAN build right now.
- **Don't let production idle** — keep queuing units. Idle production wastes time.
- **Build near your Construction Yard** — buildings must be placed adjacent to existing structures.
- **Power matters** — if power goes negative, production slows to a crawl.
- **Use `attack_move`** instead of `move` when heading toward enemies — units will engage threats.
- **A completed building blocks the queue** until placed. Always use `build_and_place()` to avoid this.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Server not running | `openra-rl server start` (needs Docker) |
| Can't build anything | Deploy MCV first with `deploy_unit()` |
| Building won't place | Use `get_valid_placements()` for valid spots |
| No money | Build Ore Refinery (`proc`) for harvesters |
| Production slow | Check power with `get_economy()` — build Power Plants |
| Can't find enemy | Scout with `attack_move` to unexplored quadrants |

## Links

- **GitHub**: https://github.com/yxc20089/OpenRA-RL
- **PyPI**: https://pypi.org/project/openra-rl/
- **Leaderboard**: https://huggingface.co/spaces/yxc20089/OpenRA-Bench
- **Discord**: https://discord.gg/openra-rl
