# OpenRA-RL Examples

## Scripted Bot

A hardcoded Red Alert bot that plays a full game through the OpenEnv client API.

**Strategy:** Deploy MCV → Build Power Plant → Build Barracks → Train 5 Rifle Infantry → Attack-move toward enemy.

### Prerequisites

```bash
# Install the project
pip install -e .

# Start the OpenRA-RL server (Docker)
docker run -p 8000:8000 openra-rl

# Or build from source first:
OPENRA_DIR=/path/to/OpenRA ./docker/build.sh
docker run -p 8000:8000 openra-rl
```

### Run

```bash
# Basic run
python examples/scripted_bot.py

# Custom server URL
python examples/scripted_bot.py --url http://localhost:8000

# Verbose mode (prints every bot decision)
python examples/scripted_bot.py --verbose

# Limit episode length
python examples/scripted_bot.py --max-steps 2000
```

### Output

```
Connecting to http://localhost:8000...
Game started! Map: singles
Step    0 | Tick     0 | $ 5000 | Units: 2 (combat: 0) | Buildings: [none] | Phase: deploy_mcv
Step  100 | Tick   100 | $ 4700 | Units: 1 (combat: 0) | Buildings: [fact] | Phase: build_base
Step  200 | Tick   200 | $ 4100 | Units: 1 (combat: 0) | Buildings: [fact, powr] | Phase: build_base
...
Game over: win after 3421 steps (tick 3421)
Total reward: 2.150
```

---

## MCP Bot

A hardcoded Red Alert bot that plays a full game through MCP tool calls (the same interface available to LLM agents).

**Strategy:** Look up tech tree & faction info → Deploy MCV → Build Power Plant → Build Barracks → Build Refinery → Build War Factory → Train infantry + vehicles → Attack-move toward enemy with sustain management.

### Prerequisites

Uncomment the following two lines in your `config.yaml` to enable all MCP tools this bot uses:

```yaml
# ── MCP Tools ─────────────────────────────────────────────────────────
tools:
  disabled: []
```

Then restart the server so the config takes effect.

```bash
# Start the OpenRA-RL server (Docker)
docker run -p 8000:8000 openra-rl

# Or from local:
python openra_env/server/app.py
```

### Run

```bash
# Basic run
python examples/mcp_bot.py

# Custom server URL
python examples/mcp_bot.py --url http://localhost:8000

# Verbose mode (prints every bot decision)
python examples/mcp_bot.py --verbose

# Limit turn count
python examples/mcp_bot.py --max-turns 2000

# Disable planning phase (for comparison runs)
python examples/mcp_bot.py --no-planning
```

### Output

```
Connecting to http://localhost:8000...
Resetting environment (launching OpenRA)...
Discovered 48 MCP tools: ['add_to_group', 'advance', 'assign_group', ...]
  [MCPBot] === Startup: Planning Phase ===
...
======================================================================
Game finished after 2850 turns
Result: WIN

--- SCORECARD ---
...

Tools exercised: 24/48
  ['advance', 'attack_move', 'build_structure', ...]
======================================================================
```

---