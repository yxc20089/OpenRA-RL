# OpenRA-RL

Play [Red Alert](https://www.openra.net/) with AI agents. LLMs, scripted bots, or RL — your agent commands armies in the classic RTS through a Python API.

```
┌──────────────────┐       HTTP / WS :8000       ┌──────────────────────────────┐
│   Your Agent     │  ◄────────────────────────►  │  OpenRA-RL Server (Docker)   │
│                  │       gRPC :9999             │  FastAPI + gRPC bridge       │
│  LLM / Bot / RL  │  ◄────────────────────────►  │  OpenRA engine (headless)    │
└──────────────────┘                              └──────────────────────────────┘
```

## Quick Start

```bash
pip install openra-rl
openra-rl play
```

On first run, an interactive wizard helps you configure your LLM provider (OpenRouter, Ollama, or LM Studio). The CLI pulls the game server Docker image and starts everything automatically.

### Skip the wizard

```bash
# Cloud (OpenRouter)
openra-rl play --provider openrouter --api-key sk-or-... --model anthropic/claude-sonnet-4-20250514

# Local (Ollama — free, no API key)
openra-rl play --provider ollama --model qwen3:32b

# Developer mode (skip Docker, run server locally)
openra-rl play --local --provider ollama --model qwen3:32b

# Reconfigure later
openra-rl config
```

### Prerequisites

- **Docker** — the game server runs in a container
- **Python 3.10+**
- An LLM endpoint (cloud API key or local model server)

## CLI Reference

```
openra-rl play       Run the LLM agent (wizard on first use)
openra-rl config     Re-run the setup wizard
openra-rl server     start | stop | status | logs
openra-rl replay     watch | list | copy | stop
openra-rl mcp-server Start MCP stdio server (for OpenClaw / Claude Desktop)
openra-rl doctor     Check system prerequisites
openra-rl version    Print version
```

## MCP Server (OpenClaw / Claude Desktop)

OpenRA-RL exposes all 48 game tools as a standard MCP server:

```bash
openra-rl mcp-server
```

Add to your MCP client config (e.g. `~/.openclaw/openclaw.json`):

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

Then chat: _"Start a game of Red Alert on easy difficulty, build a base, and defeat the enemy."_

## Architecture

| Component | Language | Role |
|-----------|----------|------|
| **OpenRA-RL** | Python | Environment wrapper, agents, HTTP/WebSocket API |
| **OpenRA** (submodule) | C# | Modified game engine with embedded gRPC server |
| **OpenEnv** (pip dep) | Python | Standardized Gymnasium-style environment interface |

**Data flow:** Agent <-> FastAPI (port 8000) <-> gRPC bridge (port 9999) <-> OpenRA game engine

The game runs at ~25 ticks/sec independent of agent speed. Observations use a DropOldest channel so the agent always sees the latest game state, even if it's slower than real time.

## Example Agents

### Scripted Bot

A hardcoded state-machine bot that demonstrates all action types. Deploys MCV, builds a base, trains infantry, and attacks.

```bash
python examples/scripted_bot.py --url http://localhost:8000 --verbose --max-steps 2000
```

### MCP Bot

A planning-aware bot that uses game knowledge tools (tech tree lookups, faction briefings, map analysis) to formulate strategy before playing.

```bash
python examples/mcp_bot.py --url http://localhost:8000 --verbose --max-turns 3000
```

### LLM Agent

An AI agent powered by any OpenAI-compatible model. Supports cloud APIs (OpenRouter, OpenAI) and local model servers (Ollama, LM Studio).

```bash
python examples/llm_agent.py \
  --config examples/config-openrouter.yaml \
  --api-key sk-or-... \
  --verbose \
  --log-file game.log
```

CLI flags override config file values. See `python examples/llm_agent.py --help` for all options.

## Configuration

OpenRA-RL uses a unified YAML config system. Settings are resolved with this precedence:

**CLI flags > Environment variables > Config file > Built-in defaults**

### Config file

Copy and edit the default config:

```bash
cp config.yaml my-config.yaml
# Edit my-config.yaml, then:
python examples/llm_agent.py --config my-config.yaml
```

Key sections:

```yaml
game:
  openra_path: "/opt/openra"      # Path to OpenRA installation
  map_name: "singles.oramap"      # Map to play
  headless: true                  # No GPU rendering
  record_replays: false           # Save .orarep replay files

opponent:
  bot_type: "normal"              # AI difficulty: easy, normal, hard
  ai_slot: "Multi0"              # AI player slot

planning:
  enabled: true                   # Pre-game planning phase
  max_turns: 10                   # Max planning turns
  max_time_s: 60.0                # Planning time limit

llm:
  base_url: "https://openrouter.ai/api/v1/chat/completions"
  model: "qwen/qwen3-coder-next"
  max_tokens: 1500
  temperature: null               # null = provider default

tools:
  categories:                     # Toggle tool groups on/off
    read: true
    knowledge: true
    movement: true
    production: true
    # ... see config.yaml for all categories
  disabled: []                    # Disable specific tools by name

alerts:
  under_attack: true
  low_power: true
  idle_production: true
  no_scouting: true
  # ... see config.yaml for all alerts
```

### Example configs

| File | Use case |
|------|----------|
| `examples/config-openrouter.yaml` | Cloud LLM via OpenRouter (Claude, GPT, etc.) |
| `examples/config-ollama.yaml` | Local LLM via Ollama |
| `examples/config-lmstudio.yaml` | Local LLM via LM Studio |
| `examples/config-minimal.yaml` | Reduced tool set for limited-context models |

### Environment variables

| Variable | Config path | Description |
|----------|-------------|-------------|
| `OPENROUTER_API_KEY` | `llm.api_key` | API key for OpenRouter |
| `LLM_API_KEY` | `llm.api_key` | Generic LLM API key (overrides OpenRouter key) |
| `LLM_BASE_URL` | `llm.base_url` | LLM endpoint URL |
| `LLM_MODEL` | `llm.model` | Model identifier |
| `BOT_TYPE` | `opponent.bot_type` | AI difficulty: easy, normal, hard |
| `OPENRA_PATH` | `game.openra_path` | Path to OpenRA installation |
| `RECORD_REPLAYS` | `game.record_replays` | Save replay files (true/false) |
| `PLANNING_ENABLED` | `planning.enabled` | Enable planning phase (true/false) |

## Using Local Models

### Ollama

```bash
# Pull a model with tool-calling support
ollama pull qwen3:32b

# For models that need more context (default is often 2048-4096 tokens):
cat > /tmp/Modelfile <<EOF
FROM qwen3:32b
PARAMETER num_ctx 32768
EOF
ollama create qwen3-32k -f /tmp/Modelfile

# Run
openra-rl play --provider ollama --model qwen3-32k
```

> **Note:** Not all Ollama models support tool calling. Check with `ollama show <model>` — the template must include a `tools` block. Models known to work: `qwen3:32b`, `qwen3:4b`.

### LM Studio

1. Load a model in LM Studio and start the local server (default port 1234)
2. Run:

```bash
openra-rl play --provider lmstudio --model <model-name>
```

## Docker

### Server management

```bash
openra-rl server start              # Start game server container
openra-rl server start --port 9000  # Custom port
openra-rl server status             # Check if running
openra-rl server logs --follow      # Tail logs
openra-rl server stop               # Stop container
```

### Docker Compose (development)

| Service | Command | Description |
|---------|---------|-------------|
| `openra-rl` | `docker compose up openra-rl` | Headless game server (ports 8000, 9999) |
| `agent` | `docker compose up agent` | LLM agent (requires `OPENROUTER_API_KEY`) |
| `mcp-bot` | `docker compose run mcp-bot` | MCP bot |

```bash
# LLM agent via Docker Compose
OPENROUTER_API_KEY=sk-or-... docker compose up agent
```

### Replays

After each game, replays are automatically copied to `~/.openra-rl/replays/`. Watch them in your browser:

```bash
openra-rl replay watch              # Watch the latest replay (opens browser via VNC)
openra-rl replay watch <file>       # Watch a specific .orarep file
openra-rl replay list               # List replays (Docker + local)
openra-rl replay copy               # Copy replays from Docker to local
openra-rl replay stop               # Stop the replay viewer
```

The replay viewer runs inside Docker using the same engine that recorded the game, so replays always play back correctly. The browser connects via noVNC — no local game install needed.

> **Version tracking:** Each replay records which Docker image version was used. When you upgrade, old replays are still viewable using their original engine version.

## Local Development (without Docker)

For running the game server natively (macOS/Linux):

### Install dependencies

```bash
# Python
pip install -e ".[dev]"

# .NET 8.0 SDK
# macOS: brew install dotnet@8
# Ubuntu: sudo apt install dotnet-sdk-8.0

# Native libraries (macOS arm64)
brew install sdl2 openal-soft freetype luajit
cp $(brew --prefix sdl2)/lib/libSDL2.dylib OpenRA/bin/SDL2.dylib
cp $(brew --prefix openal-soft)/lib/libopenal.dylib OpenRA/bin/soft_oal.dylib
cp $(brew --prefix freetype)/lib/libfreetype.dylib OpenRA/bin/freetype6.dylib
cp $(brew --prefix luajit)/lib/libluajit-5.1.dylib OpenRA/bin/lua51.dylib
```

### Build OpenRA

```bash
cd OpenRA && make && cd ..
```

### Start the server

```bash
python openra_env/server/app.py
```

### Run tests

```bash
pytest
```

## Observation Space

Each tick, the agent receives structured game state:

| Field | Description |
|-------|-------------|
| `tick` | Current game tick |
| `cash`, `ore`, `power_provided`, `power_drained` | Economy |
| `units` | Own units with position, health, type, facing, stance, speed, attack range |
| `buildings` | Own buildings with production queues, power, rally points |
| `visible_enemies`, `visible_enemy_buildings` | Fog-of-war limited enemy intel |
| `spatial_map` | 9-channel spatial tensor (terrain, height, resources, passability, fog, own buildings, own units, enemy buildings, enemy units) |
| `military` | Kill/death costs, asset value, experience, order count |
| `available_production` | What can currently be built |

## Action Space

18 action types available through the command API:

| Category | Actions |
|----------|---------|
| **Movement** | `move`, `attack_move`, `attack`, `stop` |
| **Production** | `produce`, `cancel_production` |
| **Building** | `place_building`, `sell`, `repair`, `power_down`, `set_rally_point`, `set_primary` |
| **Unit control** | `deploy`, `guard`, `set_stance`, `enter_transport`, `unload`, `harvest` |

## MCP Tools

The LLM agent interacts through 48 MCP (Model Context Protocol) tools organized into categories:

| Category | Tools | Purpose |
|----------|-------|---------|
| **Read** | `get_game_state`, `get_economy`, `get_units`, `get_buildings`, `get_enemies`, `get_production`, `get_map_info`, `get_exploration_status` | Query current game state |
| **Knowledge** | `lookup_unit`, `lookup_building`, `lookup_tech_tree`, `lookup_faction` | Static game data reference |
| **Bulk Knowledge** | `get_faction_briefing`, `get_map_analysis`, `batch_lookup` | Efficient batch queries |
| **Planning** | `start_planning_phase`, `end_planning_phase`, `get_opponent_intel`, `get_planning_status` | Pre-game strategy planning |
| **Game Control** | `advance` | Advance game ticks |
| **Movement** | `move_units`, `attack_move`, `attack_target`, `stop_units` | Unit movement commands |
| **Production** | `build_unit`, `build_structure`, `build_and_place` | Build units and structures |
| **Building Actions** | `place_building`, `cancel_production`, `deploy_unit`, `sell_building`, `repair_building`, `set_rally_point`, `guard_target`, `set_stance`, `harvest`, `power_down`, `set_primary` | Building and unit management |
| **Placement** | `get_valid_placements` | Query valid building locations |
| **Unit Groups** | `assign_group`, `add_to_group`, `get_groups`, `command_group` | Group management |
| **Compound** | `batch`, `plan` | Multi-action sequences |
| **Utility** | `get_replay_path`, `surrender` | Misc |
| **Terrain** | `get_terrain_at` | Terrain queries |

Tools can be toggled per-category or individually via `config.yaml`.

## Project Structure

```
OpenRA-RL/
├── OpenRA/                     # Game engine (git submodule, C#)
├── openra_env/                 # Python package
│   ├── cli/                    #   CLI entry point (openra-rl command)
│   ├── mcp_server.py           #   Standard MCP server (stdio transport)
│   ├── client.py               #   WebSocket client
│   ├── config.py               #   Unified YAML configuration
│   ├── models.py               #   Pydantic data models
│   ├── game_data.py            #   Unit/building stats, tech tree
│   ├── reward.py               #   Multi-component reward function
│   ├── opponent_intel.py       #   AI opponent profiles
│   ├── mcp_ws_client.py        #   MCP WebSocket client
│   ├── server/
│   │   ├── app.py              #     FastAPI application
│   │   ├── openra_environment.py  #  OpenEnv environment (reset/step/state)
│   │   ├── bridge_client.py    #     Async gRPC client
│   │   └── openra_process.py   #     OpenRA subprocess manager
│   └── generated/              #   Auto-generated protobuf stubs
├── examples/
│   ├── scripted_bot.py         #   Hardcoded strategy bot
│   ├── mcp_bot.py              #   MCP tool-based bot
│   ├── llm_agent.py            #   LLM-powered agent
│   └── config-*.yaml           #   Example configs (ollama, lmstudio, openrouter, minimal)
├── skill/                      # OpenClaw skill definition
├── proto/                      # Protobuf definitions (rl_bridge.proto)
├── tests/                      # Test suite
├── .github/workflows/          # CI, Docker publish, PyPI publish
├── config.yaml                 # Default configuration
├── docker-compose.yaml         # Service orchestration
├── Dockerfile                  # Game server image
└── Dockerfile.agent            # Lightweight agent image
```

## License

[GPL-3.0](LICENSE)
