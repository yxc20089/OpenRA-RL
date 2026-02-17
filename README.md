# OpenRA-RL

A Gymnasium-style Reinforcement Learning environment for [OpenRA](https://www.openra.net/), the open-source Red Alert / Command & Conquer RTS engine. The game engine is exposed as an RL-friendly HTTP/gRPC API via a FastAPI server running inside a headless Docker container.

## Architecture

```
┌──────────────┐       HTTP :8000        ┌─────────────────────────────┐
│  RL Agent /  │  ◄──────────────────►   │  OpenRA-RL Server (Docker)  │
│  LLM Agent   │       gRPC :9999        │  FastAPI + gRPC bridge      │
│  MCP Bot     │  ◄──────────────────►   │  OpenRA engine (headless)   │
└──────────────┘                         └─────────────────────────────┘
```

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — **Windows users:** ensure the WSL 2 backend is enabled (Settings → General → Use the WSL 2 based engine)
- Python 3.10+ (for running agents locally)
- Git (with submodules — the `OpenRA/` directory is a submodule)

> [!NOTE]
> The game server runs inside a **Linux container**. On Windows, Docker Desktop handles the Linux VM automatically via WSL 2 — no manual VM setup is needed.

## Quick Start (Windows)

All commands below are for **PowerShell**. If you prefer WSL / Git Bash, replace the syntax accordingly.

### 1. Clone the repository

```powershell
git clone --recurse-submodules https://github.com/huixu11/OpenRA-RL.git
cd OpenRA-RL
```

If you already cloned without `--recurse-submodules`, initialize the submodule now:

```powershell
git submodule update --init --recursive
```

### 2. Build the Docker images

**Game server image** (compiles the full OpenRA engine — first build takes 10–20 min):

```powershell
docker build -t openra-rl .
```

This multi-stage build will:
- Compile OpenRA from source (C# / .NET 8.0)
- Install the Python `openra_env` package and its dependencies
- Produce a final runtime image with Xvfb for headless rendering

**Agent image** (lightweight — only the Python client and agent scripts):

```powershell
docker compose build
```

> [!TIP]
> Subsequent rebuilds use Docker layer caching and are much faster. You only need to rebuild the game server image when the `OpenRA` submodule or server code changes.

### 3. Run an agent (pick one)

Three example agents are included. Each option below is self-contained — pick one.

**Option A: Scripted bot** (recommended to test first, no API key needed):

This option runs the Python bot **locally** (outside Docker), so you need to start the game server first:

```powershell
# Terminal 1: Start the game server
docker compose up openra-rl
# Wait for health check to pass — verify at http://localhost:8000/health

# Terminal 2: Run the bot
pip install -e .
python examples/scripted_bot.py --url http://localhost:8000 --verbose
```

**Option B: LLM agent** (uses OpenRouter, requires API key):

> [!NOTE]
> This command auto-starts the `openra-rl` game server via Docker Compose `depends_on` — no need to run it separately.

```powershell
$env:OPENROUTER_API_KEY = "sk-or-your-key-here"
# Optional: customize model and game settings
$env:OPENROUTER_MODEL = "qwen/qwen3-coder-next"
# $env:MAX_TURNS = "200"
# $env:AI_SLOT = "Multi0"
# $env:BOT_TYPE = "normal"
docker compose up --build agent
```

**Option C: MCP bot** (model-context-protocol bot, no API key needed):

> [!NOTE]
> Also auto-starts the game server — single command, no separate server needed.

```powershell
docker compose run --build mcp-bot
```

See [examples/README.md](examples/README.md) for more details on the scripted bot.

### 5. View game replays

Each game session saves a `.orarep` replay inside the Docker container. Copy them to your local machine:

```powershell
docker cp openra-rl-openra-rl-1:/root/.config/openra/Replays ./replays
```

> **Tip:** Add a volume mount in `docker-compose.yaml` to auto-export replays automatically:
> ```yaml
> volumes:
>   - ./replays:/root/.config/openra/Replays
> ```

> [!WARNING]
> Replays are saved under `{DEV_VERSION}` and **cannot** be opened in the
> standard [OpenRA](https://www.openra.net/) desktop release (which uses `v2`).
> You must build the dev version locally to view them.

**Build the dev client & watch replays:**

```powershell
# Step 1: Install .NET 8 SDK
winget install Microsoft.DotNet.SDK.8

# Step 2: Refresh PATH so the current terminal can find 'dotnet'
#   Option A (recommended): Close and reopen your PowerShell window
#   Option B (no restart): Run the following line to reload PATH:
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")

# Step 3: Verify installation
dotnet --version   # should print 8.x.xxx

# Step 4: Build the dev client
cd OpenRA
powershell -ExecutionPolicy Bypass -File make.ps1 all

# Step 5: Copy replays into the dev client's support directory
#   The Replay Browser looks in: %APPDATA%\OpenRA\Replays\ra\{DEV_VERSION}\
New-Item -ItemType Directory -Path "$env:APPDATA\OpenRA\Replays\ra\{DEV_VERSION}" -Force
Copy-Item ".\replays\Replays\ra\{DEV_VERSION}\*.orarep" "$env:APPDATA\OpenRA\Replays\ra\{DEV_VERSION}\"

# Step 6: Launch the game and open Extras → Replays
./launch-game.cmd Game.Mod=ra
```

**Clean up old replays** (run from the project root `OpenRA-RL/`):

```powershell
# Clear locally exported replays
Remove-Item -Recurse -Force ".\replays\*"

# Clear replays from the dev client's support directory
Remove-Item -Recurse -Force "$env:APPDATA\OpenRA\Replays\ra\{DEV_VERSION}\*"
```

## Docker Compose Services

| Service | Command | Description |
|---------|---------|-------------|
| `openra-rl` | `docker compose up openra-rl` | Headless game server |
| `agent` | `docker compose up agent` | LLM agent (needs `OPENROUTER_API_KEY`) |
| `mcp-bot` | `docker compose run mcp-bot` | MCP bot |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | API key for LLM agent |
| `OPENROUTER_MODEL` | `anthropic/claude-sonnet-4-20250514` | Model for LLM agent |
| `MAX_TURNS` | `200` (agent) / `3000` (bot) | Max game turns |
| `AI_SLOT` | `Multi0` | AI player slot |
| `BOT_TYPE` | `normal` | Built-in bot difficulty |

Setting environment variables in PowerShell:

```powershell
# Temporary (current session only)
$env:OPENROUTER_API_KEY = "sk-or-your-key-here"

# Or pass inline to docker compose
docker compose run -e OPENROUTER_API_KEY="sk-or-xxx" agent
```

## Local Development

If you only need the Python client (without the game engine):

```powershell
# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Install training extras (PyTorch, TRL, Transformers)
pip install -e ".[training]"
```

> [!IMPORTANT]
> If running `Activate.ps1` fails due to execution policy, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

## Project Structure

```
OpenRA-RL/
├── OpenRA/              # OpenRA engine (git submodule)
├── openra_env/          # Python package
│   ├── client.py        #   OpenEnv HTTP client
│   ├── models.py        #   Pydantic data models
│   ├── reward.py        #   Reward shaping
│   ├── game_data.py     #   Game state parsing
│   ├── mcp_ws_client.py #   WebSocket MCP client
│   └── server/          #   FastAPI + gRPC server
│       ├── app.py       #     Application entry point
│       ├── bridge_client.py   # gRPC bridge client
│       ├── openra_environment.py  # OpenEnv environment
│       └── openra_process.py      # OpenRA subprocess manager
├── examples/            # Example agents
│   ├── scripted_bot.py  #   Hardcoded strategy bot
│   ├── llm_agent.py     #   LLM-powered agent
│   └── mcp_bot.py       #   MCP-protocol bot
├── proto/               # Protobuf definitions
├── scripts/             # Utility scripts
├── tests/               # Test suite
├── docker-compose.yaml  # Service orchestration
├── Dockerfile           # Game server image
├── Dockerfile.agent     # Lightweight agent image
└── openenv.yaml         # OpenEnv configuration
```

## Troubleshooting (Windows)

| Problem | Solution |
|---------|----------|
| `docker compose build` fails with path errors | Make sure Docker Desktop is running and WSL 2 backend is enabled |
| Port 8000 already in use | Stop conflicting services: `netstat -ano \| findstr :8000` then `Stop-Process -Id <PID>` |
| `pip install -e .` fails | Ensure you're in a virtual environment and Python ≥ 3.10: `python --version` |
| Health check never passes | Check server logs: `docker compose logs openra-rl` |
| `Activate.ps1` cannot be loaded | Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Git submodule `OpenRA/` is empty | Run: `git submodule update --init --recursive` |

## License

[GPL-3.0](LICENSE)
