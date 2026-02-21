"""Subcommand implementations for the openra-rl CLI."""

import shutil
import sys
from typing import Optional

from openra_env.cli.console import dim, error, header, info, step, success, warn
from openra_env.cli import docker_manager as docker
from openra_env.cli.wizard import (
    CONFIG_PATH,
    has_saved_config,
    load_saved_config,
    merge_cli_into_config,
    run_wizard,
    save_config,
)


def cmd_play(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    difficulty: str = "normal",
    verbose: bool = False,
    port: int = 8000,
    server_url: Optional[str] = None,
) -> None:
    """Run the LLM agent against the game server."""
    # 1. Check Docker
    if server_url is None and not docker.check_docker():
        sys.exit(1)

    # 2. Load or create config
    has_cli_overrides = any([provider, model, api_key])

    if has_cli_overrides:
        config = load_saved_config() or {}
        config = merge_cli_into_config(config, provider=provider, model=model, api_key=api_key)
    elif has_saved_config():
        config = load_saved_config() or {}
    else:
        config = run_wizard()

    # Validate we have enough config to proceed
    llm_cfg = config.get("llm", {})
    base_url = llm_cfg.get("base_url", "")
    is_local = any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
    if not llm_cfg.get("api_key") and not is_local:
        error("No API key configured. Run `openra-rl config` or pass --api-key.")
        sys.exit(1)
    if not llm_cfg.get("model"):
        error("No model configured. Run `openra-rl config` or pass --model.")
        sys.exit(1)

    # 3. Start/reuse server
    actual_url = server_url or f"http://localhost:{port}"
    we_started_server = False

    if server_url is None:
        if docker.is_running():
            info(f"Server already running on port {port}.")
        else:
            if not docker.start_server(port=port, difficulty=difficulty):
                sys.exit(1)
            we_started_server = True
            if not docker.wait_for_health(port=port):
                sys.exit(1)

    # 4. Run the LLM agent
    header("Starting LLM agent...")
    provider_name = config.get("provider", "custom")
    info(f"Model: {llm_cfg.get('model', '?')} via {provider_name}")
    print()

    try:
        _run_llm_agent(config, actual_url, verbose)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except ConnectionRefusedError:
        error(f"Could not connect to {actual_url}. Is the server running?")
    except Exception as e:
        error(f"Agent error: {e}")

    # 5. Cleanup
    if we_started_server:
        print()
        try:
            answer = input("  Stop game server? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "y"
        if answer in ("", "y", "yes"):
            docker.stop_server()


def _run_llm_agent(config: dict, server_url: str, verbose: bool) -> None:
    """Import and run the LLM agent with the given config."""
    import asyncio

    from openra_env.config import load_config

    # Build overrides from saved config
    cli_overrides: dict = {}
    llm_cfg = config.get("llm", {})
    if llm_cfg:
        cli_overrides["llm"] = llm_cfg
    cli_overrides.setdefault("agent", {})["server_url"] = server_url
    if verbose:
        cli_overrides.setdefault("agent", {})["verbose"] = True

    app_config = load_config(cli_overrides=cli_overrides)

    from openra_env.agent import run_agent
    asyncio.run(run_agent(app_config, verbose))


def cmd_config() -> None:
    """Re-run the setup wizard."""
    run_wizard()


def cmd_server_start(port: int = 8000, difficulty: str = "normal", detach: bool = True) -> None:
    """Start the game server."""
    if not docker.check_docker():
        sys.exit(1)
    if not docker.start_server(port=port, difficulty=difficulty, detach=detach):
        sys.exit(1)
    if detach:
        docker.wait_for_health(port=port)


def cmd_server_stop() -> None:
    """Stop the game server."""
    docker.stop_server()


def cmd_server_status() -> None:
    """Show game server status."""
    status = docker.server_status()
    if status:
        success(f"Server is running: {status['status']}")
        if status.get("ports"):
            dim(f"  Ports: {status['ports']}")
    else:
        info("Server is not running.")


def cmd_server_logs(follow: bool = False) -> None:
    """Show game server logs."""
    docker.get_logs(follow=follow)


def cmd_doctor() -> None:
    """Check system prerequisites."""
    header("OpenRA-RL Doctor")
    ok = True

    # Docker
    if shutil.which("docker"):
        success("Docker CLI: installed")
        from openra_env.cli.docker_manager import _run
        result = _run(["docker", "info"])
        if result.returncode == 0:
            success("Docker daemon: running")
        else:
            warn("Docker daemon: not running")
            ok = False
    else:
        error("Docker CLI: not found")
        dim("  Install from https://docs.docker.com/get-docker/")
        ok = False

    # Image
    if docker.image_exists():
        success(f"Game image: available ({docker.IMAGE})")
    else:
        warn("Game image: not pulled yet (will be pulled on first `openra-rl play`)")

    # Server
    if docker.is_running():
        success("Game server: running")
    else:
        dim("Game server: not running")

    # Python
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        success(f"Python: {py_version}")
    else:
        error(f"Python: {py_version} (requires 3.10+)")
        ok = False

    # Saved config
    if has_saved_config():
        cfg = load_saved_config() or {}
        provider = cfg.get("provider", "unknown")
        model = cfg.get("llm", {}).get("model", "unknown")
        success(f"Config: {CONFIG_PATH}")
        dim(f"  Provider: {provider}, Model: {model}")
    else:
        dim("Config: not yet configured (run `openra-rl play` or `openra-rl config`)")

    print()
    if ok:
        success("All checks passed!")
    else:
        warn("Some checks failed. Fix the issues above and try again.")


def cmd_version() -> None:
    """Print version."""
    try:
        from importlib.metadata import version
        v = version("openra-rl")
    except Exception:
        v = "dev"
    print(f"openra-rl {v}")


def cmd_mcp_server(server_url: Optional[str] = None, port: int = 8000) -> None:
    """Start the MCP stdio server."""
    from openra_env.mcp_server import main as mcp_main
    mcp_main(server_url=server_url or f"http://localhost:{port}")
