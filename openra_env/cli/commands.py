"""Subcommand implementations for the openra-rl CLI."""

import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional

from openra_env.cli.console import dim, error, header, info, step, success, warn
from openra_env.cli import docker_manager as docker
from openra_env.cli.wizard import (
    CONFIG_PATH,
    has_saved_config,
    load_saved_config,
    merge_cli_into_config,
    run_wizard,
)


def cmd_play(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    difficulty: str = "normal",
    verbose: bool = False,
    port: int = 8000,
    server_url: Optional[str] = None,
    local: bool = False,
    image_version: Optional[str] = None,
) -> None:
    """Run the LLM agent against the game server."""
    use_docker = server_url is None and not local

    # 1. Check Docker (unless --local or --server-url)
    if use_docker and not docker.check_docker():
        sys.exit(1)

    # 1b. Version selection — let user pick if multiple versions exist locally
    if use_docker and image_version is None:
        versions = docker.list_local_versions()
        # Filter out "latest" for display — only show concrete version tags
        concrete = [v for v in versions if v != "latest"]
        if len(concrete) > 1:
            info(f"Multiple engine versions available: {', '.join(concrete)}")
            try:
                choice = input(f"  Version to use [{concrete[0]}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                choice = ""
            if choice:
                image_version = choice
            else:
                image_version = concrete[0]

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
    is_local_llm = any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
    if not llm_cfg.get("api_key") and not is_local_llm:
        error("No API key configured. Run `openra-rl config` or pass --api-key.")
        sys.exit(1)
    if not llm_cfg.get("model"):
        error("No model configured. Run `openra-rl config` or pass --model.")
        sys.exit(1)

    # 3. Start/reuse server
    actual_url = server_url or f"http://localhost:{port}"
    we_started_server = False
    local_server_proc = None

    if local:
        # Run the server locally (for developers with local OpenRA build)
        header("Starting local server...")
        local_server_proc = subprocess.Popen(
            [sys.executable, "-m", "openra_env.server.app"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        we_started_server = True
        # Wait for it to be ready
        import time
        import urllib.request
        import urllib.error
        step(f"Waiting for local server on port {port}...")
        start = time.time()
        while time.time() - start < 60:
            try:
                req = urllib.request.urlopen(f"{actual_url}/health", timeout=3)
                if req.status == 200:
                    success("Local server is ready!")
                    break
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        else:
            error("Local server did not become ready within 60s.")
            local_server_proc.terminate()
            sys.exit(1)
    elif use_docker:
        if docker.is_running():
            info(f"Server already running on port {port}.")
        else:
            if not docker.start_server(port=port, difficulty=difficulty, version=image_version):
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
        error(f"Could not connect to {actual_url}.")
        info("Try: openra-rl server start")
        info("Check: openra-rl doctor")
    except Exception as e:
        error(f"Agent error: {e}")
        info("Run with --verbose for full details, or check: openra-rl doctor")

    # 5. Auto-copy replays from Docker
    if use_docker and docker.is_running():
        new_replays = docker.copy_replays()
        if new_replays:
            print()
            for f in new_replays:
                success(f"Replay saved: {docker.LOCAL_REPLAY_DIR / f}")
            info("Watch with: openra-rl replay watch")

    # 6. Cleanup
    if we_started_server:
        print()
        if local_server_proc:
            try:
                answer = input("  Stop local server? [Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "y"
            if answer in ("", "y", "yes"):
                local_server_proc.terminate()
                local_server_proc.wait(timeout=10)
                success("Local server stopped.")
        elif use_docker:
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


# ── Replay commands ──────────────────────────────────────────────────


def cmd_replay_watch(
    file: Optional[str] = None,
    port: int = 6080,
    resolution: Optional[str] = None,
    render_mode: Optional[str] = None,
    vnc_quality: Optional[int] = None,
    vnc_compression: Optional[int] = None,
    cpu_cores: Optional[int] = None,
) -> None:
    """Watch a replay in the browser via VNC-in-Docker."""
    if not docker.check_docker():
        sys.exit(1)

    try:
        viewer_settings = docker.load_replay_viewer_settings(
            resolution=resolution,
            render_mode=render_mode,
            vnc_quality=vnc_quality,
            vnc_compression=vnc_compression,
            cpu_cores=cpu_cores,
        )
    except ValueError as exc:
        error(f"Invalid replay viewer setting: {exc}")
        sys.exit(1)

    replay_path = file

    if replay_path is None:
        # Check local replays first (most reliable — file is mounted directly)
        local_replays = sorted(docker.LOCAL_REPLAY_DIR.glob("*.orarep"))
        if local_replays:
            replay_path = str(local_replays[-1])
            info(f"Latest local replay: {local_replays[-1].name}")
        elif docker.is_running():
            # Fall back to container path (uses --volumes-from, less reliable)
            replay_path = docker.get_latest_replay()
            if replay_path:
                info(f"Latest container replay: {Path(replay_path).name}")
        if replay_path is None:
            error("No replays found. Play a game first with: openra-rl play")
            sys.exit(1)

    header("Starting replay viewer...")
    info(
        f"Settings: {viewer_settings.width}x{viewer_settings.height}, "
        f"render={viewer_settings.render_mode}, "
        f"vnc q/c={viewer_settings.vnc_quality}/{viewer_settings.vnc_compression}"
    )

    if not docker.start_replay_viewer(replay_path, port=port, settings=viewer_settings):
        sys.exit(1)

    import time
    import urllib.error
    import urllib.request

    url = (
        f"http://localhost:{port}/vnc.html?autoconnect=1&resize=scale"
        f"&quality={viewer_settings.vnc_quality}"
        f"&compression={viewer_settings.vnc_compression}"
    )
    step("Waiting for viewer to be ready...")

    ready = False
    start_time = time.time()
    timeout = 30
    while time.time() - start_time < timeout:
        if not docker.is_replay_viewer_running():
            error("Replay viewer exited before it became ready.")
            logs = docker.get_replay_viewer_logs()
            if logs:
                print()
                info("Replay viewer logs:")
                print(logs)
            sys.exit(1)
        try:
            req = urllib.request.urlopen(f"http://localhost:{port}/vnc.html", timeout=2)
            if 200 <= req.status < 500:
                ready = True
                break
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)

    if not ready:
        error(f"Viewer did not become ready within {timeout}s.")
        logs = docker.get_replay_viewer_logs()
        if logs:
            print()
            info("Replay viewer logs:")
            print(logs)
        sys.exit(1)

    info(f"Opening {url}")
    webbrowser.open(url)
    print()
    info("Tip: press F12 in the viewer for maximum replay speed.")
    info("Tip: tune with --resolution, --render, --vnc-quality, --vnc-compression.")
    info("Press Ctrl+C to stop the replay viewer")
    print()

    try:
        # Wait until container exits or user presses Ctrl+C
        while docker.is_replay_viewer_running():
            time.sleep(2)
        info("Replay viewer has stopped.")
    except KeyboardInterrupt:
        print()
        docker.stop_replay_viewer()


def cmd_replay_list() -> None:
    """List available replays from Docker and local."""
    header("Game Replays")

    # Docker replays
    if docker.is_running():
        docker_replays = docker.list_replays()
        if docker_replays:
            info(f"In Docker container ({len(docker_replays)}):")
            for r in docker_replays:
                dim(f"    {Path(r).name}")
        else:
            dim("  No replays in Docker container.")
    else:
        dim("  Docker server not running — cannot list container replays.")

    # Local replays
    print()
    local_dir = docker.LOCAL_REPLAY_DIR
    if local_dir.exists():
        local_replays = sorted(local_dir.glob("*.orarep"))
        if local_replays:
            info(f"Local ({len(local_replays)}) — {local_dir}:")
            for r in local_replays:
                dim(f"    {r.name}")
        else:
            dim(f"  No local replays in {local_dir}")
    else:
        dim(f"  No local replay directory ({local_dir})")


def cmd_replay_copy() -> None:
    """Copy replays from Docker container to local directory."""
    if not docker.check_docker():
        sys.exit(1)

    if not docker.is_running():
        error("Game server is not running. Start it first or use: openra-rl server start")
        sys.exit(1)

    header("Copying replays from Docker...")
    new_files = docker.copy_replays()
    if new_files:
        for f in new_files:
            success(f"  Copied: {f}")
        success(f"Copied {len(new_files)} new replay(s) to {docker.LOCAL_REPLAY_DIR}")
    else:
        info(f"No new replays to copy. Replays are in {docker.LOCAL_REPLAY_DIR}")


def cmd_replay_stop() -> None:
    """Stop the replay viewer."""
    docker.stop_replay_viewer()
