"""Docker orchestration for the OpenRA-RL game server."""

import shutil
import subprocess
import sys
import time
from typing import Optional

from openra_env.cli.console import error, info, step, success

IMAGE = "ghcr.io/yxc20089/openra-rl:latest"
CONTAINER_NAME = "openra-rl-server"


def _run(args: list[str], capture: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess command, capturing output by default."""
    return subprocess.run(
        args,
        capture_output=capture,
        text=True,
        **kwargs,
    )


def check_docker() -> bool:
    """Verify docker CLI is available and daemon is running."""
    if not shutil.which("docker"):
        error("Docker not found. Install it from https://docs.docker.com/get-docker/")
        return False
    result = _run(["docker", "info"])
    if result.returncode != 0:
        error("Docker daemon is not running. Start Docker Desktop and try again.")
        return False
    return True


def pull_image(quiet: bool = False) -> bool:
    """Pull the game server image from GHCR."""
    if not quiet:
        step(f"Pulling game server image ({IMAGE})...")
    result = subprocess.run(
        ["docker", "pull", IMAGE],
        stdout=sys.stdout if not quiet else subprocess.DEVNULL,
        stderr=sys.stderr if not quiet else subprocess.DEVNULL,
    )
    if result.returncode != 0:
        error(f"Failed to pull {IMAGE}")
        return False
    if not quiet:
        success("Image pulled successfully.")
    return True


def image_exists() -> bool:
    """Check if the game server image is available locally."""
    result = _run(["docker", "images", "-q", IMAGE])
    return bool(result.stdout.strip())


def is_running() -> bool:
    """Check if the game server container is running."""
    result = _run([
        "docker", "ps", "--filter", f"name={CONTAINER_NAME}",
        "--format", "{{.Names}}"
    ])
    return CONTAINER_NAME in result.stdout


def start_server(
    port: int = 8000,
    difficulty: str = "normal",
    detach: bool = True,
) -> bool:
    """Start the game server container."""
    if is_running():
        info(f"Server already running on port {port}.")
        return True

    # Ensure image exists
    if not image_exists():
        if not pull_image():
            return False

    step(f"Starting game server on port {port}...")
    cmd = [
        "docker", "run", "--rm",
        "-d" if detach else "",
        "-p", f"{port}:8000",
        "--name", CONTAINER_NAME,
        "-e", f"BOT_TYPE={difficulty}",
        IMAGE,
    ]
    # Remove empty strings from cmd
    cmd = [c for c in cmd if c]

    result = _run(cmd)
    if result.returncode != 0:
        error(f"Failed to start server: {result.stderr.strip()}")
        return False
    return True


def stop_server() -> bool:
    """Stop and remove the game server container."""
    if not is_running():
        info("Server is not running.")
        return True
    step("Stopping game server...")
    result = _run(["docker", "stop", CONTAINER_NAME])
    if result.returncode != 0:
        error(f"Failed to stop server: {result.stderr.strip()}")
        return False
    success("Server stopped.")
    return True


def wait_for_health(port: int = 8000, timeout: int = 120) -> bool:
    """Poll the health endpoint until the server is ready."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    step(f"Waiting for server to be ready (timeout {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.urlopen(url, timeout=3)
            if req.status == 200:
                success("Server is ready!")
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    error(f"Server did not become healthy within {timeout}s.")
    return False


def get_logs(follow: bool = False) -> None:
    """Print container logs."""
    if not is_running():
        # Try to get logs from stopped container too
        pass
    cmd = ["docker", "logs"]
    if follow:
        cmd.append("-f")
    cmd.append(CONTAINER_NAME)
    subprocess.run(cmd)


def server_status() -> Optional[dict]:
    """Get server container status info."""
    if not is_running():
        return None
    result = _run([
        "docker", "ps", "--filter", f"name={CONTAINER_NAME}",
        "--format", "{{.Status}}\t{{.Ports}}"
    ])
    if result.stdout.strip():
        parts = result.stdout.strip().split("\t")
        return {
            "status": parts[0] if parts else "unknown",
            "ports": parts[1] if len(parts) > 1 else "",
        }
    return None
