"""Docker orchestration for the OpenRA-RL game server."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from openra_env.cli.console import error, info, step, success

IMAGE_REPO = "ghcr.io/yxc20089/openra-rl"
IMAGE = f"{IMAGE_REPO}:latest"
CONTAINER_NAME = "openra-rl-server"
REPLAY_CONTAINER = "openra-rl-replay"
REPLAY_DIR_IN_CONTAINER = "/root/.config/openra/Replays/ra"
LOCAL_REPLAY_DIR = Path.home() / ".openra-rl" / "replays"
MANIFEST_PATH = LOCAL_REPLAY_DIR / "manifest.json"


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


def _image_tag(version: Optional[str] = None) -> str:
    """Return the full image tag for a given version (default: latest)."""
    tag = version or "latest"
    return f"{IMAGE_REPO}:{tag}"


def pull_image(version: Optional[str] = None, quiet: bool = False) -> bool:
    """Pull the game server image from GHCR."""
    image = _image_tag(version)
    if not quiet:
        step(f"Pulling game server image ({image})...")
    result = subprocess.run(
        ["docker", "pull", image],
        stdout=sys.stdout if not quiet else subprocess.DEVNULL,
        stderr=sys.stderr if not quiet else subprocess.DEVNULL,
    )
    if result.returncode != 0:
        error(f"Failed to pull {image}")
        return False
    if not quiet:
        success("Image pulled successfully.")
    return True


def image_exists(version: Optional[str] = None) -> bool:
    """Check if the game server image is available locally."""
    image = _image_tag(version)
    result = _run(["docker", "images", "-q", image])
    return bool(result.stdout.strip())


def list_local_versions() -> list[str]:
    """List all locally available openra-rl image versions (tags), newest first."""
    result = _run([
        "docker", "images", IMAGE_REPO,
        "--format", "{{.Tag}}",
    ])
    if result.returncode != 0:
        return []
    tags = [t.strip() for t in result.stdout.splitlines() if t.strip()]
    # Put "latest" first, then sort the rest in reverse
    versions = sorted([t for t in tags if t != "latest"], reverse=True)
    if "latest" in tags:
        versions.insert(0, "latest")
    return versions


def get_running_image_tag() -> Optional[str]:
    """Get the image tag of the currently running game server container."""
    if not is_running():
        return None
    result = _run([
        "docker", "inspect", CONTAINER_NAME,
        "--format", "{{.Config.Image}}",
    ])
    if result.returncode != 0:
        return None
    image = result.stdout.strip()
    # Extract tag from "ghcr.io/yxc20089/openra-rl:0.2.1"
    if ":" in image:
        return image.split(":")[-1]
    return "latest"


# ── Replay manifest ──────────────────────────────────────────────────


def _load_manifest() -> dict:
    """Load the replay manifest (replay filename → image tag)."""
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_manifest(manifest: dict) -> None:
    """Save the replay manifest."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")


def get_replay_image_tag(replay_filename: str) -> Optional[str]:
    """Look up which image tag was used to record a replay."""
    manifest = _load_manifest()
    return manifest.get(replay_filename)


def _record_replays_in_manifest(filenames: list[str], image_tag: str) -> None:
    """Record which image tag was used for newly copied replays."""
    if not filenames:
        return
    manifest = _load_manifest()
    for f in filenames:
        manifest[f] = image_tag
    _save_manifest(manifest)


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
    version: Optional[str] = None,
) -> bool:
    """Start the game server container."""
    if is_running():
        info(f"Server already running on port {port}.")
        return True

    image = _image_tag(version)

    # Ensure image exists
    if not image_exists(version):
        if not pull_image(version):
            return False

    step(f"Starting game server on port {port} ({image})...")
    cmd = [
        "docker", "run", "--rm",
        "-d" if detach else "",
        "-p", f"{port}:8000",
        "--name", CONTAINER_NAME,
        "-e", f"BOT_TYPE={difficulty}",
        image,
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


# ── Replay viewer ────────────────────────────────────────────────────


def list_replays() -> list[str]:
    """List .orarep files inside the game server container."""
    if not is_running():
        return []
    result = _run([
        "docker", "exec", CONTAINER_NAME,
        "find", REPLAY_DIR_IN_CONTAINER, "-name", "*.orarep", "-type", "f",
    ])
    if result.returncode != 0:
        return []
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    files.sort()
    return files


def get_latest_replay() -> Optional[str]:
    """Return the path of the newest replay inside the game server container."""
    replays = list_replays()
    return replays[-1] if replays else None


def copy_replays() -> list[str]:
    """Copy all replays from the game server container to ~/.openra-rl/replays/.

    Returns list of newly copied filenames.
    Also records the image tag in the manifest so replay watch uses the right version.
    """
    if not is_running():
        error("Game server is not running — cannot copy replays.")
        return []

    LOCAL_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    # Get list of replays in container
    replays = list_replays()
    if not replays:
        return []

    # Get existing local files to detect new ones
    existing = {f.name for f in LOCAL_REPLAY_DIR.iterdir() if f.suffix == ".orarep"}

    # Copy each replay individually (docker cp doesn't glob well)
    for replay_path in replays:
        filename = os.path.basename(replay_path)
        result = _run([
            "docker", "cp",
            f"{CONTAINER_NAME}:{replay_path}",
            str(LOCAL_REPLAY_DIR / filename),
        ])
        if result.returncode != 0:
            error(f"Failed to copy {filename}: {result.stderr.strip()}")

    # Determine which files are new
    after = {f.name for f in LOCAL_REPLAY_DIR.iterdir() if f.suffix == ".orarep"}
    new_files = sorted(after - existing)

    # Record the image version that produced these replays
    if new_files:
        tag = get_running_image_tag() or "latest"
        _record_replays_in_manifest(new_files, tag)

    return new_files


def is_replay_viewer_running() -> bool:
    """Check if the replay viewer container is running."""
    result = _run([
        "docker", "ps", "--filter", f"name={REPLAY_CONTAINER}",
        "--format", "{{.Names}}"
    ])
    return REPLAY_CONTAINER in result.stdout


def start_replay_viewer(replay_path: str, port: int = 6080, version: Optional[str] = None) -> bool:
    """Start the replay viewer container.

    Args:
        replay_path: Path to .orarep file (container path or local path).
        port: noVNC port to expose (default 6080).
        version: Docker image version to use (default: auto-detect from manifest).
    """
    if is_replay_viewer_running():
        error("Replay viewer is already running. Stop it first with: openra-rl replay stop")
        return False

    # Auto-detect version from manifest if not specified
    if version is None:
        filename = os.path.basename(replay_path)
        version = get_replay_image_tag(filename)
        if version:
            info(f"Using image version '{version}' (from manifest)")

    image = _image_tag(version)

    if not image_exists(version):
        step(f"Image {image} not found locally, pulling...")
        if not pull_image(version):
            return False

    # Determine if this is a local file or a container path.
    # If the file exists on the host, mount it into the container.
    # Otherwise, assume it's a path inside the game server container.
    local_file = None
    container_replay_path = replay_path
    local_path = Path(replay_path).resolve()

    if local_path.exists():
        local_file = str(local_path)
        container_replay_path = f"/tmp/replay/{local_path.name}"
    elif not replay_path.startswith("/"):
        error(f"Replay file not found: {local_path}")
        return False

    step(f"Starting replay viewer on port {port} ({image})...")

    cmd = [
        "docker", "run", "--rm", "-d",
        "-p", f"{port}:6080",
        "--name", REPLAY_CONTAINER,
        "--entrypoint", "/replay-viewer.sh",
    ]

    if local_file:
        # Mount the local replay file
        cmd.extend(["-v", f"{local_file}:{container_replay_path}:ro"])
    elif is_running():
        # Share replay volume from game server container
        cmd.extend(["--volumes-from", CONTAINER_NAME])

    cmd.extend([image, container_replay_path])

    result = _run(cmd)
    if result.returncode != 0:
        error(f"Failed to start replay viewer: {result.stderr.strip()}")
        return False

    success("Replay viewer started.")
    return True


def stop_replay_viewer() -> bool:
    """Stop the replay viewer container."""
    if not is_replay_viewer_running():
        info("Replay viewer is not running.")
        return True
    step("Stopping replay viewer...")
    result = _run(["docker", "stop", REPLAY_CONTAINER])
    if result.returncode != 0:
        error(f"Failed to stop replay viewer: {result.stderr.strip()}")
        return False
    success("Replay viewer stopped.")
    return True
