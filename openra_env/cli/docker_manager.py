"""Docker orchestration for the OpenRA-RL game server."""

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
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
        encoding="utf-8",
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
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_manifest(manifest: dict) -> None:
    """Save the replay manifest."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


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


# ── Replay viewer settings ───────────────────────────────────────────


@dataclass(frozen=True)
class ReplayViewerSettings:
    """Tunable replay viewer settings for quality/performance tradeoffs."""

    width: int = 1280
    height: int = 960
    ui_scale: float = 1.0
    viewport_distance: str = "Medium"
    mute: bool = True
    render_mode: str = "auto"  # auto | gpu | cpu
    vnc_quality: int = 8
    vnc_compression: int = 4
    cpu_cores: int = 4  # Docker --cpus limit for software rendering (0 = all available)


def _parse_resolution(value: str) -> tuple[int, int]:
    """Parse a WxH resolution string."""
    raw = value.strip().lower().replace(" ", "")
    for sep in ("x", ","):
        if sep in raw:
            left, right = raw.split(sep, 1)
            try:
                w, h = int(left), int(right)
            except ValueError:
                break
            if w < 320 or h < 240 or w > 7680 or h > 4320:
                raise ValueError(f"resolution out of range (320x240..7680x4320): {value}")
            return w, h
    raise ValueError(f"resolution must be WxH (e.g. 960x540), got: {value!r}")


def _normalize_render_mode(value: str) -> str:
    """Validate and normalize render mode."""
    mode = value.strip().lower()
    if mode not in ("auto", "gpu", "cpu"):
        raise ValueError(f"render mode must be auto/gpu/cpu, got: {value!r}")
    return mode


def _normalize_viewport(value: str) -> str:
    """Validate and normalize viewport distance."""
    mapping = {"close": "Close", "medium": "Medium", "far": "Far"}
    key = value.strip().lower()
    if key not in mapping:
        raise ValueError(f"viewport must be close/medium/far, got: {value!r}")
    return mapping[key]


def load_replay_viewer_settings(
    resolution: Optional[str] = None,
    render_mode: Optional[str] = None,
    vnc_quality: Optional[int] = None,
    vnc_compression: Optional[int] = None,
    cpu_cores: Optional[int] = None,
) -> ReplayViewerSettings:
    """Load replay viewer settings from CLI overrides → env vars → defaults."""
    env = os.environ

    res = resolution or env.get("OPENRA_RL_REPLAY_RESOLUTION", "1280x960")
    w, h = _parse_resolution(res)

    mode = _normalize_render_mode(
        render_mode if render_mode is not None else env.get("OPENRA_RL_REPLAY_RENDER", "auto")
    )

    vq = vnc_quality if vnc_quality is not None else int(env.get("OPENRA_RL_REPLAY_VNC_QUALITY", "8"))
    vc = vnc_compression if vnc_compression is not None else int(env.get("OPENRA_RL_REPLAY_VNC_COMPRESSION", "4"))
    vq = max(0, min(9, vq))
    vc = max(0, min(9, vc))

    cores = cpu_cores if cpu_cores is not None else int(env.get("OPENRA_RL_REPLAY_CPU_CORES", "4"))
    if cores <= 0:
        cores = os.cpu_count() or 4
    cores = max(1, min(32, cores))

    ui_scale = float(env.get("OPENRA_RL_REPLAY_UI_SCALE", "1"))
    viewport = _normalize_viewport(env.get("OPENRA_RL_REPLAY_VIEWPORT_DISTANCE", "medium"))
    mute_raw = env.get("OPENRA_RL_REPLAY_MUTE", "true").strip().lower()
    mute = mute_raw not in ("0", "false", "no", "off")

    return ReplayViewerSettings(
        width=w, height=h, ui_scale=ui_scale, viewport_distance=viewport,
        mute=mute, render_mode=mode, vnc_quality=vq, vnc_compression=vc,
        cpu_cores=cores,
    )


def _settings_env_args(settings: ReplayViewerSettings) -> list[str]:
    """Convert settings to docker -e KEY=VAL args."""
    return [
        "-e", f"OPENRA_RL_REPLAY_RESOLUTION={settings.width}x{settings.height}",
        "-e", f"OPENRA_RL_REPLAY_UI_SCALE={settings.ui_scale}",
        "-e", f"OPENRA_RL_REPLAY_VIEWPORT_DISTANCE={settings.viewport_distance}",
        "-e", f"OPENRA_RL_REPLAY_MUTE={'True' if settings.mute else 'False'}",
        "-e", "SDL_AUDIODRIVER=dummy",
        "-e", "OPENRA_DISPLAY_SCALE=1",
    ]


def _gpu_docker_args(mode: str, cpu_cores: int = 4) -> list[list[str]]:
    """Return docker arg variants for GPU passthrough, in preference order.

    auto: try GPU variants first, fall back to CPU.
    gpu: only try GPU variants (fail if none work).
    cpu: only try CPU (software rendering).
    cpu_cores: number of llvmpipe threads for software rendering.
    """
    cpu = ["-e", "LIBGL_ALWAYS_SOFTWARE=1", "-e", f"LP_NUM_THREADS={cpu_cores}"]
    gpu_variants = [
        ["--gpus", "all"],                     # NVIDIA
        ["--device", "/dev/dxg:/dev/dxg"],     # WSL2
        ["--device", "/dev/dri:/dev/dri"],     # Linux DRI
    ]
    if mode == "cpu":
        return [cpu]
    if mode == "gpu":
        return gpu_variants
    # auto: try all GPU variants, then CPU fallback
    return gpu_variants + [cpu]


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


def replay_viewer_exists() -> bool:
    """Check if the replay viewer container exists (running or exited)."""
    result = _run([
        "docker", "ps", "-a", "--filter", f"name={REPLAY_CONTAINER}",
        "--format", "{{.Names}}"
    ])
    return REPLAY_CONTAINER in result.stdout


def get_replay_viewer_logs(tail: int = 200) -> str:
    """Return recent replay viewer logs, or empty string if unavailable."""
    if not replay_viewer_exists():
        return ""
    result = _run(["docker", "logs", "--tail", str(tail), REPLAY_CONTAINER])
    if result.returncode != 0:
        return result.stderr.strip() or result.stdout.strip()
    return result.stdout.strip()


def start_replay_viewer(
    replay_path: str,
    port: int = 6080,
    version: Optional[str] = None,
    settings: Optional[ReplayViewerSettings] = None,
) -> bool:
    """Start the replay viewer container.

    Args:
        replay_path: Path to .orarep file (container path or local path).
        port: noVNC port to expose (default 6080).
        version: Docker image version to use (default: auto-detect from manifest).
        settings: Replay viewer tuning (resolution, render mode, etc.).
    """
    if settings is None:
        settings = load_replay_viewer_settings()

    if is_replay_viewer_running():
        error("Replay viewer is already running. Stop it first with: openra-rl replay stop")
        return False

    # Clean up stale (exited) container if it exists
    if replay_viewer_exists():
        _run(["docker", "rm", "-f", REPLAY_CONTAINER])

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

    # Build base docker command
    base_cmd = [
        "docker", "run", "-d",
        "-p", f"{port}:6080",
        "--name", REPLAY_CONTAINER,
        "--entrypoint", "/replay-viewer.sh",
    ]
    base_cmd.extend(_settings_env_args(settings))

    if local_file:
        base_cmd.extend(["-v", f"{local_file}:{container_replay_path}:ro"])
    elif is_running():
        base_cmd.extend(["--volumes-from", CONTAINER_NAME])

    # Try GPU variants in order, fall back to CPU
    last_stderr = ""
    for gpu_args in _gpu_docker_args(settings.render_mode, cpu_cores=settings.cpu_cores):
        is_gpu = "--gpus" in gpu_args or "--device" in gpu_args
        # Limit CPU for software rendering to prevent runaway usage.
        # llvmpipe busy-loops without GPU; --cpus caps Docker scheduler.
        cpu_limit = [] if is_gpu else ["--cpus", str(settings.cpu_cores)]
        cmd = base_cmd + cpu_limit + gpu_args + [image, container_replay_path]
        result = _run(cmd)
        if result.returncode == 0:
            if is_gpu:
                info("Rendering mode: GPU (hardware acceleration)")
            else:
                info(f"Rendering mode: CPU (software, {settings.cpu_cores} cores)")
            success("Replay viewer started.")
            return True
        last_stderr = result.stderr.strip()
        # Clean up the failed container before trying next variant
        _run(["docker", "rm", "-f", REPLAY_CONTAINER])

    error(f"Failed to start replay viewer: {last_stderr}")
    return False


def stop_replay_viewer() -> bool:
    """Stop and remove the replay viewer container."""
    if not replay_viewer_exists():
        info("Replay viewer is not running.")
        return True
    step("Stopping replay viewer...")
    result = _run(["docker", "rm", "-f", REPLAY_CONTAINER])
    if result.returncode != 0:
        error(f"Failed to stop replay viewer: {result.stderr.strip()}")
        return False
    success("Replay viewer stopped.")
    return True
