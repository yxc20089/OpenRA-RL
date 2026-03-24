"""OpenRA subprocess manager.

Handles launching, monitoring, and terminating OpenRA game instances
for RL training episodes.

Supports two modes:
  - Single-session (legacy): One process per game session, killed on reset
  - Multi-session (daemon): One long-lived process hosts many game sessions via gRPC
"""

import atexit
import logging
import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Global registry for atexit/signal cleanup
_active_managers: list["OpenRAProcessManager"] = []

# Default path to the OpenRA installation
DEFAULT_OPENRA_PATH = os.environ.get("OPENRA_PATH", "/opt/openra")

# Map user-friendly difficulty names to actual OpenRA bot type strings.
# Users can set either the friendly name or the raw OpenRA name.
# Difficulty tiers: beginner < easy < medium < hard < brutal
# Play styles (raw pass-through): rush, normal, turtle, naval
BOT_TYPE_MAP: dict[str, str] = {
    "beginner": "beginner",
    "easy": "easy",
    "medium": "medium",
    "hard": "normal",
    "brutal": "rush",
    "dummy": "dummy",
    "": "dummy",
}



@dataclass
class OpenRAConfig:
    """Configuration for launching an OpenRA game instance."""

    openra_path: str = DEFAULT_OPENRA_PATH
    mod: str = "ra"
    map_name: str = "singles.oramap"
    grpc_port: int = 9999
    bot_name: str = "Beginner AI"
    bot_type: str = "beginner"
    rl_slot: str = "Multi1"
    ai_slot: str = "Multi0"
    seed: Optional[int] = None
    headless: bool = True  # Use Null renderer (no GPU needed)
    record_replays: bool = False  # Enable .orarep replay recording
    multi_session: bool = False  # Multi-session daemon mode
    extra_args: dict = field(default_factory=dict)


class OpenRAProcessManager:
    """Manages an OpenRA game subprocess for RL training.

    Each episode starts a new OpenRA process with the ExternalBotBridge
    trait enabled. The process communicates with the Python environment
    via gRPC on the configured port.
    """

    def __init__(self, config: Optional[OpenRAConfig] = None):
        self.config = config or OpenRAConfig()
        self._process: Optional[subprocess.Popen] = None
        self._stdout_log: list[str] = []
        self._stderr_log: list[str] = []

    def launch(self) -> int:
        """Launch a new OpenRA game instance.

        Returns the PID of the launched process.
        Registers atexit/signal handlers to ensure cleanup on Python exit.
        """
        if self._process is not None and self._process.poll() is None:
            logger.warning("Killing existing OpenRA process before launching new one")
            self.kill()

        cmd = self._build_command()
        logger.info(f"Launching OpenRA: {' '.join(cmd)}")

        env = os.environ.copy()
        env.setdefault("DOTNET_ROLL_FORWARD", "LatestMajor")
        # Pass gRPC port so each OpenRA process binds a unique port
        env["RL_GRPC_PORT"] = str(self.config.grpc_port)

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.config.openra_path,
            env=env,
        )
        logger.info(f"OpenRA launched with PID {self._process.pid}")

        # Register for global cleanup
        if self not in _active_managers:
            _active_managers.append(self)

        return self._process.pid

    def _build_command(self) -> list[str]:
        """Build the command line for launching OpenRA.

        Uses the game client (OpenRA.dll) with Launch.Map and Launch.Bots
        to auto-start a local game with the RL bot and optional AI opponent.
        In multi-session mode, uses Launch.MultiSession instead.
        """
        openra_path = Path(self.config.openra_path)

        # Find the game client executable (OpenRA.dll, not OpenRA.Server.dll)
        exe = None
        for search_dir in [openra_path, openra_path / "bin"]:
            game_dll = search_dir / "OpenRA.dll"
            if game_dll.exists():
                exe = ["dotnet", str(game_dll)]
                break

        if exe is None:
            # Fallback: look for the RL launch script
            launch_script = openra_path / "launch-rl.sh"
            if launch_script.exists():
                exe = ["bash", str(launch_script)]
            else:
                raise FileNotFoundError(
                    f"Could not find OpenRA game client in {openra_path}. "
                    "Expected OpenRA.dll in root or bin/, or launch-rl.sh"
                )

        args = [
            *exe,
            f"Engine.EngineDir={self.config.openra_path}",
            f"Game.Mod={self.config.mod}",
        ]

        if self.config.multi_session:
            # Multi-session daemon mode: no map/bots at launch time,
            # sessions are created via gRPC CreateSession RPC
            args.append(f"Launch.MultiSession={self.config.grpc_port}")
        else:
            # Single-session mode: start a specific map with bots
            # Build bots configuration: slot:bottype,slot:bottype
            bots = f"{self.config.rl_slot}:rl-agent"
            if self.config.ai_slot:
                # Map difficulty tiers to OpenRA bot types
                actual_type = BOT_TYPE_MAP.get(self.config.bot_type, self.config.bot_type)
                bots += f",{self.config.ai_slot}:{actual_type}"

            args.extend([
                f"Launch.Map={self.config.map_name}",
                f"Launch.Bots={bots}",
            ])

        # Use Null renderer for headless operation (no GPU/OpenGL needed)
        if self.config.headless:
            args.append("Game.Platform=Null")

        if self.config.record_replays:
            args.append("Server.RecordReplays=True")

        for key, value in self.config.extra_args.items():
            args.append(f"{key}={value}")

        return [a for a in args if a]

    def is_alive(self) -> bool:
        """Check if the OpenRA process is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def kill(self, timeout: float = 5.0) -> Optional[int]:
        """Terminate the OpenRA process.

        Returns the exit code, or None if the process had to be force-killed.
        """
        if self._process is None:
            return None

        pid = self._process.pid

        # Try graceful termination first
        try:
            self._process.terminate()
            try:
                exit_code = self._process.wait(timeout=timeout)
                logger.info(f"OpenRA process {pid} terminated gracefully (exit code {exit_code})")
                self._process = None
                return exit_code
            except subprocess.TimeoutExpired:
                pass
        except ProcessLookupError:
            self._process = None
            return None

        # Force kill
        try:
            self._process.kill()
            self._process.wait(timeout=2.0)
            logger.warning(f"OpenRA process {pid} force-killed")
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass

        self._process = None
        return None

    def reap(self) -> Optional[int]:
        """Reap a finished child process to prevent zombies.

        Returns exit code if process has exited, None if still running.
        """
        if self._process is None:
            return None
        rc = self._process.poll()  # calls waitpid(WNOHANG) internally
        if rc is not None:
            logger.info(f"OpenRA process {self._process.pid} exited with code {rc} (reaped)")
            self._process = None
        return rc

    def get_stdout(self) -> str:
        """Read available stdout from the process."""
        if self._process is None or self._process.stdout is None:
            return ""
        try:
            # Non-blocking read
            import select

            if select.select([self._process.stdout], [], [], 0.0)[0]:
                data = self._process.stdout.read(4096)
                if data:
                    text = data.decode("utf-8", errors="replace")
                    self._stdout_log.append(text)
                    return text
        except Exception:
            pass
        return ""

    def get_stderr(self) -> str:
        """Read available stderr from the process."""
        if self._process is None or self._process.stderr is None:
            return ""
        try:
            import select

            if select.select([self._process.stderr], [], [], 0.0)[0]:
                data = self._process.stderr.read(4096)
                if data:
                    text = data.decode("utf-8", errors="replace")
                    self._stderr_log.append(text)
                    return text
        except Exception:
            pass
        return ""

    @property
    def pid(self) -> Optional[int]:
        """Get the PID of the running process."""
        if self._process is None:
            return None
        return self._process.pid

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.kill()


def _cleanup_all_managers():
    """Kill all tracked OpenRA processes. Called on interpreter exit."""
    for mgr in _active_managers:
        try:
            if mgr.is_alive():
                logger.info(f"atexit: killing OpenRA process {mgr.pid}")
                mgr.kill(timeout=3.0)
            else:
                mgr.reap()  # reap zombie if already exited
        except Exception:
            pass
    _active_managers.clear()


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT: kill child processes then re-raise."""
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name}, shutting down OpenRA processes...")
    _cleanup_all_managers()
    # Re-raise with default handler so the process actually exits
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Register cleanup handlers
atexit.register(_cleanup_all_managers)
# Only install signal handlers in main thread
try:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
except ValueError:
    pass  # Not in main thread — atexit still works
