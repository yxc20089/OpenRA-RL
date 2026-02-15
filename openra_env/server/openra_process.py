"""OpenRA subprocess manager.

Handles launching, monitoring, and terminating OpenRA game instances
for RL training episodes.
"""

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default path to the OpenRA installation
DEFAULT_OPENRA_PATH = os.environ.get("OPENRA_PATH", "/opt/openra")


@dataclass
class OpenRAConfig:
    """Configuration for launching an OpenRA game instance."""

    openra_path: str = DEFAULT_OPENRA_PATH
    mod: str = "ra"
    map_name: str = "singles.oramap"
    grpc_port: int = 9999
    bot_name: str = "Normal AI"
    bot_type: str = "normal"
    rl_slot: str = "Multi1"
    ai_slot: str = ""
    seed: Optional[int] = None
    headless: bool = True  # Use Null renderer (no GPU needed)
    record_replays: bool = False  # Enable .orarep replay recording
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
        """
        if self._process is not None and self._process.poll() is None:
            logger.warning("Killing existing OpenRA process before launching new one")
            self.kill()

        cmd = self._build_command()
        logger.info(f"Launching OpenRA: {' '.join(cmd)}")

        env = os.environ.copy()
        env.setdefault("DOTNET_ROLL_FORWARD", "LatestMajor")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.config.openra_path,
            env=env,
        )
        logger.info(f"OpenRA launched with PID {self._process.pid}")
        return self._process.pid

    def _build_command(self) -> list[str]:
        """Build the command line for launching OpenRA.

        Uses the game client (OpenRA.dll) with Launch.Map and Launch.Bots
        to auto-start a local game with the RL bot and optional AI opponent.
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

        # Build bots configuration: slot:bottype,slot:bottype
        bots = f"{self.config.rl_slot}:rl-agent"
        if self.config.ai_slot:
            bots += f",{self.config.ai_slot}:{self.config.bot_type}"

        args = [
            *exe,
            f"Engine.EngineDir={self.config.openra_path}",
            f"Game.Mod={self.config.mod}",
            f"Launch.Map={self.config.map_name}",
            f"Launch.Bots={bots}",
        ]

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
        if self._process is not None and self._process.poll() is None:
            try:
                self._process.kill()
            except Exception:
                pass
