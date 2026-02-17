"""OpenRA-RL environment client.

Provides an async WebSocket client for connecting to the OpenRA-RL
environment server. This is a standalone async implementation that
does not rely on the sync EnvClient base class.
"""

import asyncio
import json
import os
from typing import Any, Dict, Optional

from openenv.core.client_types import StepResult
from websockets.asyncio.client import connect as ws_connect

from openra_env.models import (
    BuildingInfoModel,
    EconomyInfo,
    MapInfoModel,
    MilitaryInfo,
    OpenRAAction,
    OpenRAObservation,
    OpenRAState,
    ProductionInfoModel,
    UnitInfoModel,
)


class OpenRAEnv:
    """Async WebSocket client for the OpenRA-RL environment.

    Usage:
        async with OpenRAEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            while not result.done:
                action = OpenRAAction(commands=[...])
                result = await env.step(action)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
    ):
        # Convert HTTP URL to WebSocket URL
        ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = ws_url.rstrip("/")
        self._ws_url = f"{ws_url}/ws"
        self._connect_timeout = connect_timeout_s
        self._message_timeout = message_timeout_s
        self._ws = None

    async def connect(self) -> "OpenRAEnv":
        """Connect with ping keepalive disabled.

        OpenRA operations (especially reset) can take 60-120+ seconds
        with software rendering. The default websockets ping_interval=20s
        would kill the connection before the server responds.
        """
        if self._ws is not None:
            return self

        ws_url_lower = self._ws_url.lower()
        is_localhost = "localhost" in ws_url_lower or "127.0.0.1" in ws_url_lower

        old_no_proxy = os.environ.get("NO_PROXY")
        if is_localhost:
            current_no_proxy = old_no_proxy or ""
            if "localhost" not in current_no_proxy.lower():
                os.environ["NO_PROXY"] = (
                    f"{current_no_proxy},localhost,127.0.0.1"
                    if current_no_proxy
                    else "localhost,127.0.0.1"
                )

        try:
            self._ws = await ws_connect(
                self._ws_url,
                open_timeout=self._connect_timeout,
                max_size=50 * 1024 * 1024,  # 50 MB for large spatial observations
                ping_interval=None,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self._ws_url}: {e}") from e
        finally:
            if is_localhost:
                if old_no_proxy is None:
                    os.environ.pop("NO_PROXY", None)
                else:
                    os.environ["NO_PROXY"] = old_no_proxy

        return self

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def __aenter__(self) -> "OpenRAEnv":
        """Support `async with OpenRAEnv(...) as env:` syntax."""
        return await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close connection on context exit."""
        await self.close()

    # ── Core protocol ──────────────────────────────────────────────

    async def _send_and_receive(self, message: dict) -> dict:
        """Send a JSON message and wait for a JSON response."""
        if self._ws is None:
            raise RuntimeError("Not connected. Call connect() first.")

        await self._ws.send(json.dumps(message))
        raw = await asyncio.wait_for(
            self._ws.recv(), timeout=self._message_timeout
        )
        return json.loads(raw)

    # ── Environment API ────────────────────────────────────────────

    async def reset(self, **kwargs: Any) -> StepResult[OpenRAObservation]:
        """Reset the environment and start a new game."""
        message = {
            "type": "reset",
            "data": kwargs,
        }
        response = await self._send_and_receive(message)
        return self._parse_result(response.get("data", {}))

    async def step(self, action: OpenRAAction, **kwargs: Any) -> StepResult[OpenRAObservation]:
        """Execute an action in the environment."""
        message = {
            "type": "step",
            "data": action.model_dump(),
        }
        response = await self._send_and_receive(message)
        return self._parse_result(response.get("data", {}))

    # ── Parsing ────────────────────────────────────────────────────

    def _parse_result(self, data: Dict[str, Any]) -> StepResult[OpenRAObservation]:
        """Parse server response into StepResult."""
        obs_data = data.get("observation", data)

        observation = OpenRAObservation(
            tick=obs_data.get("tick", 0),
            economy=EconomyInfo(**obs_data.get("economy", {})),
            military=MilitaryInfo(**obs_data.get("military", {})),
            units=[UnitInfoModel(**u) for u in obs_data.get("units", [])],
            buildings=[BuildingInfoModel(**b) for b in obs_data.get("buildings", [])],
            production=[ProductionInfoModel(**p) for p in obs_data.get("production", [])],
            visible_enemies=[UnitInfoModel(**u) for u in obs_data.get("visible_enemies", [])],
            visible_enemy_buildings=[BuildingInfoModel(**b) for b in obs_data.get("visible_enemy_buildings", [])],
            map_info=MapInfoModel(**obs_data.get("map_info", {})),
            available_production=obs_data.get("available_production", []),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
            result=obs_data.get("result", ""),
            spatial_map=obs_data.get("spatial_map", ""),
            spatial_channels=obs_data.get("spatial_channels", 0),
        )

        return StepResult(
            observation=observation,
            reward=data.get("reward", obs_data.get("reward")),
            done=data.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, data: Dict[str, Any]) -> OpenRAState:
        """Parse state response into OpenRAState."""
        return OpenRAState(**data)
