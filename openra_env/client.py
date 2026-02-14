"""OpenRA-RL environment client.

Provides the EnvClient subclass for connecting to the OpenRA-RL
environment server over WebSocket.
"""

import os
from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
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


class OpenRAEnv(EnvClient[OpenRAAction, OpenRAObservation, OpenRAState]):
    """WebSocket client for the OpenRA-RL environment.

    Usage:
        async with OpenRAEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            while not result.done:
                action = OpenRAAction(commands=[...])
                result = await env.step(action)
    """

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
                max_size=self._max_message_size,
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

    def _step_payload(self, action: OpenRAAction) -> Dict[str, Any]:
        """Convert action to JSON for WebSocket transport."""
        return action.model_dump()

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
            map_info=MapInfoModel(**obs_data.get("map_info", {})),
            available_production=obs_data.get("available_production", []),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
            result=obs_data.get("result", ""),
        )

        return StepResult(
            observation=observation,
            reward=data.get("reward", obs_data.get("reward")),
            done=data.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, data: Dict[str, Any]) -> OpenRAState:
        """Parse state response into OpenRAState."""
        return OpenRAState(**data)
