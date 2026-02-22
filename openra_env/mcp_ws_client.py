"""WebSocket MCP client for OpenRA-RL.

Talks to the OpenEnv server's /ws endpoint using the correct message
protocol for MCP tool calls:
  - {"type": "reset"}                    → reset environment
  - {"type": "mcp", "data": {...}}       → JSON-RPC MCP call (tools/list, tools/call)
  - {"type": "step", "data": {...}}      → Gym-style step (OpenRAAction)

MCPToolClient from OpenEnv sends ListToolsAction via "step" which the
server tries to parse as OpenRAAction and fails. This client uses the
correct "mcp" message type instead.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

from websockets.asyncio.client import connect as ws_connect


@dataclass
class Tool:
    """MCP tool descriptor."""
    name: str
    description: str
    input_schema: dict


class OpenRAMCPClient:
    """Async WebSocket client for OpenRA-RL with MCP tool support.

    Usage:
        async with OpenRAMCPClient("http://localhost:8000") as client:
            await client.reset()
            tools = await client.list_tools()
            result = await client.call_tool("get_game_state")
            result = await client.call_tool("build_structure", building_type="powr")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        message_timeout_s: float = 300.0,
    ):
        # Convert HTTP URL to WebSocket URL
        ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = ws_url.rstrip("/")
        self._ws_url = f"{ws_url}/ws"
        self._timeout = message_timeout_s
        self._ws = None
        self._rpc_id = 0
        self._tools_cache: Optional[list[Tool]] = None

    async def connect(self) -> "OpenRAMCPClient":
        """Connect to the WebSocket endpoint."""
        if self._ws is not None:
            return self

        # Handle proxy bypass for localhost
        ws_lower = self._ws_url.lower()
        is_localhost = "localhost" in ws_lower or "127.0.0.1" in ws_lower
        old_no_proxy = os.environ.get("NO_PROXY")

        if is_localhost:
            current = old_no_proxy or ""
            if "localhost" not in current.lower():
                os.environ["NO_PROXY"] = (
                    f"{current},localhost,127.0.0.1" if current else "localhost,127.0.0.1"
                )

        try:
            self._ws = await ws_connect(
                self._ws_url,
                open_timeout=30.0,
                max_size=50 * 1024 * 1024,  # 50 MB
                ping_interval=None,
            )
        except (asyncio.TimeoutError, OSError, ConnectionRefusedError) as e:
            raise RuntimeError(
                f"Could not connect to game server at {self._ws_url}: {e}\n"
                f"  Is the server running? Try: openra-rl server start"
            ) from e
        finally:
            if is_localhost:
                if old_no_proxy is None:
                    os.environ.pop("NO_PROXY", None)
                else:
                    os.environ["NO_PROXY"] = old_no_proxy

        return self

    async def close(self):
        """Close the WebSocket connection."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def __aenter__(self) -> "OpenRAMCPClient":
        return await self.connect()

    async def __aexit__(self, *args):
        await self.close()

    async def _send_recv(self, message: dict) -> dict:
        """Send a message and wait for response."""
        if self._ws is None:
            raise RuntimeError("Not connected. Call connect() first.")

        await self._ws.send(json.dumps(message))
        raw = await asyncio.wait_for(self._ws.recv(), timeout=self._timeout)
        return json.loads(raw)

    # ── Environment Control ───────────────────────────────────────

    async def reset(self, **kwargs) -> dict:
        """Reset the environment and start a new game."""
        response = await self._send_recv({"type": "reset", "data": kwargs})
        if response.get("type") == "error":
            raise RuntimeError(f"Reset failed: {response.get('data', {}).get('message', '?')}")
        return response.get("data", {})

    # ── MCP Tool Operations ───────────────────────────────────────

    async def list_tools(self, use_cache: bool = True) -> list[Tool]:
        """List available MCP tools."""
        if use_cache and self._tools_cache is not None:
            return self._tools_cache

        self._rpc_id += 1
        rpc_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": self._rpc_id,
        }

        response = await self._send_recv({"type": "mcp", "data": rpc_request})
        rpc_response = response.get("data", {})

        if "error" in rpc_response:
            raise RuntimeError(f"tools/list failed: {rpc_response['error']}")

        tools_data = rpc_response.get("result", {}).get("tools", [])
        self._tools_cache = [
            Tool(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", t.get("input_schema", {})),
            )
            for t in tools_data
        ]
        return self._tools_cache

    async def call_tool(self, name: str, **kwargs) -> Any:
        """Call an MCP tool by name with keyword arguments."""
        self._rpc_id += 1
        rpc_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": name, "arguments": kwargs},
            "id": self._rpc_id,
        }

        response = await self._send_recv({"type": "mcp", "data": rpc_request})
        rpc_response = response.get("data", {})

        if "error" in rpc_response:
            error = rpc_response["error"]
            raise RuntimeError(f"Tool '{name}' failed: {error.get('message', error)}")

        result = rpc_response.get("result")
        return self._unwrap_mcp_result(result)

    @staticmethod
    def _unwrap_mcp_result(result: Any) -> Any:
        """Unwrap FastMCP tool result to plain Python data.

        FastMCP wraps results as:
          {
            "content": [{"type": "text", "text": "..."}],
            "structured_content": {"result": <actual_data>},
            "data": <actual_data>,
            "is_error": false
          }

        Priority: structured_content.result > data > content text > raw result
        """
        if not isinstance(result, dict):
            return result

        # data field is correct for dicts, buggy ([{}]) for lists.
        # structured_content.result is correct for lists, empty string for dicts.
        # Strategy: use data if it's a non-empty dict, else structured_content.result,
        # else fall back to content text parsing.
        data = result.get("data")
        if isinstance(data, dict) and data:
            return data

        sc = result.get("structured_content")
        if isinstance(sc, dict):
            sc_result = sc.get("result")
            if sc_result is not None and sc_result != "":
                return sc_result

        # data for empty lists (both data=[] and sc.result=[])
        if isinstance(data, list) and data != [{}]:
            return data

        # Fallback: parse content text items
        content = result.get("content")
        if isinstance(content, list) and content:
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    try:
                        texts.append(json.loads(text))
                    except (json.JSONDecodeError, TypeError):
                        texts.append(text)
                else:
                    texts.append(item)
            if len(texts) == 1:
                return texts[0]
            return texts

        return result
