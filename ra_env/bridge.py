"""Thin synchronous gRPC client for the OpenRA multi-session daemon.

One Bridge per daemon address. The underlying channel is shared and
thread-safe, so many GameSessions (one per thread) can use the same Bridge.
The daemon ticks all of its sessions on a single thread: extra sessions per
daemon hide round-trip latency, extra daemons add CPU.
"""

import os
import time

import grpc

from ra_env._proto import rl_bridge_pb2 as pb
from ra_env._proto import rl_bridge_pb2_grpc as pb_grpc

# Must not exceed the C# FastAdvance deadline (OPENRA_RL_FAST_ADVANCE_DEADLINE_S).
DEFAULT_DEADLINE_S = float(os.environ.get("OPENRA_RL_FAST_ADVANCE_DEADLINE_S", "90"))

READY_PHASES = ("playing", "paused", "ready")


class SessionPoisoned(RuntimeError):
    """The daemon aborted this session (deadline / hang). Never reuse its id."""


class Bridge:
    def __init__(self, address: str = "localhost:9999", deadline_s: float = DEFAULT_DEADLINE_S):
        self.address = address
        self.deadline_s = max(5.0, deadline_s)
        self._channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.max_send_message_length", 16 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 10000),
                ("grpc.keepalive_timeout_ms", 5000),
            ],
        )
        self._stub = pb_grpc.RLBridgeStub(self._channel)

    def create_session(self, map_name: str, bots: str, seed: int = 0,
                       player_faction: str = "", enemy_faction: str = "", agent_brain: str = "") -> str:
        req = pb.CreateSessionRequest(
            map_name=map_name, bots=bots, seed=seed,
            player_faction=player_faction, enemy_faction=enemy_faction, agent_brain=agent_brain,
        )
        return self._stub.CreateSession(req, timeout=60.0, wait_for_ready=True).session_id

    def destroy_session(self, session_id: str) -> None:
        try:
            self._stub.DestroySession(pb.DestroySessionRequest(session_id=session_id), timeout=30.0)
        except grpc.RpcError:
            pass  # already purged by the daemon

    def get_state(self, session_id: str, timeout_s: float = 5.0) -> pb.GameState:
        return self._stub.GetState(pb.StateRequest(session_id=session_id), timeout=timeout_s)

    def wait_ready(self, session_id: str, timeout_s: float = 45.0) -> bool:
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            try:
                phase = self.get_state(session_id, timeout_s=min(5.0, timeout_s)).phase
            except grpc.RpcError:
                phase = ""
            if phase in READY_PHASES:
                return True
            if phase == "error":
                return False
            time.sleep(0.25)
        return False

    def fast_advance(self, session_id: str, ticks: int, decisions: list[pb.Decision] | None = None,
                     commands: list[pb.Command] | None = None) -> pb.GameObservation:
        req = pb.FastAdvanceRequest(ticks=ticks, session_id=session_id)
        if decisions:
            req.decisions.extend(decisions)
        if commands:
            req.commands.extend(commands)
        try:
            return self._stub.FastAdvance(req, timeout=self.deadline_s)
        except grpc.RpcError as e:
            if e.code() in (grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.ABORTED):
                raise SessionPoisoned(f"{session_id}: {e.code().name} {e.details()}") from e
            raise

    def close(self) -> None:
        self._channel.close()
