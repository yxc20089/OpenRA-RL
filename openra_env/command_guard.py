"""Action-level command guard for dedupe and anti-loop checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from openra_env.normalization import get_production_item, normalize_name


@dataclass
class GuardDecision:
    """Guard decision for one command."""

    status: str  # allow | block | defer
    reason: str = ""
    message: str = ""
    next_action_hint: str = ""
    cooldown_until_tick: int = 0

    @property
    def allowed(self) -> bool:
        return self.status == "allow"

    def to_result(self, tick: int = 0) -> dict[str, Any]:
        result: dict[str, Any] = {
            "guard_status": self.status,
            "guard_reason": self.reason,
        }
        if self.next_action_hint:
            result["next_action_hint"] = self.next_action_hint
        if self.cooldown_until_tick > 0:
            result["cooldown_until_tick"] = self.cooldown_until_tick
        if tick > 0:
            result["tick"] = tick
        if self.status == "block":
            result["error"] = self.message or "Command blocked by guard."
        elif self.status == "defer":
            result["note"] = self.message or "Command deferred by guard."
        return result


class CommandGuard:
    """Centralized idempotency, anti-loop and lifecycle gate."""

    _CONTROL_TOOLS = {
        "move_units",
        "attack_move",
        "attack_target",
        "stop_units",
        "guard_target",
        "set_stance",
        "harvest",
    }
    _LOOP_WINDOW_TICKS = 120
    _BUILD_COOLDOWN_TICKS = 50
    _PLACE_COOLDOWN_TICKS = 25
    _CANCEL_COOLDOWN_TICKS = 40
    _MAX_UNIT_QUEUE_PER_TYPE = 8
    _MAX_UNIT_QUEUE_TOTAL = 24

    def __init__(self) -> None:
        self._cooldowns: dict[str, int] = {}
        self._last_control_tick: dict[str, int] = {}
        self._building_states: dict[str, str] = {}
        self._building_action_history: dict[str, list[tuple[int, str]]] = {}

    def reset(self) -> None:
        self._cooldowns.clear()
        self._last_control_tick.clear()
        self._building_states.clear()
        self._building_action_history.clear()

    def sync_building_states(
        self,
        obs: dict[str, Any],
        pending_placements: Optional[dict[str, Any]] = None,
        placeable_queue_types: Optional[set[str]] = None,
    ) -> None:
        """Sync lightweight building lifecycle from current observation."""
        pending_placements = pending_placements or {}
        placeable_queue_types = placeable_queue_types or {"Building", "Defense"}

        queue_state: dict[str, str] = {}
        for p in obs.get("production", []):
            queue_type = p.get("queue_type", "")
            if queue_type not in placeable_queue_types:
                continue
            item = get_production_item(p)
            progress = float(p.get("progress", 0.0))
            queue_state[item] = "ready_to_place" if progress >= 0.99 else "building"

        for btype, state in queue_state.items():
            self._building_states[btype] = state

        for btype in pending_placements:
            current = self._building_states.get(btype, "idle")
            if current in ("idle", "placed"):
                self._building_states[btype] = "queued"

        built_types = {normalize_name(b.get("type", "")) for b in obs.get("buildings", [])}
        for btype in list(self._building_states.keys()):
            if btype in built_types and btype not in queue_state:
                self._building_states[btype] = "placed"
            elif btype not in queue_state and btype not in pending_placements and self._building_states.get(btype) in (
                "queued",
                "building",
                "ready_to_place",
            ):
                self._building_states[btype] = "idle"

    def evaluate(
        self,
        tool_name: str,
        params: dict[str, Any],
        obs: dict[str, Any],
        pending_placements: Optional[dict[str, Any]] = None,
        placeable_queue_types: Optional[set[str]] = None,
    ) -> GuardDecision:
        """Evaluate one tool call and return allow/block/defer decision."""
        tick = int(obs.get("tick", 0))
        pending_placements = pending_placements or {}
        placeable_queue_types = placeable_queue_types or {"Building", "Defense"}
        self.sync_building_states(obs, pending_placements, placeable_queue_types)

        if tool_name == "build_unit":
            return self._evaluate_unit_build_tool(tool_name, params, obs, placeable_queue_types)

        if tool_name in {"build_structure", "build_and_place", "place_building", "cancel_production"}:
            return self._evaluate_building_tool(
                tool_name,
                params,
                obs,
                tick,
                pending_placements,
                placeable_queue_types,
            )

        if tool_name in self._CONTROL_TOOLS:
            return self._evaluate_control_tool(tool_name, params, tick)

        return GuardDecision(status="allow")

    def _evaluate_unit_build_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
        obs: dict[str, Any],
        placeable_queue_types: set[str],
    ) -> GuardDecision:
        unit_type = normalize_name(params.get("unit_type", ""))
        if not unit_type:
            return GuardDecision(status="allow")

        production = obs.get("production", [])
        unit_queue = [p for p in production if p.get("queue_type", "") not in placeable_queue_types]
        same_type_count = sum(1 for p in unit_queue if get_production_item(p) == unit_type)
        if same_type_count >= self._MAX_UNIT_QUEUE_PER_TYPE:
            return GuardDecision(
                status="defer",
                reason="unit_queue_type_limit",
                message=(
                    f"'{unit_type}' already has {same_type_count} queued items "
                    f"(limit {self._MAX_UNIT_QUEUE_PER_TYPE})."
                ),
                next_action_hint="advance_or_change_unit_mix",
            )
        if len(unit_queue) >= self._MAX_UNIT_QUEUE_TOTAL:
            return GuardDecision(
                status="defer",
                reason="unit_queue_total_limit",
                message=(
                    f"Total unit queue is {len(unit_queue)} "
                    f"(limit {self._MAX_UNIT_QUEUE_TOTAL})."
                ),
                next_action_hint="advance_or_trim_queue",
            )
        return GuardDecision(status="allow")

    def _evaluate_building_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
        obs: dict[str, Any],
        tick: int,
        pending_placements: dict[str, Any],
        placeable_queue_types: set[str],
    ) -> GuardDecision:
        item_param = "building_type" if "building_type" in params else "item_type"
        btype = normalize_name(params.get(item_param, ""))
        if not btype:
            return GuardDecision(status="allow")

        cooldown_key = f"{tool_name}:{btype}"
        until = self._cooldowns.get(cooldown_key, 0)
        if until > tick:
            return GuardDecision(
                status="defer",
                reason="cooldown_active",
                message=f"'{btype}' is in a short cooldown window.",
                next_action_hint="advance_or_refresh",
                cooldown_until_tick=until,
            )

        queue_entry = self._find_queue_entry(obs, btype, placeable_queue_types)
        pending = self._has_pending_type(pending_placements, btype)

        if tool_name in {"build_structure", "build_and_place"}:
            if pending:
                until_tick = tick + self._BUILD_COOLDOWN_TICKS
                self._cooldowns[cooldown_key] = until_tick
                return GuardDecision(
                    status="defer",
                    reason="already_pending",
                    message=f"'{btype}' is already queued and pending placement.",
                    next_action_hint="get_production_or_advance",
                    cooldown_until_tick=until_tick,
                )
            if queue_entry is not None:
                until_tick = tick + self._BUILD_COOLDOWN_TICKS
                self._cooldowns[cooldown_key] = until_tick
                return GuardDecision(
                    status="defer",
                    reason="already_in_queue",
                    message=f"'{btype}' is already being built.",
                    next_action_hint="get_production_or_advance",
                    cooldown_until_tick=until_tick,
                )
            if self._is_toggle_loop(btype, "build", tick):
                until_tick = tick + self._BUILD_COOLDOWN_TICKS
                self._cooldowns[cooldown_key] = until_tick
                return GuardDecision(
                    status="defer",
                    reason="cancel_build_loop",
                    message=f"Detected cancel/build loop for '{btype}'.",
                    next_action_hint="advance_and_reassess",
                    cooldown_until_tick=until_tick,
                )
            self._record_building_action(btype, "build", tick)
            return GuardDecision(status="allow")

        if tool_name == "place_building":
            if pending:
                return GuardDecision(
                    status="defer",
                    reason="auto_managed",
                    message=f"'{btype}' is queued via build_and_place — placement is automatic.",
                    next_action_hint="advance_or_wait",
                )
            if queue_entry is None:
                return GuardDecision(
                    status="block",
                    reason="not_in_queue",
                    message=f"'{btype}' is not in a placeable production queue.",
                    next_action_hint="build_or_wait",
                )
            progress = float(queue_entry.get("progress", 0.0))
            if progress < 0.99:
                until_tick = tick + self._PLACE_COOLDOWN_TICKS
                self._cooldowns[cooldown_key] = until_tick
                return GuardDecision(
                    status="defer",
                    reason="not_ready_to_place",
                    message=f"'{btype}' is not ready to place yet.",
                    next_action_hint="advance_until_ready",
                    cooldown_until_tick=until_tick,
                )
            # Even if currently ready, back off immediate repeated place calls.
            self._cooldowns[cooldown_key] = tick + self._PLACE_COOLDOWN_TICKS
            self._record_building_action(btype, "place", tick)
            return GuardDecision(status="allow")

        if tool_name == "cancel_production":
            queue_match = self._find_any_queue_entry(obs, btype)
            if queue_match is None:
                return GuardDecision(
                    status="block",
                    reason="not_in_queue",
                    message=f"'{btype}' is not in the production queue.",
                    next_action_hint="refresh_queue_first",
                )
            if self._is_repeated_action(btype, "cancel", tick, self._CANCEL_COOLDOWN_TICKS):
                until_tick = tick + self._CANCEL_COOLDOWN_TICKS
                self._cooldowns[cooldown_key] = until_tick
                return GuardDecision(
                    status="defer",
                    reason="cancel_repeat_backoff",
                    message=f"Repeated cancel on '{btype}' detected. Backing off briefly.",
                    next_action_hint="advance_or_refresh_queue",
                    cooldown_until_tick=until_tick,
                )
            if self._is_toggle_loop(btype, "cancel", tick):
                until_tick = tick + self._BUILD_COOLDOWN_TICKS
                self._cooldowns[cooldown_key] = until_tick
                return GuardDecision(
                    status="defer",
                    reason="cancel_build_loop",
                    message=f"Detected cancel/build loop for '{btype}'.",
                    next_action_hint="advance_and_reassess",
                    cooldown_until_tick=until_tick,
                )
            self._record_building_action(btype, "cancel", tick)
            return GuardDecision(status="allow")

        return GuardDecision(status="allow")

    def _evaluate_control_tool(self, tool_name: str, params: dict[str, Any], tick: int) -> GuardDecision:
        key = self._control_dedupe_key(tool_name, params)
        if self._last_control_tick.get(key) == tick:
            return GuardDecision(
                status="block",
                reason="duplicate_control_same_tick",
                message="Duplicate control command in same tick was blocked.",
                next_action_hint="refresh_units_before_reissue",
            )
        self._last_control_tick[key] = tick
        return GuardDecision(status="allow")

    def _control_dedupe_key(self, tool_name: str, params: dict[str, Any]) -> str:
        unit_ids = params.get("resolved_unit_ids")
        if isinstance(unit_ids, list):
            units_repr = ",".join(str(uid) for uid in sorted(unit_ids))
        else:
            units_repr = str(params.get("unit_ids", params.get("unit_id", "")))
        target_repr = ",".join(
            str(params.get(k, ""))
            for k in ("target_x", "target_y", "target_actor_id", "stance", "queued")
        )
        return f"{tool_name}|{units_repr}|{target_repr}"

    @staticmethod
    def _find_queue_entry(
        obs: dict[str, Any],
        item: str,
        placeable_queue_types: set[str],
    ) -> Optional[dict[str, Any]]:
        for p in obs.get("production", []):
            if p.get("queue_type", "") not in placeable_queue_types:
                continue
            if get_production_item(p) == item:
                return p
        return None

    @staticmethod
    def _find_any_queue_entry(obs: dict[str, Any], item: str) -> Optional[dict[str, Any]]:
        for p in obs.get("production", []):
            if get_production_item(p) == item:
                return p
        return None

    @staticmethod
    def _has_pending_type(pending_placements: dict[str, Any], btype: str) -> bool:
        """True when a building type has at least one pending placement entry."""
        for key, value in pending_placements.items():
            if normalize_name(key) != btype:
                continue
            if isinstance(value, list):
                return len(value) > 0
            if isinstance(value, dict):
                return True
            return bool(value)
        return False

    def _record_building_action(self, item: str, action: str, tick: int) -> None:
        hist = self._building_action_history.setdefault(item, [])
        hist.append((tick, action))
        if len(hist) > 5:
            hist.pop(0)

    def _is_toggle_loop(self, item: str, action: str, tick: int) -> bool:
        hist = self._building_action_history.get(item, [])
        if len(hist) < 2:
            return False
        (_, prev_action_2), (last_tick, prev_action_1) = hist[-2], hist[-1]
        if tick - last_tick > self._LOOP_WINDOW_TICKS:
            return False
        # Only treat explicit build<->cancel alternation as a toggle loop.
        # Legitimate sequences like build -> place -> build should remain allowed.
        if action == "build":
            return prev_action_2 == "build" and prev_action_1 == "cancel"
        if action == "cancel":
            return prev_action_2 == "cancel" and prev_action_1 == "build"
        return False

    def _is_repeated_action(self, item: str, action: str, tick: int, window: int) -> bool:
        hist = self._building_action_history.get(item, [])
        if not hist:
            return False
        last_tick, last_action = hist[-1]
        return last_action == action and tick - last_tick <= window
