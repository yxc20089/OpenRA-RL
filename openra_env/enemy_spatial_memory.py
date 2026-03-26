"""Enemy spatial memory for fog-of-war tracking in OpenRA-RL.

Maintains a per-cell memory of last-known enemy positions that decays
over RL steps (not game ticks).  Designed to be independent of game time
so it behaves consistently regardless of how many ticks ``advance()``
advances per RL step.

Model
-----
Each cell maintains three independent scalar fields:

    memory_presence[y, x]  — 1.0 if there is active memory at this cell,
                              0.0 otherwise.
    memory_age[y, x]       — Integer step count since the cell was last
                              updated.  Starts at 0 when an enemy is first
                              seen; increments by 1 on every step.
    memory_decay[y, x]     — Confidence score, starts at 1.0 when first
                              recorded and multiplied by ``decay_factor``
                              on every step.

On each call to ``step()``:
    1. All active cells (presence == 1): age += 1, decay *= decay_factor.
    2. Cells where age >= max_age are cleared (presence=0, age=0, decay=0).
    3. Newly visible enemy cells are stamped: presence=1, age=0, decay=1.

Step 3 runs *after* the expiry check so a freshly-seen cell is never
cleared on the same step it is observed.

Exposed observation channels
-----------------------------
``get_channels()`` returns a (H, W, 3) float32 array:

    Channel 0  enemy_last_seen_presence  — binary: 0.0 or 1.0
    Channel 1  enemy_last_seen_age_norm  — age / max_age  ∈ [0, 1)
    Channel 2  enemy_last_seen_decay     — running decay  ∈ [0, 1]

For cells with presence == 0 all three channels are 0.0.

Usage::

    from openra_env.enemy_spatial_memory import EnemySpatialMemory, EnemySpatialMemoryConfig

    cfg = EnemySpatialMemoryConfig(decay_factor=0.97, max_age=150)
    mem = EnemySpatialMemory(map_h=64, map_w=64, config=cfg)

    # Episode loop
    mem.reset()
    for obs in episode:
        mem.step(obs["visible_enemies"], obs["visible_enemy_buildings"])
        channels = mem.get_channels()  # (64, 64, 3) float32
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Channel indices within get_channels() output ─────────────────────────────

CH_PRESENCE: int = 0   # enemy_last_seen_presence
CH_AGE_NORM: int = 1   # enemy_last_seen_age_norm
CH_DECAY: int = 2      # enemy_last_seen_decay

#: Names of the three output channels in order.
CHANNEL_NAMES: tuple[str, str, str] = (
    "enemy_last_seen_presence",
    "enemy_last_seen_age_norm",
    "enemy_last_seen_decay",
)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class EnemySpatialMemoryConfig:
    """Configuration for :class:`EnemySpatialMemory`.

    Attributes:
        decay_factor: Multiplicative decay applied to ``memory_decay`` on
            every step.  Range (0, 1); lower = faster forgetting.
            Default 0.98 → ~50 % confidence after ~35 steps.
        max_age: Maximum number of steps a cell may remain active before
            being unconditionally cleared.  Must be > 0.
            Default 200 steps.
    """

    decay_factor: float = 0.98
    max_age: int = 200


# ── Core class ────────────────────────────────────────────────────────────────

class EnemySpatialMemory:
    """Step-based spatial memory of last-known enemy positions.

    Tracks both enemy units and buildings in a single combined presence
    map.  All three internal arrays are allocated once on construction
    (or ``reset()``) and updated in-place for efficiency.

    Args:
        map_h: Map height in cells (must match the observation ``map_info``).
        map_w: Map width in cells.
        config: Decay and expiry settings.  If ``None``, defaults are used.
    """

    def __init__(
        self,
        map_h: int,
        map_w: int,
        config: EnemySpatialMemoryConfig | None = None,
    ) -> None:
        self.map_h = map_h
        self.map_w = map_w
        self.config = config or EnemySpatialMemoryConfig()
        self._presence: np.ndarray  # (H, W) float32
        self._age: np.ndarray       # (H, W) int32
        self._decay: np.ndarray     # (H, W) float32
        self.reset()

    # ── Episode lifecycle ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all memory.  Call at the start of every episode."""
        self._presence = np.zeros((self.map_h, self.map_w), dtype=np.float32)
        self._age      = np.zeros((self.map_h, self.map_w), dtype=np.int32)
        self._decay    = np.zeros((self.map_h, self.map_w), dtype=np.float32)

    # ── Per-step update ───────────────────────────────────────────────────────

    def step(
        self,
        visible_enemies: list[dict],
        visible_enemy_buildings: list[dict],
    ) -> None:
        """Advance memory by one RL step and record new sightings.

        Should be called **once per ``build()`` call** in
        :class:`~openra_env.obs_tensor.ObsTensorBuilder`, not once per
        game tick.

        Args:
            visible_enemies: List of enemy unit dicts (must have ``cell_x``,
                ``cell_y`` fields).
            visible_enemy_buildings: List of enemy building dicts (same
                schema).
        """
        active = self._presence > 0  # boolean mask (H, W)

        # 1. Advance age and apply decay to all active cells
        self._age[active]   += 1
        self._decay[active] *= self.config.decay_factor

        # 2. Expire cells that have exceeded max_age
        expired = active & (self._age >= self.config.max_age)
        self._presence[expired] = 0.0
        self._age[expired]      = 0
        self._decay[expired]    = 0.0

        # 3. Stamp newly visible cells (runs after expiry so fresh sightings
        #    are never immediately cleared)
        self._stamp(visible_enemies)
        self._stamp(visible_enemy_buildings)

    def _stamp(self, actors: list[dict]) -> None:
        """Mark each actor's cell as freshly observed."""
        for actor in actors:
            cy = actor.get("cell_y", -1)
            cx = actor.get("cell_x", -1)
            if 0 <= cy < self.map_h and 0 <= cx < self.map_w:
                self._presence[cy, cx] = 1.0
                self._age[cy, cx]      = 0
                self._decay[cy, cx]    = 1.0

    # ── Observation output ────────────────────────────────────────────────────

    def get_channels(self) -> np.ndarray:
        """Return the (H, W, 3) float32 observation tensor.

        Channel layout (see module constants ``CH_PRESENCE``, ``CH_AGE_NORM``,
        ``CH_DECAY``)::

            0  enemy_last_seen_presence — 0.0 or 1.0
            1  enemy_last_seen_age_norm — age / max_age  in [0, 1)
            2  enemy_last_seen_decay    — running decay  in [0, 1]

        All three channels are 0.0 for cells with no active memory.
        """
        active = self._presence > 0
        age_norm = np.where(
            active,
            np.clip(self._age / self.config.max_age, 0.0, 1.0),
            0.0,
        ).astype(np.float32)
        decay_out = np.where(active, self._decay, 0.0).astype(np.float32)
        return np.stack([self._presence, age_norm, decay_out], axis=-1)

    # ── Summary statistics ────────────────────────────────────────────────────

    def coverage(self) -> float:
        """Fraction of map cells with active memory (presence == 1)."""
        return float(self._presence.mean())

    def decay_mean(self) -> float:
        """Mean decay value across cells with active memory.

        Returns 0.0 when no cells have active memory.
        """
        active = self._presence > 0
        if not active.any():
            return 0.0
        return float(self._decay[active].mean())

    def active_cell_count(self) -> int:
        """Number of cells currently holding active memory."""
        return int((self._presence > 0).sum())
