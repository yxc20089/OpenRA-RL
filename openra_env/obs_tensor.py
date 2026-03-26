"""RL observation tensor builder with RTS fog-of-war memory.

Converts raw OpenRA game observations into structured numeric tensors
suitable for neural network training (CNN + MLP heads).  Maintains
per-episode step-based memory of last-seen enemy positions.

Output format (from ``ObsTensorBuilder.build()``)::

    {
        "spatial":     np.ndarray  # (H, W, 13)  float32 — map features + memory
        "global_vec":  np.ndarray  # (28,)        float32 — normalised scalar features
    }

Spatial channels (13 total)
----------------------------
Channels 0-8 come directly from the OpenRA bridge spatial tensor:

    0   terrain_index      — integer terrain type (0-8), stored as float
    1   height             — elevation / height map
    2   resources          — ore / gem density
    3   passability        — walkability (0 = blocked, 1 = passable)
    4   fog                — fog-of-war coverage (0 = unexplored, 1 = visible)
    5   own_buildings      — friendly building density
    6   own_units          — friendly unit density
    7   enemy_buildings    — visible enemy building density (fog-limited)
    8   enemy_units        — visible enemy unit density (fog-limited)

Channels 9-12 are memory channels from :mod:`openra_env.enemy_memory`:

    9   presence   — binary: 1.0 if cell has active memory, else 0.0
    10  age_norm   — age / max_age in [0, 1)
    11  decay      — running confidence e^(−λ·age) in (0, 1]; 0.0 when inactive
    12  ever_seen  — cumulative 1.0 once a cell has been observed; never resets

Memory lifecycle per RL step (inside ``build()``)
---------------------------------------------------
1. ``memory.step_decay()``                   — age + decay existing cells
2. Collect (row, col) cells from             — extract cell_y, cell_x from
   visible_enemies + visible_enemy_buildings   the observation dict
3. ``memory.observe_enemy_positions(cells)`` — stamp fresh sightings
4. ``memory.export_observation_planes(stack=True)``  → (H, W, 4) appended
   to the 9 game channels to form the (H, W, 13) spatial tensor

Step_decay runs *before* observe so that a freshly-seen cell is never
expired on the same step it is stamped.

Global vector features (33 total)
----------------------------------
See ``GLOBAL_VEC_FEATURES`` for the ordered list of feature names.
All values are normalised so that typical game values fall in [0, 1].
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from openra_env.enemy_memory import EnemySpatialMemory

# ── Channel constants ─────────────────────────────────────────────────────────

GAME_SPATIAL_CHANNELS: int = 9    # Channels provided by the OpenRA bridge
MEMORY_CHANNELS: int = 4          # presence, age_norm, decay, ever_seen
OBS_SPATIAL_CHANNELS: int = GAME_SPATIAL_CHANNELS + MEMORY_CHANNELS  # 13

# Indices within the game-provided spatial tensor
CH_TERRAIN: int = 0
CH_HEIGHT: int = 1
CH_RESOURCES: int = 2
CH_PASSABILITY: int = 3
CH_FOG: int = 4
CH_OWN_BUILDINGS: int = 5
CH_OWN_UNITS: int = 6
CH_ENEMY_BUILDINGS: int = 7
CH_ENEMY_UNITS: int = 8

# Indices of memory channels within the full spatial tensor
CH_MEM_PRESENCE:  int = GAME_SPATIAL_CHANNELS + 0   # 9
CH_MEM_AGE_NORM:  int = GAME_SPATIAL_CHANNELS + 1   # 10
CH_MEM_DECAY:     int = GAME_SPATIAL_CHANNELS + 2   # 11
CH_MEM_EVER_SEEN: int = GAME_SPATIAL_CHANNELS + 3   # 12

# ── Global vector ─────────────────────────────────────────────────────────────

GLOBAL_VEC_SIZE: int = 28

#: Feature names in global-vector order — useful for debugging / logging.
GLOBAL_VEC_FEATURES: tuple[str, ...] = (
    # Economy (4)
    "cash_norm",
    "power_balance_norm",
    "harvester_count_norm",
    "ore_fill_ratio",
    # Military (6)
    "army_value_norm",
    "active_unit_count_norm",
    "kills_cost_norm",
    "deaths_cost_norm",
    "experience_norm",
    "kd_value_ratio",
    # Own assets (4)
    "own_building_count_norm",
    "own_unit_count_norm",
    "has_construction_yard",
    "has_war_factory",
    # Production (3)
    "production_queue_count_norm",
    "idle_production",
    "production_progress_mean",
    # Enemies (2)
    "visible_enemy_count_norm",
    "visible_enemy_building_count_norm",
    # Game time & status (4)
    "tick_norm",
    "is_low_power",
    "has_harvester",
    "has_mcv",
    # Memory (2)
    "enemy_memory_coverage",
    "enemy_memory_decay_mean",
    # Result one-hot (3)
    "result_win",
    "result_lose",
    "result_draw",
)

assert len(GLOBAL_VEC_FEATURES) == GLOBAL_VEC_SIZE, (
    f"GLOBAL_VEC_FEATURES length mismatch: {len(GLOBAL_VEC_FEATURES)} != {GLOBAL_VEC_SIZE}"
)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ObsTensorConfig:
    """Hyperparameters controlling normalisation and enemy memory.

    Attributes:
        memory_enabled: When ``False`` (default), all memory operations are
            skipped and channels 9-12 in the spatial tensor are 0.0.  The
            tensor shape is always ``(H, W, 13)`` regardless of this flag so
            the observation space remains fixed.  Set to ``True`` to activate
            step-based fog-of-war tracking.
        memory_max_age: Hard expiry threshold in RL steps.  A cell without a
            new sighting for this many steps is cleared.  Only used when
            ``memory_enabled=True``.  Default 200.
        memory_decay_lambda: Exponential decay rate λ per step.  Confidence
            multiplied by e^(−λ) each step.  λ≈0.02 gives ~50 % confidence
            after 35 steps.  Only used when ``memory_enabled=True``.
            Default 0.02.
        memory_store_threat: When ``True``, allocate the *threat* plane inside
            ``EnemySpatialMemory``.  Available to post-step hooks and
            subclasses; not exported to the spatial tensor by default.
            Only used when ``memory_enabled=True``.  Default ``False``.
        memory_diffusion_enabled: When ``True``, register a Gaussian diffusion
            hook on the memory object.  After each ``step_decay()`` the
            ``_presence`` and ``_decay`` planes are blurred, spreading
            uncertainty about enemy locations to neighbouring cells.
            Only used when ``memory_enabled=True``.  Default ``False``.
        memory_diffusion_sigma: Standard deviation (in cells) of the Gaussian
            kernel used for diffusion.  Larger values spread uncertainty
            further.  Only used when ``memory_diffusion_enabled=True``.
            Default 0.8.
        cash_scale: Cash value mapped to 1.0 (typical income ~$500/tick burst).
        army_scale: Army value mapped to 1.0.
        unit_scale: Unit count mapped to 1.0.
        building_scale: Building count mapped to 1.0.
        tick_scale: Tick count mapped to 1.0 (approx full-length game).
        cost_scale: Kill/death cost mapped to 1.0.
        experience_scale: Experience points mapped to 1.0.
    """

    memory_enabled: bool = False
    memory_max_age: int = 200
    memory_decay_lambda: float = 0.02
    memory_store_threat: bool = False
    memory_diffusion_enabled: bool = False
    memory_diffusion_sigma: float = 0.8
    cash_scale: float = 5000.0
    army_scale: float = 10000.0
    unit_scale: float = 50.0
    building_scale: float = 20.0
    tick_scale: float = 50000.0
    cost_scale: float = 50000.0
    experience_scale: float = 5000.0

    @classmethod
    def from_observation_config(cls, obs_cfg: "ObservationConfig", **overrides) -> "ObsTensorConfig":
        """Construct from an :class:`~openra_env.config.ObservationConfig`.

        All ``enemy_memory_*`` fields are mapped to their counterparts; any
        additional keyword arguments override the resulting instance.

        Example::

            from openra_env.config import load_config
            from openra_env.obs_tensor import ObsTensorConfig

            cfg = load_config()
            obs_cfg = ObsTensorConfig.from_observation_config(cfg.observation)
        """
        return cls(
            memory_enabled=obs_cfg.enemy_memory_enabled,
            memory_max_age=obs_cfg.enemy_memory_max_age,
            memory_decay_lambda=obs_cfg.enemy_memory_decay_lambda,
            memory_store_threat=obs_cfg.enemy_memory_store_threat,
            memory_diffusion_enabled=obs_cfg.enemy_memory_diffusion_enabled,
            memory_diffusion_sigma=obs_cfg.enemy_memory_diffusion_sigma,
            **overrides,
        )


# Forward-declare for the classmethod above; resolved at import time.
try:
    from openra_env.config import ObservationConfig  # noqa: E402
except ImportError:
    ObservationConfig = None  # type: ignore[assignment,misc]


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _actors_to_cells(actors: List[dict]) -> List[Tuple[int, int]]:
    """Convert a list of actor dicts to ``(row, col)`` cell tuples.

    OpenRA actors carry both world coordinates (``pos_x``, ``pos_y``) and
    pre-computed cell coordinates (``cell_x``, ``cell_y``).  We use the cell
    coordinates directly — they are already in grid-index space and match
    the spatial tensor's (row=cell_y, col=cell_x) layout.

    Out-of-bounds or missing values are left in the list; ``EnemySpatialMemory``
    silently ignores them.
    """
    cells = []
    for actor in actors:
        cy = actor.get("cell_y", -1)
        cx = actor.get("cell_x", -1)
        cells.append((int(cy), int(cx)))
    return cells


# ── Observation tensor builder ────────────────────────────────────────────────

class ObsTensorBuilder:
    """Converts raw OpenRA observations to structured RL tensors.

    Maintains per-episode fog-of-war memory via :class:`EnemySpatialMemory`.
    Call ``reset()`` at episode start; the builder also auto-initialises
    lazily on the first ``build()`` call if map dimensions were not provided.

    Memory lifecycle inside ``build()``
    ------------------------------------
    The three memory operations are always executed in this fixed order:

    1. ``memory.step_decay()`` — advance age, apply exponential decay, expire
       old cells.  Runs *before* observations so fresh stamps are never
       immediately cleared.
    2. Collect ``(row, col)`` cells from ``visible_enemies`` and
       ``visible_enemy_buildings`` using ``cell_y``/``cell_x`` from the obs dict.
    3. ``memory.observe_enemy_positions(cells)`` — stamp sightings.

    The resulting memory is then written into spatial channels 9-12.

    Example::

        builder = ObsTensorBuilder()

        # env.reset()
        builder.reset()
        tensors = builder.build(obs_dict, faction="russia")
        space = builder.observation_space(map_h=64, map_w=64)

        # env.step()
        tensors = builder.build(obs_dict, faction="russia")
        spatial    = tensors["spatial"]      # (H, W, 13) float32
        global_vec = tensors["global_vec"]   # (33,) float32
        act_mask   = tensors["action_mask"]  # (21,) bool
    """

    def __init__(self, config: Optional[ObsTensorConfig] = None) -> None:
        self.config = config or ObsTensorConfig()
        self._memory: Optional[EnemySpatialMemory] = None
        self._map_h: int = 0
        self._map_w: int = 0

    # ── Episode management ────────────────────────────────────────────────────

    def reset(self, map_h: int = 0, map_w: int = 0) -> None:
        """Reset episode state.

        Args:
            map_h: Map height in cells.  When 0, the builder lazily
                initialises from the first ``build()`` call's map_info.
            map_w: Map width in cells.
        """
        if map_h > 0 and map_w > 0:
            self._map_h = map_h
            self._map_w = map_w
            self._memory = self._make_memory(map_h, map_w)
        else:
            self._memory = None
            self._map_h = 0
            self._map_w = 0

    def _make_memory(self, map_h: int, map_w: int) -> EnemySpatialMemory:
        """Construct a fresh memory for the given map dimensions."""
        from openra_env.enemy_memory import make_diffusion_hook

        cfg = self.config
        mem = EnemySpatialMemory(
            height=map_h,
            width=map_w,
            max_age=cfg.memory_max_age,
            decay_lambda=cfg.memory_decay_lambda,
            track_threat=cfg.memory_store_threat,
        )
        if cfg.memory_diffusion_enabled:
            mem.add_post_step_hook(make_diffusion_hook(sigma=cfg.memory_diffusion_sigma))
        return mem

    def _ensure_memory(self, map_h: int, map_w: int) -> Optional[EnemySpatialMemory]:
        """Lazily create or re-create memory when map dimensions are known.

        Returns ``None`` when ``config.memory_enabled`` is ``False`` so that
        all callers can gate memory operations with a single ``if memory``
        check.
        """
        if not self.config.memory_enabled:
            return None
        if (
            self._memory is None
            or self._map_h != map_h
            or self._map_w != map_w
        ):
            self._map_h = map_h
            self._map_w = map_w
            self._memory = self._make_memory(map_h, map_w)
        return self._memory

    # ── Main build ────────────────────────────────────────────────────────────

    def build(self, obs_dict: dict, faction: str = "") -> dict[str, np.ndarray]:
        """Build RL tensors from a raw game observation.

        Memory update order (see module docstring):
            1. ``memory.step_decay()``
            2. Convert visible enemy cell coords → ``(row, col)`` list
            3. ``memory.observe_enemy_positions(cells)``

        Args:
            obs_dict: Raw observation dict (same schema as ``OpenRAObservation``
                serialised to dict).
            faction: Player faction, e.g. ``"russia"`` or ``"england"``.
                Used for the action-mask static-fallback side filter.

        Returns:
            Dict with keys:
                ``"spatial"``    — ``np.ndarray`` shape ``(H, W, 13)`` float32
                ``"global_vec"`` — ``np.ndarray`` shape ``(28,)`` float32
        """
        map_info = obs_dict.get("map_info", {})
        map_h = max(map_info.get("height", 0), 0)
        map_w = max(map_info.get("width", 0), 0)
        tick: int = obs_dict.get("tick", 0)

        memory = self._ensure_memory(map_h, map_w)  # None when disabled

        if memory is not None:
            # ── 1. Advance memory: age + decay + expiry ───────────────────────
            memory.step_decay()

            # ── 2. Convert visible enemy positions to (row, col) cell tuples ─
            #     cell_y maps to the row axis; cell_x to the column axis.
            #     Both enemy units and buildings are combined into one stamp so
            #     a single cell may be re-confirmed by either actor type.
            enemy_cells = _actors_to_cells(obs_dict.get("visible_enemies", []))
            enemy_cells += _actors_to_cells(obs_dict.get("visible_enemy_buildings", []))

            # ── 3. Stamp fresh sightings ─────────────────────────────────────
            if enemy_cells:
                memory.observe_enemy_positions(enemy_cells)

        return {
            "spatial":     self._build_spatial(obs_dict, map_h, map_w, memory),
            "global_vec":  self._build_global_vec(obs_dict, memory, tick),
        }

    # ── Spatial tensor ────────────────────────────────────────────────────────

    @staticmethod
    def _decode_spatial(
        spatial_b64: str,
        map_h: int,
        map_w: int,
        n_channels: int,
    ) -> Optional[np.ndarray]:
        """Decode a base64 spatial blob to (H, W, C) float32, or None on error."""
        if not spatial_b64 or map_h == 0 or map_w == 0:
            return None
        try:
            raw = base64.b64decode(spatial_b64)
            expected_bytes = map_h * map_w * n_channels * 4  # float32 = 4 bytes
            if len(raw) < expected_bytes:
                return None
            return (
                np.frombuffer(raw[:expected_bytes], dtype=np.float32)
                .reshape(map_h, map_w, n_channels)
                .copy()  # make writable
            )
        except Exception:
            return None

    def _build_spatial(
        self,
        obs_dict: dict,
        map_h: int,
        map_w: int,
        memory: Optional[EnemySpatialMemory],
    ) -> np.ndarray:
        """Assemble the (H, W, 13) spatial tensor.

        Layout:
            channels  0-8  — game-provided channels (decoded from base64 blob)
            channels 9-12  — memory planes (presence, age_norm, decay, ever_seen)
                             All zero when ``memory_enabled=False``.
        """
        fallback_h = max(map_h, 1)
        fallback_w = max(map_w, 1)

        n_game_ch = obs_dict.get("spatial_channels", GAME_SPATIAL_CHANNELS)
        game_spatial = self._decode_spatial(
            obs_dict.get("spatial_map", ""), map_h, map_w, n_game_ch
        )

        if game_spatial is None:
            return np.zeros(
                (fallback_h, fallback_w, OBS_SPATIAL_CHANNELS), dtype=np.float32
            )

        # Normalise to exactly GAME_SPATIAL_CHANNELS channels
        if n_game_ch < GAME_SPATIAL_CHANNELS:
            pad = np.zeros(
                (map_h, map_w, GAME_SPATIAL_CHANNELS - n_game_ch), dtype=np.float32
            )
            game_spatial = np.concatenate([game_spatial, pad], axis=-1)
        else:
            game_spatial = game_spatial[:, :, :GAME_SPATIAL_CHANNELS]

        # Append memory channels → (H, W, 13).
        # When memory is disabled, append MEMORY_CHANNELS zero planes so the
        # tensor shape is always (H, W, OBS_SPATIAL_CHANNELS).
        if memory is not None:
            mem_planes = memory.export_observation_planes(stack=True)  # (H, W, 4)
        else:
            mem_planes = np.zeros((map_h, map_w, MEMORY_CHANNELS), dtype=np.float32)

        return np.concatenate([game_spatial, mem_planes], axis=-1).astype(np.float32)

    # ── Global vector ─────────────────────────────────────────────────────────

    def _build_global_vec(
        self,
        obs_dict: dict,
        memory: Optional[EnemySpatialMemory],
        tick: int,
    ) -> np.ndarray:
        """Assemble the (33,) normalised scalar feature vector."""
        cfg = self.config
        eco = obs_dict.get("economy", {})
        mil = obs_dict.get("military", {})
        units: list[dict] = obs_dict.get("units", [])
        buildings: list[dict] = obs_dict.get("buildings", [])
        production: list[dict] = obs_dict.get("production", [])
        vis_enemies: list[dict] = obs_dict.get("visible_enemies", [])
        vis_enemy_bldgs: list[dict] = obs_dict.get("visible_enemy_buildings", [])

        power_provided = eco.get("power_provided", 0)
        power_drained = eco.get("power_drained", 0)
        resource_cap = max(eco.get("resource_capacity", 1), 1)
        kills_cost = mil.get("kills_cost", 0)
        deaths_cost = max(mil.get("deaths_cost", 1), 1)
        owned_types = {b["type"] for b in buildings if b.get("hp_percent", 1.0) > 0}
        result = obs_dict.get("result", "")

        prod_progress_mean = (
            float(np.mean([p.get("progress", 0.0) for p in production]))
            if production else 0.0
        )

        def _norm(val: float, scale: float, clip: float = 3.0) -> float:
            return float(np.clip(val / scale, 0.0, clip))

        vec = np.array([
            # Economy (4)
            _norm(eco.get("cash", 0), cfg.cash_scale),
            float(np.clip((power_provided - power_drained) / 100.0, -2.0, 2.0)),
            _norm(eco.get("harvester_count", 0), 5.0),
            float(eco.get("ore", 0)) / resource_cap,
            # Military (6)
            _norm(mil.get("army_value", 0), cfg.army_scale),
            _norm(mil.get("active_unit_count", 0), cfg.unit_scale),
            _norm(kills_cost, cfg.cost_scale),
            _norm(mil.get("deaths_cost", 0), cfg.cost_scale),
            _norm(mil.get("experience", 0), cfg.experience_scale),
            float(np.clip(kills_cost / deaths_cost, 0.0, 5.0) / 5.0),
            # Own assets (4)
            _norm(len(buildings), cfg.building_scale),
            _norm(len(units), cfg.unit_scale),
            float("fact" in owned_types),
            float("weap" in owned_types),
            # Production (3)
            _norm(len(production), 5.0),
            float(len(production) == 0),
            prod_progress_mean,
            # Enemies (2)
            _norm(len(vis_enemies), cfg.unit_scale),
            _norm(len(vis_enemy_bldgs), 10.0),
            # Game time & status (4)
            _norm(tick, cfg.tick_scale),
            float(power_drained > power_provided),
            float(any(u.get("type") == "harv" for u in units)),
            float(any(u.get("type") == "mcv" for u in units)),
            # Memory stats (2) — 0.0 when memory_enabled=False
            memory.coverage   if memory is not None else 0.0,
            memory.mean_decay if memory is not None else 0.0,
            # Result one-hot (3)
            float(result == "win"),
            float(result == "lose"),
            float(result == "draw"),
        ], dtype=np.float32)

        assert len(vec) == GLOBAL_VEC_SIZE  # sanity: caught at dev-time
        return vec

    # ── Gymnasium space ───────────────────────────────────────────────────────

    def observation_space(self, map_h: int, map_w: int):
        """Return a ``gymnasium.spaces.Dict`` observation space.

        Returns ``None`` if gymnasium is not installed.

        The spatial space bounds are deliberately loose ([0, 1] for most
        channels, except terrain which can be 0-8).  Algorithms that need
        tight bounds should subclass and override.
        """
        try:
            import gymnasium as gym  # type: ignore[import]
        except ImportError:
            return None

        return gym.spaces.Dict({
            "spatial": gym.spaces.Box(
                low=-1.0,
                high=8.0,
                shape=(map_h, map_w, OBS_SPATIAL_CHANNELS),
                dtype=np.float32,
            ),
            "global_vec": gym.spaces.Box(
                low=-2.0,
                high=3.0,
                shape=(GLOBAL_VEC_SIZE,),
                dtype=np.float32,
            ),
        })
