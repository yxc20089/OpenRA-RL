"""Unit tests for openra_env.enemy_memory.

Covers EnemySpatialMemory (single-env) and BatchedEnemySpatialMemory:
1. Memory updates correctly when an enemy is observed.
2. Age increments each step.
3. Decay decreases monotonically.
4. Memory clears at max_age.
5. Observation planes have the correct shapes.
6. Disabled config (memory_enabled=False) produces all-zero memory channels
   identical to the baseline (no-memory) observation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from openra_env.enemy_memory import (
    BatchedEnemySpatialMemory,
    EnemySpatialMemory,
    benchmark_step_latency,
    make_diffusion_hook,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mem(
    h: int = 16,
    w: int = 16,
    max_age: int = 20,
    decay_lambda: float = 0.05,
) -> EnemySpatialMemory:
    """Construct a fresh EnemySpatialMemory with sensible test defaults."""
    return EnemySpatialMemory(
        height=h,
        width=w,
        max_age=max_age,
        decay_lambda=decay_lambda,
    )


def _stamp(mem: EnemySpatialMemory, *cells: tuple[int, int]) -> None:
    """Observe a list of (row, col) cells in one call."""
    mem.observe_enemy_positions(list(cells))


# ── 1. Memory updates correctly when enemy observed ───────────────────────────


class TestMemoryUpdatesOnObservation:
    """After observe_enemy_positions the stamped cells have the expected values."""

    def test_presence_set_to_1(self):
        mem = _mem()
        _stamp(mem, (3, 5))
        assert mem._presence[3, 5] == pytest.approx(1.0)

    def test_age_reset_to_0(self):
        mem = _mem()
        _stamp(mem, (3, 5))
        assert mem._age[3, 5] == pytest.approx(0.0)

    def test_decay_set_to_1(self):
        mem = _mem()
        _stamp(mem, (3, 5))
        assert mem._decay[3, 5] == pytest.approx(1.0)

    def test_ever_seen_set_to_1(self):
        mem = _mem()
        _stamp(mem, (3, 5))
        assert mem._ever_seen[3, 5] == pytest.approx(1.0)

    def test_unstamped_cells_remain_zero(self):
        mem = _mem()
        _stamp(mem, (0, 0))
        planes = mem.export_observation_planes()
        # Every cell except (0,0) should be inactive (presence == 0)
        mask = np.ones((mem.height, mem.width), dtype=bool)
        mask[0, 0] = False
        assert (planes["presence"][mask] == 0.0).all()

    def test_multiple_cells_all_stamped(self):
        mem = _mem()
        cells = [(0, 0), (7, 7), (15, 15)]
        mem.observe_enemy_positions(cells)
        for row, col in cells:
            assert mem._presence[row, col] == pytest.approx(1.0)
            assert mem._decay[row, col] == pytest.approx(1.0)
            assert mem._ever_seen[row, col] == pytest.approx(1.0)

    def test_out_of_bounds_cells_ignored(self):
        mem = _mem(h=8, w=8)
        mem.observe_enemy_positions([(-1, 0), (0, -1), (8, 0), (0, 8)])
        assert mem.active_cell_count == 0

    def test_restamp_refreshes_age(self):
        """Re-observing a previously aged cell resets its age to 0."""
        mem = _mem()
        _stamp(mem, (2, 2))
        mem.step_decay()  # age goes to 1
        assert mem._age[2, 2] == pytest.approx(1.0)
        _stamp(mem, (2, 2))  # re-stamp
        assert mem._age[2, 2] == pytest.approx(0.0)
        assert mem._decay[2, 2] == pytest.approx(1.0)

    def test_ever_seen_persists_after_expiry(self):
        """ever_seen is never cleared, even when a cell expires."""
        mem = _mem(max_age=2)
        _stamp(mem, (1, 1))
        mem.step_decay()  # age=1
        mem.step_decay()  # age=2 → expired
        assert mem._presence[1, 1] == pytest.approx(0.0)
        assert mem._ever_seen[1, 1] == pytest.approx(1.0)


# ── 2. Age increments each step ───────────────────────────────────────────────


class TestAgeIncrements:
    """_age increments by 1 per step_decay() call for active cells."""

    def test_age_is_0_immediately_after_stamp(self):
        mem = _mem()
        _stamp(mem, (0, 0))
        assert mem._age[0, 0] == pytest.approx(0.0)

    def test_age_increments_by_1_after_one_step(self):
        mem = _mem()
        _stamp(mem, (0, 0))
        mem.step_decay()
        assert mem._age[0, 0] == pytest.approx(1.0)

    def test_age_increments_by_1_per_step(self):
        mem = _mem(max_age=100)
        _stamp(mem, (4, 4))
        for expected_age in range(1, 11):
            mem.step_decay()
            assert mem._age[4, 4] == pytest.approx(float(expected_age))

    def test_inactive_cells_age_stays_0(self):
        mem = _mem()
        _stamp(mem, (0, 0))
        mem.step_decay()
        # Cell (1,1) was never stamped — its age must remain 0
        assert mem._age[1, 1] == pytest.approx(0.0)

    def test_age_resets_to_0_on_restamp(self):
        mem = _mem(max_age=100)
        _stamp(mem, (5, 5))
        for _ in range(7):
            mem.step_decay()
        assert mem._age[5, 5] == pytest.approx(7.0)
        _stamp(mem, (5, 5))
        assert mem._age[5, 5] == pytest.approx(0.0)

    def test_age_norm_tracks_age(self):
        mem = _mem(max_age=10)
        _stamp(mem, (0, 0))
        n_steps = 8
        for _ in range(n_steps):
            mem.step_decay()
        planes = mem.export_observation_planes()
        expected = min(n_steps / 10, 1.0)
        assert planes["age_norm"][0, 0] == pytest.approx(expected)

    def test_step_decay_on_empty_memory_is_safe(self):
        mem = _mem()
        mem.step_decay()  # no active cells — should not raise
        assert mem.active_cell_count == 0


# ── 3. Decay decreases monotonically ─────────────────────────────────────────


class TestDecayDecreasesMonotonically:
    """_decay *= exp(-lambda) each step; after N steps = exp(-lambda * N)."""

    def test_decay_is_1_after_stamp(self):
        mem = _mem(decay_lambda=0.05)
        _stamp(mem, (0, 0))
        assert mem._decay[0, 0] == pytest.approx(1.0)

    def test_decay_after_one_step(self):
        lam = 0.10
        mem = _mem(decay_lambda=lam)
        _stamp(mem, (0, 0))
        mem.step_decay()
        assert mem._decay[0, 0] == pytest.approx(math.exp(-lam), rel=1e-5)

    def test_decay_compounded_after_n_steps(self):
        lam = 0.05
        n = 10
        mem = _mem(max_age=100, decay_lambda=lam)
        _stamp(mem, (2, 3))
        for _ in range(n):
            mem.step_decay()
        assert mem._decay[2, 3] == pytest.approx(math.exp(-lam * n), rel=1e-5)

    def test_decay_is_strictly_decreasing(self):
        mem = _mem(max_age=50, decay_lambda=0.05)
        _stamp(mem, (0, 0))
        prev = 1.0
        for _ in range(20):
            mem.step_decay()
            current = float(mem._decay[0, 0])
            assert current < prev
            prev = current

    def test_zero_lambda_no_decay(self):
        """decay_lambda=0 means the decay multiplier is exp(0)=1 — no decay."""
        mem = _mem(max_age=50, decay_lambda=0.0)
        _stamp(mem, (0, 0))
        for _ in range(10):
            mem.step_decay()
        assert mem._decay[0, 0] == pytest.approx(1.0)

    def test_decay_bounded_in_0_1(self):
        mem = _mem(max_age=100, decay_lambda=0.02)
        _stamp(mem, (0, 0))
        for _ in range(50):
            mem.step_decay()
        d = float(mem._decay[0, 0])
        assert 0.0 <= d <= 1.0

    def test_inactive_cell_decay_is_zero(self):
        mem = _mem()
        _stamp(mem, (0, 0))
        mem.step_decay()
        # Cell (1,1) was never stamped
        assert mem._decay[1, 1] == pytest.approx(0.0)


# ── 4. Memory clears at max_age ───────────────────────────────────────────────


class TestMemoryClearsAtMaxAge:
    """All core planes except ever_seen are zeroed when age reaches max_age."""

    def test_cell_clears_at_max_age(self):
        max_age = 5
        mem = _mem(max_age=max_age)
        _stamp(mem, (3, 3))
        for _ in range(max_age):
            mem.step_decay()
        assert mem._presence[3, 3] == pytest.approx(0.0)
        assert mem._age[3, 3] == pytest.approx(0.0)
        assert mem._decay[3, 3] == pytest.approx(0.0)

    def test_cell_not_cleared_before_max_age(self):
        max_age = 5
        mem = _mem(max_age=max_age)
        _stamp(mem, (3, 3))
        for _ in range(max_age - 1):
            mem.step_decay()
        assert mem._presence[3, 3] == pytest.approx(1.0)

    def test_ever_seen_not_cleared_at_max_age(self):
        max_age = 3
        mem = _mem(max_age=max_age)
        _stamp(mem, (1, 1))
        for _ in range(max_age):
            mem.step_decay()
        assert mem._ever_seen[1, 1] == pytest.approx(1.0)

    def test_active_count_drops_to_zero_on_expiry(self):
        mem = _mem(max_age=3)
        _stamp(mem, (0, 0))
        for _ in range(3):
            mem.step_decay()
        assert mem.active_cell_count == 0

    def test_restamp_after_expiry_re_activates_cell(self):
        max_age = 2
        mem = _mem(max_age=max_age)
        _stamp(mem, (2, 2))
        for _ in range(max_age):
            mem.step_decay()
        assert mem._presence[2, 2] == pytest.approx(0.0)
        # Stamp again after expiry
        _stamp(mem, (2, 2))
        assert mem._presence[2, 2] == pytest.approx(1.0)
        assert mem._decay[2, 2] == pytest.approx(1.0)
        assert mem._age[2, 2] == pytest.approx(0.0)

    def test_max_age_1_expires_after_one_step(self):
        mem = _mem(max_age=1)
        _stamp(mem, (0, 0))
        mem.step_decay()
        assert mem._presence[0, 0] == pytest.approx(0.0)

    def test_multiple_cells_expire_independently(self):
        mem = _mem(max_age=4)
        _stamp(mem, (0, 0))
        mem.step_decay()
        mem.step_decay()
        _stamp(mem, (1, 1))  # second cell stamped at step 2
        for _ in range(2):   # two more steps: cell (0,0) reaches age 4, expires
            mem.step_decay()
        assert mem._presence[0, 0] == pytest.approx(0.0), "first cell should have expired"
        assert mem._presence[1, 1] == pytest.approx(1.0), "second cell should still be active"


# ── 5. Observation planes have correct shapes ─────────────────────────────────


class TestObservationPlaneShapes:
    """export_observation_planes returns correctly shaped float32 arrays."""

    def test_dict_mode_core_planes_present(self):
        mem = _mem(h=10, w=12)
        planes = mem.export_observation_planes()
        assert set(EnemySpatialMemory.CORE_PLANES).issubset(planes.keys())

    def test_dict_mode_each_plane_shape(self):
        H, W = 10, 12
        mem = _mem(h=H, w=W)
        planes = mem.export_observation_planes()
        for name in EnemySpatialMemory.CORE_PLANES:
            assert planes[name].shape == (H, W), f"plane {name!r} shape mismatch"

    def test_dict_mode_dtype_float32(self):
        mem = _mem()
        planes = mem.export_observation_planes()
        for name, arr in planes.items():
            assert arr.dtype == np.float32, f"plane {name!r} dtype mismatch"

    def test_stack_mode_shape(self):
        H, W = 8, 16
        mem = _mem(h=H, w=W)
        stacked = mem.export_observation_planes(stack=True)
        n_core = len(EnemySpatialMemory.CORE_PLANES)
        assert stacked.shape == (H, W, n_core)

    def test_stack_mode_dtype_float32(self):
        mem = _mem()
        stacked = mem.export_observation_planes(stack=True)
        assert stacked.dtype == np.float32

    def test_with_threat_plane_stack_shape(self):
        H, W = 6, 6
        mem = EnemySpatialMemory(H, W, max_age=10, decay_lambda=0.05, track_threat=True)
        stacked = mem.export_observation_planes(stack=True)
        # core planes + threat
        expected_channels = len(EnemySpatialMemory.CORE_PLANES) + 1
        assert stacked.shape == (H, W, expected_channels)

    def test_with_extra_plane_stack_shape(self):
        H, W = 4, 4
        mem = _mem(h=H, w=W)
        mem.register_plane("heat")
        stacked = mem.export_observation_planes(stack=True)
        expected_channels = len(EnemySpatialMemory.CORE_PLANES) + 1
        assert stacked.shape == (H, W, expected_channels)

    def test_planes_are_copies_not_views(self):
        """Modifying returned planes should not corrupt internal state."""
        mem = _mem()
        _stamp(mem, (0, 0))
        planes = mem.export_observation_planes()
        planes["presence"][:] = 99.0
        assert mem._presence[0, 0] == pytest.approx(1.0)

    def test_all_planes_zero_after_reset(self):
        mem = _mem()
        _stamp(mem, (0, 0))
        mem.reset()
        planes = mem.export_observation_planes()
        for name, arr in planes.items():
            assert (arr == 0.0).all(), f"plane {name!r} not zero after reset"

    def test_inactive_cells_presence_zero(self):
        mem = _mem(h=4, w=4)
        _stamp(mem, (2, 2))
        planes = mem.export_observation_planes()
        mask = np.ones((4, 4), dtype=bool)
        mask[2, 2] = False
        assert (planes["presence"][mask] == 0.0).all()


# ── 6. Disabled config produces identical all-zero memory channels ────────────


class TestDisabledConfigProducesZeroMemory:
    """When memory is not used, all memory planes must be zero.

    This mirrors the ObsTensorConfig(memory_enabled=False) behaviour:
    memory channels 9–12 in the spatial tensor remain 0.0 throughout
    training, making the disabled run a true no-memory baseline.

    We test this by constructing the memory but never calling step_decay()
    or observe_enemy_positions() — i.e. treating it as disabled — and
    verifying that export_observation_planes() returns all-zero arrays.
    We also verify that a freshly-reset memory matches a brand-new memory.
    """

    def test_fresh_memory_all_zero_stacked(self):
        mem = _mem()
        stacked = mem.export_observation_planes(stack=True)
        assert (stacked == 0.0).all()

    def test_fresh_memory_all_zero_dict(self):
        mem = _mem()
        planes = mem.export_observation_planes()
        for name, arr in planes.items():
            assert (arr == 0.0).all(), f"plane {name!r} not zero on fresh memory"

    def test_reset_restores_all_zero(self):
        mem = _mem()
        _stamp(mem, (0, 0), (3, 3))
        mem.step_decay()
        mem.reset()
        stacked = mem.export_observation_planes(stack=True)
        assert (stacked == 0.0).all()

    def test_disabled_matches_baseline(self):
        """Two fresh memories should be identical (simulate disabled vs baseline)."""
        disabled = _mem(h=8, w=8)
        baseline = _mem(h=8, w=8)
        stacked_disabled = disabled.export_observation_planes(stack=True)
        stacked_baseline = baseline.export_observation_planes(stack=True)
        np.testing.assert_array_equal(stacked_disabled, stacked_baseline)

    def test_obs_tensor_memory_channels_zero_when_disabled(self):
        """Integration: ObsTensorBuilder with memory_enabled=False has zero memory channels."""
        from openra_env.obs_tensor import ObsTensorBuilder, ObsTensorConfig

        cfg = ObsTensorConfig(memory_enabled=False)
        builder = ObsTensorBuilder(config=cfg)
        builder.reset(map_h=8, map_w=8)

        import base64

        H, W = 8, 8
        # Minimal observation dict with map_info so build() knows dimensions
        game_spatial_bytes = bytes(H * W * 9 * 4)  # 9 game channels, zeros
        obs = {
            "tick": 0,
            "cash": 0,
            "map_info": {"height": H, "width": W},
            "units": [],
            "buildings": [],
            "production": [],
            "visible_enemies": [],
            "visible_enemy_buildings": [],
            "available_production": [],
            "spatial_map": base64.b64encode(game_spatial_bytes).decode("ascii"),
            "result": "",
            "done": False,
        }
        out = builder.build(obs)
        spatial = out["spatial"]
        # Channels 9-12 are memory channels — must be zero when disabled
        from openra_env.obs_tensor import CH_MEM_PRESENCE, CH_MEM_EVER_SEEN
        mem_channels = spatial[:, :, CH_MEM_PRESENCE : CH_MEM_EVER_SEEN + 1]
        assert (mem_channels == 0.0).all()

    def test_obs_tensor_memory_channels_nonzero_when_enabled(self):
        """Sanity: when memory_enabled=True, stamped cells produce nonzero channels."""
        from openra_env.obs_tensor import ObsTensorBuilder, ObsTensorConfig

        cfg = ObsTensorConfig(memory_enabled=True, memory_max_age=50, memory_decay_lambda=0.02)
        builder = ObsTensorBuilder(config=cfg)
        builder.reset(map_h=8, map_w=8)

        import base64

        H, W = 8, 8
        game_spatial_bytes = bytes(H * W * 9 * 4)
        obs = {
            "tick": 0,
            "cash": 0,
            "map_info": {"height": H, "width": W},
            "units": [],
            "buildings": [],
            "production": [],
            "visible_enemies": [{"cell_x": 2, "cell_y": 3}],
            "visible_enemy_buildings": [],
            "available_production": [],
            "spatial_map": base64.b64encode(game_spatial_bytes).decode("ascii"),
            "result": "",
            "done": False,
        }
        out = builder.build(obs)
        spatial = out["spatial"]
        from openra_env.obs_tensor import CH_MEM_PRESENCE
        # Cell (row=3, col=2) should have presence=1
        assert spatial[3, 2, CH_MEM_PRESENCE] == pytest.approx(1.0)


# ── 7. Bonus: diffusion hook integration ─────────────────────────────────────


class TestDiffusionHookIntegration:
    """make_diffusion_hook works correctly as a post-step hook."""

    def test_noop_hook_for_sigma_zero(self):
        mem = _mem()
        hook = make_diffusion_hook(sigma=0.0)
        _stamp(mem, (7, 7))
        before = mem._presence.copy()
        hook(mem)
        np.testing.assert_array_equal(mem._presence, before)

    def test_diffusion_spreads_presence(self):
        """After diffusion, neighbouring cells of a stamped cell get nonzero presence."""
        mem = _mem(h=16, w=16)
        mem.add_post_step_hook(make_diffusion_hook(sigma=1.0, presence_threshold=0.0))
        _stamp(mem, (8, 8))
        mem.step_decay()
        # After one step with diffusion the immediate neighbours should be nonzero
        assert mem._presence[7, 8] > 0.0 or mem._presence[8, 7] > 0.0

    def test_diffusion_clips_presence_to_1(self):
        mem = _mem()
        mem.add_post_step_hook(make_diffusion_hook(sigma=2.0))
        _stamp(mem, (0, 0))
        mem.step_decay()
        assert mem._presence.max() <= 1.0

    def test_diffusion_does_not_affect_ever_seen(self):
        mem = _mem(h=8, w=8)
        mem.add_post_step_hook(make_diffusion_hook(sigma=1.0, presence_threshold=0.0))
        _stamp(mem, (4, 4))
        ever_seen_before = mem._ever_seen.copy()
        mem.step_decay()
        # Diffusion must not modify ever_seen
        np.testing.assert_array_equal(mem._ever_seen, ever_seen_before)

    def test_diffusion_does_not_affect_age(self):
        mem = _mem(h=8, w=8)
        mem.add_post_step_hook(make_diffusion_hook(sigma=1.0))
        _stamp(mem, (4, 4))
        mem.step_decay()
        # Only the stamped cell (now aged 1) should have nonzero age
        assert mem._age[4, 4] == pytest.approx(1.0)
        # Neighbouring cells should NOT have their age changed by diffusion
        assert mem._age[3, 4] == pytest.approx(0.0)
        assert mem._age[4, 3] == pytest.approx(0.0)


# ── BatchedEnemySpatialMemory ─────────────────────────────────────────────────


def _batch(
    B: int = 4,
    H: int = 16,
    W: int = 16,
    max_age: int = 20,
    decay_lambda: float = 0.05,
) -> BatchedEnemySpatialMemory:
    return BatchedEnemySpatialMemory(B, H, W, max_age=max_age, decay_lambda=decay_lambda)


class TestBatchedConstruction:
    def test_repr(self):
        b = _batch()
        assert "BatchedEnemySpatialMemory" in repr(b)
        assert "numpy[cpu]" in repr(b)

    def test_shapes_after_init(self):
        B, H, W = 4, 8, 10
        b = BatchedEnemySpatialMemory(B, H, W, max_age=10, decay_lambda=0.1)
        assert b._presence.shape  == (B, H, W)
        assert b._age.shape       == (B, H, W)
        assert b._decay.shape     == (B, H, W)
        assert b._ever_seen.shape == (B, H, W)

    def test_all_zero_after_init(self):
        b = _batch()
        assert (b._presence  == 0.0).all()
        assert (b._age       == 0.0).all()
        assert (b._decay     == 0.0).all()
        assert (b._ever_seen == 0.0).all()

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError):
            BatchedEnemySpatialMemory(0, 8, 8, max_age=10, decay_lambda=0.1)

    def test_invalid_dimensions_raise(self):
        with pytest.raises(ValueError):
            BatchedEnemySpatialMemory(4, 0, 8, max_age=10, decay_lambda=0.1)


class TestBatchedReset:
    def test_full_reset_clears_all(self):
        b = _batch(B=4)
        b.observe_batch([0, 1, 2, 3], [[(0, 0)], [(1, 1)], [(2, 2)], [(3, 3)]])
        b.reset()
        assert (b._presence == 0.0).all()
        assert (b._ever_seen == 0.0).all()

    def test_partial_reset_clears_selected(self):
        b = _batch(B=4)
        b.observe_batch([0, 1, 2, 3], [[(0, 0)], [(1, 1)], [(2, 2)], [(3, 3)]])
        b.reset(env_ids=[1, 3])
        # envs 0 and 2 retain their state
        assert b._presence[0, 0, 0] == pytest.approx(1.0)
        assert b._presence[2, 2, 2] == pytest.approx(1.0)
        # envs 1 and 3 are cleared
        assert (b._presence[1] == 0.0).all()
        assert (b._presence[3] == 0.0).all()

    def test_ever_seen_cleared_on_reset(self):
        b = _batch(B=2)
        b.observe_batch([0], [[(5, 5)]])
        b.reset()
        assert (b._ever_seen == 0.0).all()


class TestBatchedObserve:
    def test_single_env_stamp(self):
        b = _batch(B=4)
        b.observe_batch([2], [[(3, 7)]])
        assert b._presence[2, 3, 7] == pytest.approx(1.0)
        assert b._decay[2, 3, 7]    == pytest.approx(1.0)
        assert b._age[2, 3, 7]      == pytest.approx(0.0)
        assert b._ever_seen[2, 3, 7] == pytest.approx(1.0)

    def test_other_envs_unaffected(self):
        b = _batch(B=4)
        b.observe_batch([1], [[(0, 0)]])
        assert (b._presence[0] == 0.0).all()
        assert (b._presence[2] == 0.0).all()
        assert (b._presence[3] == 0.0).all()

    def test_multiple_envs_simultaneously(self):
        b = _batch(B=4)
        b.observe_batch([0, 2], [[(1, 1)], [(5, 5)]])
        assert b._presence[0, 1, 1] == pytest.approx(1.0)
        assert b._presence[2, 5, 5] == pytest.approx(1.0)
        assert (b._presence[1] == 0.0).all()
        assert (b._presence[3] == 0.0).all()

    def test_multiple_cells_per_env(self):
        b = _batch(B=2)
        b.observe_batch([0], [[(0, 0), (1, 1), (2, 2)]])
        assert b._presence[0, 0, 0] == pytest.approx(1.0)
        assert b._presence[0, 1, 1] == pytest.approx(1.0)
        assert b._presence[0, 2, 2] == pytest.approx(1.0)

    def test_out_of_bounds_ignored(self):
        b = _batch(B=2, H=8, W=8)
        b.observe_batch([0], [[(-1, 0), (0, -1), (8, 0), (0, 8)]])
        assert (b._presence[0] == 0.0).all()

    def test_empty_cell_list_is_safe(self):
        b = _batch(B=2)
        b.observe_batch([0], [[]])   # should not raise
        assert (b._presence[0] == 0.0).all()


class TestBatchedStepDecay:
    def test_age_increments_for_active_cells(self):
        b = _batch(B=2, max_age=50)
        b.observe_batch([0, 1], [[(0, 0)], [(5, 5)]])
        b.step_decay()
        assert b._age[0, 0, 0] == pytest.approx(1.0)
        assert b._age[1, 5, 5] == pytest.approx(1.0)

    def test_inactive_cells_age_stays_zero(self):
        b = _batch(B=2)
        b.observe_batch([0], [[(0, 0)]])
        b.step_decay()
        assert b._age[1, 0, 0] == pytest.approx(0.0)

    def test_decay_applied_correctly(self):
        import math
        lam = 0.10
        b = BatchedEnemySpatialMemory(2, 8, 8, max_age=50, decay_lambda=lam)
        b.observe_batch([0], [[(3, 3)]])
        b.step_decay()
        assert b._decay[0, 3, 3] == pytest.approx(math.exp(-lam), rel=1e-5)

    def test_decay_does_not_affect_inactive_envs(self):
        b = _batch(B=3, max_age=50)
        b.observe_batch([1], [[(0, 0)]])
        b.step_decay()
        assert (b._decay[0] == 0.0).all()
        assert (b._decay[2] == 0.0).all()

    def test_expiry_clears_all_planes(self):
        b = _batch(B=2, max_age=3)
        b.observe_batch([0], [[(4, 4)]])
        for _ in range(3):
            b.step_decay()
        assert b._presence[0, 4, 4] == pytest.approx(0.0)
        assert b._age[0, 4, 4]      == pytest.approx(0.0)
        assert b._decay[0, 4, 4]    == pytest.approx(0.0)

    def test_step_decay_on_empty_batch_is_safe(self):
        b = _batch(B=4)
        b.step_decay()   # no active cells — should not raise
        assert (b._presence == 0.0).all()

    def test_envs_expire_independently(self):
        b = _batch(B=2, max_age=3)
        b.observe_batch([0], [[(0, 0)]])
        b.step_decay()
        b.step_decay()
        # Stamp env 1 after 2 steps (env 0 expires at step 3, env 1 should survive)
        b.observe_batch([1], [[(0, 0)]])
        b.step_decay()   # env 0: age=3 → expired; env 1: age=1, still active
        assert b._presence[0, 0, 0] == pytest.approx(0.0), "env 0 should have expired"
        assert b._presence[1, 0, 0] == pytest.approx(1.0), "env 1 should still be active"


class TestBatchedExport:
    def test_shape_default_channel_last(self):
        B, H, W = 3, 8, 10
        b = BatchedEnemySpatialMemory(B, H, W, max_age=10, decay_lambda=0.05)
        out = b.export_observation_planes()
        assert out.shape == (B, H, W, 4)

    def test_shape_channel_first(self):
        B, H, W = 3, 8, 10
        b = BatchedEnemySpatialMemory(B, H, W, max_age=10, decay_lambda=0.05)
        out = b.export_observation_planes(channel_first=True)
        assert out.shape == (B, 4, H, W)

    def test_dtype_float32(self):
        b = _batch()
        out = b.export_observation_planes()
        assert out.dtype == np.float32

    def test_all_zero_on_fresh_batch(self):
        b = _batch()
        out = b.export_observation_planes()
        assert (out == 0.0).all()

    def test_stamped_cell_presence_one(self):
        b = _batch(B=4)
        b.observe_batch([2], [[(3, 7)]])
        out = b.export_observation_planes()   # (B, H, W, 4)
        # channel 0 = presence
        assert out[2, 3, 7, 0] == pytest.approx(1.0)

    def test_unstamped_cells_presence_zero(self):
        b = _batch(B=4)
        b.observe_batch([2], [[(3, 7)]])
        out = b.export_observation_planes()
        # All other envs should have zero presence
        assert (out[0, :, :, 0] == 0.0).all()
        assert (out[1, :, :, 0] == 0.0).all()
        assert (out[3, :, :, 0] == 0.0).all()

    def test_env_ids_subset(self):
        b = _batch(B=4)
        b.observe_batch([0, 1, 2, 3], [[(i, i)] for i in range(4)])
        out = b.export_observation_planes(env_ids=[1, 3])
        assert out.shape[0] == 2   # only 2 envs
        assert out[0, 1, 1, 0] == pytest.approx(1.0)   # env 1 → index 0
        assert out[1, 3, 3, 0] == pytest.approx(1.0)   # env 3 → index 1

    def test_age_norm_channel(self):
        b = _batch(B=2, max_age=10)
        b.observe_batch([0], [[(0, 0)]])
        b.step_decay()   # age=1
        out = b.export_observation_planes()
        assert out[0, 0, 0, 1] == pytest.approx(1.0 / 10.0)   # channel 1 = age_norm

    def test_ever_seen_not_cleared_after_expiry(self):
        b = _batch(B=2, max_age=2)
        b.observe_batch([0], [[(5, 5)]])
        b.step_decay()
        b.step_decay()   # expires
        out = b.export_observation_planes()
        assert out[0, 5, 5, 0] == pytest.approx(0.0)    # presence cleared
        assert out[0, 5, 5, 3] == pytest.approx(1.0)    # ever_seen preserved


class TestBatchedSummaryStats:
    def test_active_cell_counts_zero_initially(self):
        b = _batch(B=3)
        counts = b.active_cell_counts
        assert counts.shape == (3,)
        assert (counts == 0).all()

    def test_active_cell_counts_after_stamp(self):
        b = _batch(B=3)
        b.observe_batch([1], [[(0, 0), (1, 1)]])
        counts = b.active_cell_counts
        assert counts[0] == 0
        assert counts[1] == 2
        assert counts[2] == 0

    def test_coverage_zero_initially(self):
        b = _batch(B=2)
        cov = b.coverage
        np.testing.assert_allclose(cov, 0.0)

    def test_coverage_after_stamp(self):
        b = _batch(B=1, H=4, W=4)
        b.observe_batch([0], [[(0, 0)]])   # 1 of 16 cells
        cov = b.coverage
        assert cov[0] == pytest.approx(1.0 / 16.0)


# ── benchmark_step_latency ────────────────────────────────────────────────────


class TestBenchmarkStepLatency:
    def test_returns_expected_keys(self):
        results = benchmark_step_latency(
            height=32, width=32, batch_size=4, n_steps=50,
            n_enemies_per_env=5, print_results=False,
        )
        expected = {
            "single_step_us", "single_observe_us", "single_total_us",
            "batch_step_us", "batch_observe_us", "batch_total_us",
            "batch_per_env_us", "speedup",
        }
        assert expected.issubset(results.keys())

    def test_all_values_positive(self):
        results = benchmark_step_latency(
            height=16, width=16, batch_size=2, n_steps=20,
            n_enemies_per_env=3, print_results=False,
        )
        for k, v in results.items():
            assert v > 0, f"{k} should be positive, got {v}"

    def test_speedup_is_reasonable(self):
        """Batched per-env cost should be ≤ single-env cost (possibly ±noise)."""
        results = benchmark_step_latency(
            height=64, width=64, batch_size=8, n_steps=100,
            n_enemies_per_env=10, print_results=False,
        )
        # batch_per_env_us < single_total_us or at worst within 5× (CI machines are noisy)
        assert results["speedup"] > 0.2
