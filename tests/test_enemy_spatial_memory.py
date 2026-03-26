"""Tests for openra_env.enemy_spatial_memory."""

from __future__ import annotations

import numpy as np
import pytest

from openra_env.enemy_spatial_memory import (
    CH_AGE_NORM,
    CH_DECAY,
    CH_PRESENCE,
    CHANNEL_NAMES,
    EnemySpatialMemory,
    EnemySpatialMemoryConfig,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_unit(cx: int, cy: int) -> dict:
    return {"cell_x": cx, "cell_y": cy}


def _make_mem(h: int = 8, w: int = 8, decay: float = 0.9, max_age: int = 10) -> EnemySpatialMemory:
    cfg = EnemySpatialMemoryConfig(decay_factor=decay, max_age=max_age)
    return EnemySpatialMemory(map_h=h, map_w=w, config=cfg)


# ── Config defaults ───────────────────────────────────────────────────────────

class TestEnemySpatialMemoryConfig:
    def test_defaults(self):
        cfg = EnemySpatialMemoryConfig()
        assert cfg.decay_factor == 0.98
        assert cfg.max_age == 200

    def test_custom(self):
        cfg = EnemySpatialMemoryConfig(decay_factor=0.5, max_age=50)
        assert cfg.decay_factor == 0.5
        assert cfg.max_age == 50


# ── Module constants ──────────────────────────────────────────────────────────

class TestModuleConstants:
    def test_channel_indices(self):
        assert CH_PRESENCE == 0
        assert CH_AGE_NORM == 1
        assert CH_DECAY == 2

    def test_channel_names_length(self):
        assert len(CHANNEL_NAMES) == 3

    def test_channel_names_content(self):
        assert CHANNEL_NAMES[0] == "enemy_last_seen_presence"
        assert CHANNEL_NAMES[1] == "enemy_last_seen_age_norm"
        assert CHANNEL_NAMES[2] == "enemy_last_seen_decay"


# ── Construction and reset ────────────────────────────────────────────────────

class TestConstruction:
    def test_arrays_zeroed_on_init(self):
        mem = _make_mem()
        ch = mem.get_channels()
        assert ch.shape == (8, 8, 3)
        assert (ch == 0.0).all()

    def test_reset_clears_state(self):
        mem = _make_mem()
        mem.step([_make_unit(2, 3)], [])
        mem.reset()
        ch = mem.get_channels()
        assert (ch == 0.0).all()

    def test_default_config_used_when_none(self):
        mem = EnemySpatialMemory(4, 4)
        assert mem.config.decay_factor == 0.98
        assert mem.config.max_age == 200

    def test_map_dimensions_stored(self):
        mem = EnemySpatialMemory(map_h=16, map_w=32)
        assert mem.map_h == 16
        assert mem.map_w == 32


# ── Stamping (step with new sightings) ────────────────────────────────────────

class TestStamp:
    def test_fresh_cell_presence_1(self):
        mem = _make_mem()
        mem.step([_make_unit(3, 5)], [])
        ch = mem.get_channels()
        assert ch[5, 3, CH_PRESENCE] == 1.0

    def test_fresh_cell_age_norm_0(self):
        mem = _make_mem()
        mem.step([_make_unit(3, 5)], [])
        ch = mem.get_channels()
        assert ch[5, 3, CH_AGE_NORM] == 0.0

    def test_fresh_cell_decay_1(self):
        mem = _make_mem()
        mem.step([_make_unit(3, 5)], [])
        ch = mem.get_channels()
        assert ch[5, 3, CH_DECAY] == 1.0

    def test_building_stamped(self):
        mem = _make_mem()
        mem.step([], [_make_unit(1, 2)])
        ch = mem.get_channels()
        assert ch[2, 1, CH_PRESENCE] == 1.0
        assert ch[2, 1, CH_DECAY] == 1.0

    def test_multiple_cells_stamped(self):
        mem = _make_mem()
        mem.step([_make_unit(0, 0), _make_unit(7, 7)], [])
        ch = mem.get_channels()
        assert ch[0, 0, CH_PRESENCE] == 1.0
        assert ch[7, 7, CH_PRESENCE] == 1.0

    def test_out_of_bounds_ignored(self):
        mem = _make_mem(h=4, w=4)
        mem.step([_make_unit(-1, 0), _make_unit(0, -1), _make_unit(4, 0), _make_unit(0, 4)], [])
        ch = mem.get_channels()
        assert (ch[:, :, CH_PRESENCE] == 0.0).all()

    def test_missing_cell_keys_ignored(self):
        mem = _make_mem()
        mem.step([{}], [])  # no cell_x / cell_y
        ch = mem.get_channels()
        assert (ch[:, :, CH_PRESENCE] == 0.0).all()


# ── Age and decay over multiple steps ─────────────────────────────────────────

class TestAgeDecay:
    def test_age_norm_increases_each_step(self):
        mem = _make_mem(max_age=10)
        mem.step([_make_unit(2, 2)], [])  # stamp at step 0; age=0 after stamp
        # Step 2: age advances to 1 first, then no re-stamp → age_norm = 1/10
        mem.step([], [])
        ch = mem.get_channels()
        assert ch[2, 2, CH_AGE_NORM] == pytest.approx(1 / 10)

    def test_decay_applied_each_step(self):
        mem = _make_mem(decay=0.9, max_age=10)
        mem.step([_make_unit(0, 0)], [])  # decay=1.0 after stamp
        mem.step([], [])                  # decay *= 0.9 → 0.9
        ch = mem.get_channels()
        assert ch[0, 0, CH_DECAY] == pytest.approx(0.9)

    def test_decay_compounded_over_steps(self):
        mem = _make_mem(decay=0.8, max_age=20)
        mem.step([_make_unit(1, 1)], [])
        for _ in range(4):
            mem.step([], [])
        ch = mem.get_channels()
        expected = 0.8 ** 4
        assert ch[1, 1, CH_DECAY] == pytest.approx(expected, rel=1e-5)

    def test_presence_stays_1_until_expiry(self):
        mem = _make_mem(max_age=5)
        mem.step([_make_unit(3, 3)], [])
        for _ in range(4):  # 4 more steps (age reaches 4 — still < 5)
            mem.step([], [])
        ch = mem.get_channels()
        assert ch[3, 3, CH_PRESENCE] == 1.0

    def test_inactive_cells_all_zero(self):
        mem = _make_mem()
        mem.step([_make_unit(0, 0)], [])
        ch = mem.get_channels()
        # All cells except (0,0) should be zero across all channels
        mask = np.ones((8, 8), dtype=bool)
        mask[0, 0] = False
        assert (ch[mask] == 0.0).all()


# ── Max-age expiry ────────────────────────────────────────────────────────────

class TestExpiry:
    def test_cell_cleared_at_max_age(self):
        mem = _make_mem(max_age=3)
        mem.step([_make_unit(4, 4)], [])   # age=0
        mem.step([], [])                    # age=1
        mem.step([], [])                    # age=2
        mem.step([], [])                    # age=3 → expired
        ch = mem.get_channels()
        assert ch[4, 4, CH_PRESENCE] == 0.0
        assert ch[4, 4, CH_AGE_NORM] == 0.0
        assert ch[4, 4, CH_DECAY] == 0.0

    def test_cell_not_cleared_before_max_age(self):
        mem = _make_mem(max_age=3)
        mem.step([_make_unit(4, 4)], [])   # age=0
        mem.step([], [])                    # age=1
        mem.step([], [])                    # age=2 — still valid
        ch = mem.get_channels()
        assert ch[4, 4, CH_PRESENCE] == 1.0

    def test_fresh_stamp_resets_age_after_expiry(self):
        mem = _make_mem(max_age=2)
        mem.step([_make_unit(1, 1)], [])   # age=0
        mem.step([], [])                    # age=1
        mem.step([], [])                    # age=2 → expired
        # Now re-stamp in same step as expiry check runs before stamp
        mem.step([_make_unit(1, 1)], [])   # re-stamp → age=0, decay=1
        ch = mem.get_channels()
        assert ch[1, 1, CH_PRESENCE] == 1.0
        assert ch[1, 1, CH_AGE_NORM] == 0.0
        assert ch[1, 1, CH_DECAY] == pytest.approx(1.0)

    def test_fresh_stamp_never_cleared_same_step(self):
        """Step 3 runs after expiry, so a freshly stamped cell survives."""
        mem = _make_mem(max_age=1)
        # max_age=1: after 1 step without re-sighting, cell expires.
        # But if we re-stamp at the same step it would expire, it survives.
        mem.step([_make_unit(2, 2)], [])   # age=0
        mem.step([_make_unit(2, 2)], [])   # age would reach 1 (expire) but re-stamp saves it
        ch = mem.get_channels()
        assert ch[2, 2, CH_PRESENCE] == 1.0


# ── get_channels output format ────────────────────────────────────────────────

class TestGetChannels:
    def test_shape(self):
        mem = EnemySpatialMemory(10, 15)
        ch = mem.get_channels()
        assert ch.shape == (10, 15, 3)

    def test_dtype_float32(self):
        mem = _make_mem()
        ch = mem.get_channels()
        assert ch.dtype == np.float32

    def test_presence_binary(self):
        mem = _make_mem()
        mem.step([_make_unit(0, 0), _make_unit(3, 3)], [])
        ch = mem.get_channels()
        unique_vals = set(ch[:, :, CH_PRESENCE].flat)
        assert unique_vals <= {0.0, 1.0}

    def test_age_norm_in_range(self):
        mem = _make_mem(max_age=10)
        mem.step([_make_unit(0, 0)], [])
        for _ in range(5):
            mem.step([], [])
        ch = mem.get_channels()
        assert 0.0 <= ch[0, 0, CH_AGE_NORM] < 1.0

    def test_decay_in_range(self):
        mem = _make_mem(decay=0.5, max_age=20)
        mem.step([_make_unit(0, 0)], [])
        for _ in range(5):
            mem.step([], [])
        ch = mem.get_channels()
        assert 0.0 <= ch[0, 0, CH_DECAY] <= 1.0

    def test_no_side_effects_on_repeated_call(self):
        mem = _make_mem()
        mem.step([_make_unit(1, 1)], [])
        ch1 = mem.get_channels()
        ch2 = mem.get_channels()
        assert np.array_equal(ch1, ch2)


# ── Summary statistics ────────────────────────────────────────────────────────

class TestSummaryStats:
    def test_coverage_zero_initially(self):
        mem = _make_mem()
        assert mem.coverage() == pytest.approx(0.0)

    def test_coverage_nonzero_after_stamp(self):
        mem = _make_mem(h=4, w=4)
        mem.step([_make_unit(0, 0)], [])
        # 1 cell out of 16
        assert mem.coverage() == pytest.approx(1 / 16)

    def test_coverage_zero_after_reset(self):
        mem = _make_mem()
        mem.step([_make_unit(0, 0)], [])
        mem.reset()
        assert mem.coverage() == pytest.approx(0.0)

    def test_decay_mean_zero_initially(self):
        mem = _make_mem()
        assert mem.decay_mean() == pytest.approx(0.0)

    def test_decay_mean_1_immediately_after_stamp(self):
        mem = _make_mem()
        mem.step([_make_unit(0, 0)], [])
        assert mem.decay_mean() == pytest.approx(1.0)

    def test_decay_mean_decreases_over_steps(self):
        mem = _make_mem(decay=0.9, max_age=20)
        mem.step([_make_unit(0, 0)], [])
        mean0 = mem.decay_mean()
        mem.step([], [])
        mean1 = mem.decay_mean()
        assert mean1 < mean0

    def test_decay_mean_zero_after_reset(self):
        mem = _make_mem()
        mem.step([_make_unit(0, 0)], [])
        mem.reset()
        assert mem.decay_mean() == pytest.approx(0.0)

    def test_active_cell_count_zero_initially(self):
        mem = _make_mem()
        assert mem.active_cell_count() == 0

    def test_active_cell_count_after_stamps(self):
        mem = _make_mem()
        mem.step([_make_unit(0, 0), _make_unit(1, 1)], [_make_unit(2, 2)])
        assert mem.active_cell_count() == 3

    def test_active_cell_count_decreases_at_expiry(self):
        mem = _make_mem(max_age=2)
        mem.step([_make_unit(0, 0)], [])
        mem.step([], [])   # age=1
        mem.step([], [])   # age=2 → expired
        assert mem.active_cell_count() == 0

    def test_active_cell_count_zero_after_reset(self):
        mem = _make_mem()
        mem.step([_make_unit(0, 0)], [])
        mem.reset()
        assert mem.active_cell_count() == 0


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_sightings_no_change(self):
        mem = _make_mem()
        mem.step([], [])
        ch = mem.get_channels()
        assert (ch == 0.0).all()

    def test_1x1_map(self):
        cfg = EnemySpatialMemoryConfig(decay_factor=0.5, max_age=5)
        mem = EnemySpatialMemory(1, 1, cfg)
        mem.step([_make_unit(0, 0)], [])
        ch = mem.get_channels()
        assert ch.shape == (1, 1, 3)
        assert ch[0, 0, CH_PRESENCE] == 1.0

    def test_unit_and_building_same_cell_combined(self):
        """Unit and building at same cell: both calls stamp the same cell."""
        mem = _make_mem()
        mem.step([_make_unit(3, 3)], [_make_unit(3, 3)])
        ch = mem.get_channels()
        assert ch[3, 3, CH_PRESENCE] == 1.0

    def test_large_map_no_error(self):
        mem = EnemySpatialMemory(128, 128)
        units = [_make_unit(i, i) for i in range(64)]
        mem.step(units, [])
        assert mem.active_cell_count() == 64

    def test_decay_factor_1_never_decays(self):
        cfg = EnemySpatialMemoryConfig(decay_factor=1.0, max_age=100)
        mem = EnemySpatialMemory(4, 4, cfg)
        mem.step([_make_unit(0, 0)], [])
        for _ in range(10):
            mem.step([], [])
        ch = mem.get_channels()
        assert ch[0, 0, CH_DECAY] == pytest.approx(1.0)

    def test_max_age_1_expires_next_step(self):
        cfg = EnemySpatialMemoryConfig(decay_factor=0.9, max_age=1)
        mem = EnemySpatialMemory(4, 4, cfg)
        mem.step([_make_unit(0, 0)], [])   # age=0
        mem.step([], [])                    # age=1 → expired
        ch = mem.get_channels()
        assert ch[0, 0, CH_PRESENCE] == 0.0
