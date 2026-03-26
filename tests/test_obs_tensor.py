"""Tests for openra_env.obs_tensor."""

import base64
import math

import numpy as np
import pytest

from openra_env.obs_tensor import (
    ACTION_MASK_SIZE,
    CH_MEM_AGE_NORM,
    CH_MEM_DECAY,
    CH_MEM_EVER_SEEN,
    CH_MEM_PRESENCE,
    GAME_SPATIAL_CHANNELS,
    GLOBAL_VEC_FEATURES,
    GLOBAL_VEC_SIZE,
    MEMORY_CHANNELS,
    OBS_SPATIAL_CHANNELS,
    CH_FOG,
    CH_RESOURCES,
    ObsTensorBuilder,
    ObsTensorConfig,
    _actors_to_cells,
)
from openra_env.models import ActionType


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_spatial_blob(map_h: int, map_w: int, n_channels: int = GAME_SPATIAL_CHANNELS) -> str:
    """Create a base64 spatial blob filled with zeros."""
    raw = bytes(map_h * map_w * n_channels * 4)
    return base64.b64encode(raw).decode("ascii")


def make_spatial_blob_value(
    map_h: int, map_w: int, n_channels: int, value: float
) -> str:
    """Create a base64 spatial blob filled with a constant float value."""
    arr = np.full((map_h, map_w, n_channels), value, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def make_obs(
    map_h: int = 8,
    map_w: int = 8,
    tick: int = 0,
    cash: int = 0,
    units: list[dict] | None = None,
    buildings: list[dict] | None = None,
    production: list[dict] | None = None,
    visible_enemies: list[dict] | None = None,
    visible_enemy_buildings: list[dict] | None = None,
    available_production: list[str] | None = None,
    spatial_map: str | None = None,
    spatial_channels: int = GAME_SPATIAL_CHANNELS,
    result: str = "",
    done: bool = False,
    power_provided: int = 100,
    power_drained: int = 50,
) -> dict:
    return {
        "tick": tick,
        "economy": {
            "cash": cash,
            "ore": 0,
            "power_provided": power_provided,
            "power_drained": power_drained,
            "resource_capacity": 2000,
            "harvester_count": 0,
        },
        "military": {
            "units_killed": 0,
            "units_lost": 0,
            "buildings_killed": 0,
            "buildings_lost": 0,
            "army_value": 0,
            "active_unit_count": 0,
            "kills_cost": 0,
            "deaths_cost": 0,
            "assets_value": 0,
            "experience": 0,
            "order_count": 0,
        },
        "units": units or [],
        "buildings": buildings or [],
        "production": production or [],
        "visible_enemies": visible_enemies or [],
        "visible_enemy_buildings": visible_enemy_buildings or [],
        "map_info": {"width": map_w, "height": map_h, "map_name": "test"},
        "available_production": available_production or [],
        "spatial_map": spatial_map or make_spatial_blob(map_h, map_w),
        "spatial_channels": spatial_channels,
        "result": result,
        "done": done,
    }


def make_enemy_unit(cell_x: int, cell_y: int, can_attack: bool = True) -> dict:
    return {
        "actor_id": 99,
        "type": "e1",
        "can_attack": can_attack,
        "hp_percent": 1.0,
        "pos_x": cell_x * 1024,
        "pos_y": cell_y * 1024,
        "cell_x": cell_x,
        "cell_y": cell_y,
        "is_idle": True,
        "current_activity": "",
        "owner": "enemy",
        "facing": 0,
        "experience_level": 0,
        "stance": 1,
        "speed": 56,
        "attack_range": 0,
        "passenger_count": -1,
        "is_building": False,
    }


def make_enemy_building(cell_x: int, cell_y: int, type_: str = "barr") -> dict:
    return {
        "actor_id": 100,
        "type": type_,
        "pos_x": cell_x * 1024,
        "pos_y": cell_y * 1024,
        "hp_percent": 1.0,
        "owner": "enemy",
        "is_producing": False,
        "production_progress": 0.0,
        "producing_item": "",
        "is_powered": True,
        "is_repairing": False,
        "sell_value": 0,
        "rally_x": -1,
        "rally_y": -1,
        "power_amount": 0,
        "can_produce": [],
        "cell_x": cell_x,
        "cell_y": cell_y,
    }


# ── _actors_to_cells ──────────────────────────────────────────────────────────


class TestActorsToCells:
    def test_basic_conversion(self):
        actors = [{"cell_x": 3, "cell_y": 5}]
        cells = _actors_to_cells(actors)
        assert cells == [(5, 3)]  # (row=cell_y, col=cell_x)

    def test_multiple_actors(self):
        actors = [{"cell_x": 0, "cell_y": 1}, {"cell_x": 7, "cell_y": 2}]
        cells = _actors_to_cells(actors)
        assert cells == [(1, 0), (2, 7)]

    def test_empty_list(self):
        assert _actors_to_cells([]) == []

    def test_missing_keys_default_to_minus_one(self):
        cells = _actors_to_cells([{}])
        assert cells == [(-1, -1)]

    def test_full_actor_dict(self):
        """Full actor dict (with pos_x/pos_y world coords) — cell coords used."""
        actor = make_enemy_unit(cell_x=4, cell_y=6)
        cells = _actors_to_cells([actor])
        assert cells == [(6, 4)]


# ── ObsTensorConfig ───────────────────────────────────────────────────────────


class TestObsTensorConfig:
    def test_defaults(self):
        cfg = ObsTensorConfig()
        assert cfg.memory_enabled is False      # default: memory disabled
        assert cfg.memory_max_age == 200
        assert cfg.memory_decay_lambda == pytest.approx(0.02)
        assert cfg.memory_store_threat is False
        assert cfg.cash_scale == pytest.approx(5000.0)

    def test_custom_memory_params(self):
        cfg = ObsTensorConfig(
            memory_enabled=True, memory_max_age=50,
            memory_decay_lambda=0.1, memory_store_threat=True,
        )
        assert cfg.memory_enabled is True
        assert cfg.memory_max_age == 50
        assert cfg.memory_decay_lambda == pytest.approx(0.1)
        assert cfg.memory_store_threat is True

    def test_from_observation_config(self):
        from openra_env.config import ObservationConfig
        obs_cfg = ObservationConfig(
            enemy_memory_enabled=True,
            enemy_memory_max_age=100,
            enemy_memory_decay_lambda=0.05,
            enemy_memory_store_threat=True,
        )
        cfg = ObsTensorConfig.from_observation_config(obs_cfg)
        assert cfg.memory_enabled is True
        assert cfg.memory_max_age == 100
        assert cfg.memory_decay_lambda == pytest.approx(0.05)
        assert cfg.memory_store_threat is True

    def test_from_observation_config_defaults_disabled(self):
        from openra_env.config import ObservationConfig
        cfg = ObsTensorConfig.from_observation_config(ObservationConfig())
        assert cfg.memory_enabled is False


# ── Channel constants ─────────────────────────────────────────────────────────


class TestChannelConstants:
    def test_memory_channels_is_4(self):
        assert MEMORY_CHANNELS == 4

    def test_obs_spatial_channels_is_13(self):
        assert OBS_SPATIAL_CHANNELS == GAME_SPATIAL_CHANNELS + MEMORY_CHANNELS
        assert OBS_SPATIAL_CHANNELS == 13

    def test_memory_channel_indices(self):
        assert CH_MEM_PRESENCE  == GAME_SPATIAL_CHANNELS + 0   # 9
        assert CH_MEM_AGE_NORM  == GAME_SPATIAL_CHANNELS + 1   # 10
        assert CH_MEM_DECAY     == GAME_SPATIAL_CHANNELS + 2   # 11
        assert CH_MEM_EVER_SEEN == GAME_SPATIAL_CHANNELS + 3   # 12

    def test_global_vec_features_length(self):
        assert len(GLOBAL_VEC_FEATURES) == GLOBAL_VEC_SIZE


# ── ObsTensorBuilder — shapes ─────────────────────────────────────────────────


class TestObsTensorBuilderShapes:
    def test_spatial_shape(self):
        builder = ObsTensorBuilder()
        tensors = builder.build(make_obs(map_h=16, map_w=24))
        assert tensors["spatial"].shape == (16, 24, OBS_SPATIAL_CHANNELS)
        assert tensors["spatial"].dtype == np.float32

    def test_global_vec_shape(self):
        tensors = ObsTensorBuilder().build(make_obs())
        assert tensors["global_vec"].shape == (GLOBAL_VEC_SIZE,)
        assert tensors["global_vec"].dtype == np.float32

    def test_global_vec_size_is_33(self):
        assert GLOBAL_VEC_SIZE == 33

    def test_action_mask_shape(self):
        tensors = ObsTensorBuilder().build(make_obs())
        assert tensors["action_mask"].shape == (ACTION_MASK_SIZE,)
        assert tensors["action_mask"].dtype == bool

    def test_action_mask_length_matches_enum(self):
        tensors = ObsTensorBuilder().build(make_obs())
        assert len(tensors["action_mask"]) == len(list(ActionType))


# ── ObsTensorBuilder — step_decay ordering ───────────────────────────────────


def enabled_cfg(**kw) -> ObsTensorConfig:
    """Return an ObsTensorConfig with memory_enabled=True plus optional overrides."""
    return ObsTensorConfig(memory_enabled=True, **kw)


# ── Memory disabled (default) ─────────────────────────────────────────────────


class TestMemoryDisabled:
    """Verify that the default (memory_enabled=False) reproduces a memory-free
    observation — shape is still (H,W,13) but channels 9-12 are all 0.0."""

    def test_memory_channels_always_zero(self):
        builder = ObsTensorBuilder()  # default: memory disabled
        obs = make_obs(map_h=8, map_w=8, visible_enemies=[make_enemy_unit(2, 2)])
        tensors = builder.build(obs)
        assert (tensors["spatial"][:, :, GAME_SPATIAL_CHANNELS:] == 0.0).all()

    def test_shape_unchanged_when_disabled(self):
        tensors = ObsTensorBuilder().build(make_obs(map_h=4, map_w=6))
        assert tensors["spatial"].shape == (4, 6, OBS_SPATIAL_CHANNELS)

    def test_multiple_steps_no_memory_accumulation(self):
        builder = ObsTensorBuilder()
        builder.build(make_obs(map_h=8, map_w=8, visible_enemies=[make_enemy_unit(0, 0)]))
        tensors = builder.build(make_obs(map_h=8, map_w=8))
        assert (tensors["spatial"][:, :, GAME_SPATIAL_CHANNELS:] == 0.0).all()

    def test_global_vec_memory_stats_zero(self):
        vec = ObsTensorBuilder().build(make_obs())["global_vec"]
        assert vec[list(GLOBAL_VEC_FEATURES).index("enemy_memory_coverage")]   == pytest.approx(0.0)
        assert vec[list(GLOBAL_VEC_FEATURES).index("enemy_memory_decay_mean")] == pytest.approx(0.0)

    def test_reset_with_disabled_memory_is_safe(self):
        builder = ObsTensorBuilder()
        builder.reset(map_h=8, map_w=8)
        tensors = builder.build(make_obs(map_h=8, map_w=8))
        assert tensors["spatial"].shape == (8, 8, OBS_SPATIAL_CHANNELS)


# ── Memory step ordering ──────────────────────────────────────────────────────


class TestMemoryStepOrdering:
    """Verify that step_decay() runs before observe so fresh stamps survive."""

    def test_fresh_stamp_not_cleared_by_same_step_expiry(self):
        """Even with max_age=1, a cell stamped in build() step N survives step N."""
        cfg = enabled_cfg(memory_max_age=1, memory_decay_lambda=0.1)
        builder = ObsTensorBuilder(config=cfg)
        builder.reset()

        obs0 = make_obs(map_h=8, map_w=8, visible_enemies=[make_enemy_unit(2, 2)])
        builder.build(obs0)  # stamp at step 0; age=0 after stamp

        # Step 1: step_decay → age becomes 1 → expired (max_age=1).
        # Then re-stamp → should survive.
        obs1 = make_obs(map_h=8, map_w=8, visible_enemies=[make_enemy_unit(2, 2)])
        tensors = builder.build(obs1)
        assert tensors["spatial"][2, 2, CH_MEM_PRESENCE] == pytest.approx(1.0)

    def test_unseen_cell_expires_after_max_age(self):
        cfg = enabled_cfg(memory_max_age=2, memory_decay_lambda=0.1)
        builder = ObsTensorBuilder(config=cfg)
        builder.reset()

        builder.build(make_obs(map_h=8, map_w=8, visible_enemies=[make_enemy_unit(3, 3)]))
        builder.build(make_obs(map_h=8, map_w=8))  # age=1
        builder.build(make_obs(map_h=8, map_w=8))  # age=2 → expired
        tensors = builder.build(make_obs(map_h=8, map_w=8))
        assert tensors["spatial"][3, 3, CH_MEM_PRESENCE] == pytest.approx(0.0)


# ── ObsTensorBuilder — spatial tensor ────────────────────────────────────────


class TestObsTensorBuilderSpatial:
    def test_empty_spatial_map_returns_zeros(self):
        obs = make_obs(map_h=4, map_w=4, spatial_map="")
        tensors = ObsTensorBuilder().build(obs)
        assert (tensors["spatial"] == 0.0).all()

    def test_game_channels_decoded(self):
        map_h, map_w = 4, 4
        blob = make_spatial_blob_value(map_h, map_w, GAME_SPATIAL_CHANNELS, value=0.5)
        obs = make_obs(map_h=map_h, map_w=map_w, spatial_map=blob)
        tensors = ObsTensorBuilder().build(obs)
        game_portion = tensors["spatial"][:, :, :GAME_SPATIAL_CHANNELS]
        assert game_portion.mean() == pytest.approx(0.5, abs=1e-4)

    def test_memory_channels_are_last_four(self):
        """Memory channels 9-12 occupy the last MEMORY_CHANNELS=4 slots."""
        map_h, map_w = 4, 4
        tensors = ObsTensorBuilder().build(make_obs(map_h=map_h, map_w=map_w))
        mem = tensors["spatial"][:, :, GAME_SPATIAL_CHANNELS:]
        assert mem.shape == (map_h, map_w, MEMORY_CHANNELS)

    def test_memory_channels_initially_zero_except_ever_seen(self):
        """Before any enemy: presence/age/decay are 0; ever_seen also 0."""
        builder = ObsTensorBuilder()
        builder.reset()
        tensors = builder.build(make_obs())
        assert (tensors["spatial"][:, :, GAME_SPATIAL_CHANNELS:] == 0.0).all()

    def test_presence_channel_set_on_enemy_sighting(self):
        """Presence channel (9) is 1.0 at (cell_y, cell_x) after sighting."""
        builder = ObsTensorBuilder(config=enabled_cfg())
        builder.reset()
        obs = make_obs(map_h=8, map_w=8, visible_enemies=[make_enemy_unit(3, 5)])
        tensors = builder.build(obs)
        assert tensors["spatial"][5, 3, CH_MEM_PRESENCE] == pytest.approx(1.0)

    def test_ever_seen_channel_set_and_persists_after_expiry(self):
        """ever_seen (12) is set on first sighting and never cleared."""
        cfg = enabled_cfg(memory_max_age=1, memory_decay_lambda=0.5)
        builder = ObsTensorBuilder(config=cfg)
        builder.reset()

        # Stamp once
        builder.build(make_obs(map_h=8, map_w=8, visible_enemies=[make_enemy_unit(1, 1)]))

        # Let cell expire: 2 more steps without re-sighting
        builder.build(make_obs(map_h=8, map_w=8))  # age=1 → expires
        tensors = builder.build(make_obs(map_h=8, map_w=8))

        # Presence should be 0 (expired), ever_seen should be 1 (persists)
        assert tensors["spatial"][1, 1, CH_MEM_PRESENCE]  == pytest.approx(0.0)
        assert tensors["spatial"][1, 1, CH_MEM_EVER_SEEN] == pytest.approx(1.0)

    def test_decay_channel_starts_at_1_and_decreases(self):
        cfg = enabled_cfg(memory_max_age=100, memory_decay_lambda=0.2)
        builder = ObsTensorBuilder(config=cfg)
        builder.reset()

        builder.build(make_obs(map_h=8, map_w=8, visible_enemies=[make_enemy_unit(0, 0)]))
        tensors = builder.build(make_obs(map_h=8, map_w=8))  # one decay step

        expected = math.exp(-0.2)
        assert tensors["spatial"][0, 0, CH_MEM_DECAY] == pytest.approx(expected, rel=1e-4)

    def test_decay_compounds_over_steps(self):
        cfg = enabled_cfg(memory_max_age=100, memory_decay_lambda=0.15)
        builder = ObsTensorBuilder(config=cfg)
        builder.reset()

        builder.build(make_obs(map_h=8, map_w=8, visible_enemies=[make_enemy_unit(2, 2)]))
        for _ in range(5):
            builder.build(make_obs(map_h=8, map_w=8))

        expected = math.exp(-0.15 * 5)
        tensors = builder.build(make_obs(map_h=8, map_w=8))
        # After 6 decay steps total
        expected = math.exp(-0.15 * 6)
        assert tensors["spatial"][2, 2, CH_MEM_DECAY] == pytest.approx(expected, rel=1e-3)

    def test_enemy_building_stamped(self):
        """Buildings also write to the memory channels."""
        builder = ObsTensorBuilder(config=enabled_cfg())
        builder.reset()
        obs = make_obs(
            map_h=8, map_w=8,
            visible_enemy_buildings=[make_enemy_building(4, 6)],
        )
        tensors = builder.build(obs)
        assert tensors["spatial"][6, 4, CH_MEM_PRESENCE] == pytest.approx(1.0)

    def test_reset_clears_all_memory_channels(self):
        builder = ObsTensorBuilder()
        builder.build(make_obs(map_h=8, map_w=8, visible_enemies=[make_enemy_unit(0, 0)]))
        builder.reset()
        tensors = builder.build(make_obs(map_h=8, map_w=8))
        assert (tensors["spatial"][:, :, GAME_SPATIAL_CHANNELS:] == 0.0).all()

    def test_fewer_game_channels_padded(self):
        """When bridge sends fewer channels, the game portion is zero-padded."""
        map_h, map_w = 4, 4
        n_ch = 5
        blob = make_spatial_blob_value(map_h, map_w, n_ch, value=1.0)
        obs = make_obs(map_h=map_h, map_w=map_w, spatial_map=blob, spatial_channels=n_ch)
        tensors = ObsTensorBuilder().build(obs)
        assert np.allclose(tensors["spatial"][:, :, :n_ch], 1.0)
        assert np.allclose(tensors["spatial"][:, :, n_ch:GAME_SPATIAL_CHANNELS], 0.0)


# ── ObsTensorBuilder — global vector ─────────────────────────────────────────


class TestObsTensorBuilderGlobalVec:
    def test_cash_norm(self):
        obs = make_obs(cash=2500)
        vec = ObsTensorBuilder().build(obs)["global_vec"]
        idx = list(GLOBAL_VEC_FEATURES).index("cash_norm")
        assert vec[idx] == pytest.approx(0.5)

    def test_cash_norm_clipped_at_3(self):
        obs = make_obs(cash=100_000)
        vec = ObsTensorBuilder().build(obs)["global_vec"]
        idx = list(GLOBAL_VEC_FEATURES).index("cash_norm")
        assert vec[idx] == pytest.approx(3.0)

    def test_power_balance_norm_positive(self):
        obs = make_obs(power_provided=200, power_drained=100)
        vec = ObsTensorBuilder().build(obs)["global_vec"]
        idx = list(GLOBAL_VEC_FEATURES).index("power_balance_norm")
        assert vec[idx] == pytest.approx(1.0)

    def test_power_balance_norm_negative(self):
        obs = make_obs(power_provided=0, power_drained=100)
        vec = ObsTensorBuilder().build(obs)["global_vec"]
        idx = list(GLOBAL_VEC_FEATURES).index("power_balance_norm")
        assert vec[idx] == pytest.approx(-1.0)

    def test_is_low_power_flag(self):
        obs = make_obs(power_provided=50, power_drained=100)
        vec = ObsTensorBuilder().build(obs)["global_vec"]
        idx = list(GLOBAL_VEC_FEATURES).index("is_low_power")
        assert vec[idx] == pytest.approx(1.0)

    def test_is_low_power_false(self):
        obs = make_obs(power_provided=200, power_drained=100)
        vec = ObsTensorBuilder().build(obs)["global_vec"]
        idx = list(GLOBAL_VEC_FEATURES).index("is_low_power")
        assert vec[idx] == pytest.approx(0.0)

    def test_tick_norm(self):
        cfg = ObsTensorConfig(tick_scale=50000.0)
        obs = make_obs(tick=25000)
        vec = ObsTensorBuilder(config=cfg).build(obs)["global_vec"]
        idx = list(GLOBAL_VEC_FEATURES).index("tick_norm")
        assert vec[idx] == pytest.approx(0.5)

    def test_result_win(self):
        obs = make_obs(result="win")
        vec = ObsTensorBuilder().build(obs)["global_vec"]
        assert vec[list(GLOBAL_VEC_FEATURES).index("result_win")]  == pytest.approx(1.0)
        assert vec[list(GLOBAL_VEC_FEATURES).index("result_lose")] == pytest.approx(0.0)
        assert vec[list(GLOBAL_VEC_FEATURES).index("result_draw")] == pytest.approx(0.0)

    def test_result_in_progress_all_zero(self):
        obs = make_obs(result="")
        vec = ObsTensorBuilder().build(obs)["global_vec"]
        assert vec[list(GLOBAL_VEC_FEATURES).index("result_win")]  == pytest.approx(0.0)
        assert vec[list(GLOBAL_VEC_FEATURES).index("result_lose")] == pytest.approx(0.0)
        assert vec[list(GLOBAL_VEC_FEATURES).index("result_draw")] == pytest.approx(0.0)

    def test_enemy_memory_decay_mean_1_after_stamp(self):
        """mean_decay is 1.0 immediately after stamping (before any decay step)."""
        builder = ObsTensorBuilder(config=enabled_cfg())
        obs = make_obs(visible_enemies=[make_enemy_unit(0, 0)])
        vec = builder.build(obs)["global_vec"]
        idx = list(GLOBAL_VEC_FEATURES).index("enemy_memory_decay_mean")
        assert vec[idx] == pytest.approx(1.0)

    def test_enemy_memory_coverage_zero_initially(self):
        builder = ObsTensorBuilder(config=enabled_cfg())
        builder.reset()
        vec = builder.build(make_obs())["global_vec"]
        idx = list(GLOBAL_VEC_FEATURES).index("enemy_memory_coverage")
        assert vec[idx] == pytest.approx(0.0)

    def test_enemy_memory_coverage_nonzero_after_stamp(self):
        builder = ObsTensorBuilder(config=enabled_cfg())
        obs = make_obs(map_h=8, map_w=8, visible_enemies=[make_enemy_unit(0, 0)])
        vec = builder.build(obs)["global_vec"]
        idx = list(GLOBAL_VEC_FEATURES).index("enemy_memory_coverage")
        assert vec[idx] > 0.0

    def test_can_attack_reflects_action_mask(self):
        from tests.test_action_mask import make_unit
        obs = make_obs(units=[make_unit("e1", can_attack=True)])
        vec = ObsTensorBuilder().build(obs)["global_vec"]
        idx = list(GLOBAL_VEC_FEATURES).index("can_attack")
        assert vec[idx] == pytest.approx(1.0)

    def test_can_attack_false_when_no_combat_units(self):
        obs = make_obs()
        vec = ObsTensorBuilder().build(obs)["global_vec"]
        idx = list(GLOBAL_VEC_FEATURES).index("can_attack")
        assert vec[idx] == pytest.approx(0.0)

    def test_production_progress_mean_correct(self):
        production = [
            {"queue_type": "Infantry", "item": "e1", "progress": 0.4,
             "remaining_ticks": 100, "remaining_cost": 50, "paused": False},
            {"queue_type": "Infantry", "item": "e2", "progress": 0.6,
             "remaining_ticks": 80, "remaining_cost": 70, "paused": False},
        ]
        obs = make_obs(production=production)
        vec = ObsTensorBuilder().build(obs)["global_vec"]
        idx = list(GLOBAL_VEC_FEATURES).index("production_progress_mean")
        assert vec[idx] == pytest.approx(0.5)

    def test_no_nan_or_inf(self):
        obs = make_obs(cash=1000, tick=5000)
        vec = ObsTensorBuilder().build(obs)["global_vec"]
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))


# ── ObsTensorBuilder — action mask ───────────────────────────────────────────


class TestObsTensorBuilderActionMask:
    def test_empty_obs_only_always_valid_actions(self):
        """Empty obs: only no_op and surrender are True."""
        mask = ObsTensorBuilder().build(make_obs())["action_mask"]
        at_values = [at.value for at in ActionType]
        for i, at_val in enumerate(at_values):
            expected = at_val in ("no_op", "surrender")
            assert mask[i] == expected, (
                f"action_mask[{i}] ({at_val!r}) expected {expected}, got {mask[i]}"
            )

    def test_no_op_always_true(self):
        mask = ObsTensorBuilder().build(make_obs())["action_mask"]
        no_op_idx = [at.value for at in ActionType].index("no_op")
        assert bool(mask[no_op_idx]) is True


# ── ObsTensorBuilder — memory persistence ────────────────────────────────────


class TestObsTensorBuilderMemoryPersistence:
    def test_presence_persists_to_next_step(self):
        """Memory written at step N is visible at step N+1 (before expiry)."""
        builder = ObsTensorBuilder(config=enabled_cfg())
        builder.reset()
        builder.build(make_obs(map_h=8, map_w=8, visible_enemies=[make_enemy_unit(1, 1)]))
        tensors = builder.build(make_obs(map_h=8, map_w=8))
        assert tensors["spatial"][1, 1, CH_MEM_PRESENCE] == pytest.approx(1.0)

    def test_decay_value_after_two_build_calls(self):
        """After stamp + 1 decay step, decay = e^{-λ}."""
        lam = 0.3
        cfg = enabled_cfg(memory_max_age=100, memory_decay_lambda=lam)
        builder = ObsTensorBuilder(config=cfg)
        builder.reset()

        builder.build(make_obs(map_h=8, map_w=8, visible_enemies=[make_enemy_unit(2, 2)]))
        tensors = builder.build(make_obs(map_h=8, map_w=8))

        expected = math.exp(-lam)
        assert tensors["spatial"][2, 2, CH_MEM_DECAY] == pytest.approx(expected, rel=1e-4)

    def test_memory_clears_on_reset(self):
        builder = ObsTensorBuilder()
        builder.build(make_obs(map_h=8, map_w=8, visible_enemies=[make_enemy_unit(0, 0)]))
        builder.reset()
        tensors = builder.build(make_obs(map_h=8, map_w=8))
        assert (tensors["spatial"][:, :, GAME_SPATIAL_CHANNELS:] == 0.0).all()

    def test_lazy_init_without_reset(self):
        """Builder works without an explicit reset() call."""
        builder = ObsTensorBuilder()
        tensors = builder.build(make_obs(map_h=6, map_w=6))
        assert tensors["spatial"].shape == (6, 6, OBS_SPATIAL_CHANNELS)

    def test_map_resize_reinitialises_memory(self):
        """Changing map size between episodes creates fresh memory."""
        builder = ObsTensorBuilder()
        builder.build(make_obs(map_h=8, map_w=8, visible_enemies=[make_enemy_unit(0, 0)]))
        # New episode with different map size
        tensors = builder.build(make_obs(map_h=16, map_w=16))
        assert tensors["spatial"].shape == (16, 16, OBS_SPATIAL_CHANNELS)
        assert (tensors["spatial"][:, :, GAME_SPATIAL_CHANNELS:] == 0.0).all()


# ── ObsTensorBuilder — gymnasium space ────────────────────────────────────────


class TestObsTensorBuilderObsSpace:
    def test_returns_none_without_gymnasium(self):
        import sys
        import unittest.mock as mock

        builder = ObsTensorBuilder()
        with mock.patch.dict(sys.modules, {"gymnasium": None}):
            space = builder.observation_space(map_h=64, map_w=64)
        assert space is None

    def test_with_gymnasium(self):
        pytest.importorskip("gymnasium")
        import gymnasium as gym

        builder = ObsTensorBuilder()
        space = builder.observation_space(map_h=32, map_w=32)
        assert space is not None
        assert isinstance(space, gym.spaces.Dict)
        assert "spatial" in space.spaces
        assert "global_vec" in space.spaces
        assert "action_mask" in space.spaces

    def test_space_shapes(self):
        pytest.importorskip("gymnasium")
        builder = ObsTensorBuilder()
        space = builder.observation_space(map_h=32, map_w=48)
        assert space["spatial"].shape == (32, 48, OBS_SPATIAL_CHANNELS)
        assert space["global_vec"].shape == (GLOBAL_VEC_SIZE,)
        assert space["action_mask"].n == ACTION_MASK_SIZE
