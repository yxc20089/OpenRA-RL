"""Tests for the OpenRA-RL reward function."""

import pytest

from openra_env.reward import OpenRARewardFunction, RewardState, RewardWeights


def make_obs(
    cash=0,
    units_killed=0,
    units_lost=0,
    buildings_killed=0,
    buildings_lost=0,
    army_value=0,
    done=False,
    result="",
):
    return {
        "economy": {"cash": cash},
        "military": {
            "units_killed": units_killed,
            "units_lost": units_lost,
            "buildings_killed": buildings_killed,
            "buildings_lost": buildings_lost,
            "army_value": army_value,
        },
        "done": done,
        "result": result,
    }


class TestRewardWeights:
    def test_defaults(self):
        w = RewardWeights()
        assert w.survival == 0.001
        assert w.economic_efficiency == 0.01
        assert w.aggression == 0.1
        assert w.defense == 0.05
        assert w.victory == 1.0
        assert w.defeat == -1.0

    def test_custom_weights(self):
        w = RewardWeights(survival=0.0, victory=10.0)
        assert w.survival == 0.0
        assert w.victory == 10.0


class TestRewardState:
    def test_defaults(self):
        s = RewardState()
        assert s.prev_cash == 0
        assert s.prev_units_killed == 0
        assert s.prev_units_lost == 0


class TestOpenRARewardFunction:
    def test_survival_reward(self):
        rf = OpenRARewardFunction()
        obs = make_obs()
        reward = rf.compute(obs)
        assert reward == pytest.approx(0.001)

    def test_cash_increase_reward(self):
        rf = OpenRARewardFunction()
        # First tick establishes baseline
        rf.compute(make_obs(cash=1000))
        # Cash increased by 2000
        reward = rf.compute(make_obs(cash=3000))
        # survival (0.001) + economic_efficiency * (2000/1000) = 0.001 + 0.01 * 2.0 = 0.021
        assert reward == pytest.approx(0.021)

    def test_cash_decrease_no_reward(self):
        rf = OpenRARewardFunction()
        rf.compute(make_obs(cash=5000))
        # Cash decreased — no economic reward (only survival)
        reward = rf.compute(make_obs(cash=3000))
        assert reward == pytest.approx(0.001)

    def test_kill_reward(self):
        rf = OpenRARewardFunction()
        rf.compute(make_obs())
        # Killed 2 enemy units
        reward = rf.compute(make_obs(units_killed=2))
        # survival + aggression * 2 = 0.001 + 0.1 * 2 = 0.201
        assert reward == pytest.approx(0.201)

    def test_building_kill_reward(self):
        rf = OpenRARewardFunction()
        rf.compute(make_obs())
        # Killed 1 enemy building
        reward = rf.compute(make_obs(buildings_killed=1))
        # survival + aggression * 1 = 0.001 + 0.1 = 0.101
        assert reward == pytest.approx(0.101)

    def test_loss_penalty(self):
        rf = OpenRARewardFunction()
        rf.compute(make_obs())
        # Lost 3 units
        reward = rf.compute(make_obs(units_lost=3))
        # survival - defense * 3 = 0.001 - 0.05 * 3 = -0.149
        assert reward == pytest.approx(-0.149)

    def test_building_loss_penalty(self):
        rf = OpenRARewardFunction()
        rf.compute(make_obs())
        reward = rf.compute(make_obs(buildings_lost=2))
        # survival - defense * 2 = 0.001 - 0.05 * 2 = -0.099
        assert reward == pytest.approx(-0.099)

    def test_victory_reward(self):
        rf = OpenRARewardFunction()
        reward = rf.compute(make_obs(done=True, result="win"))
        # survival + victory = 0.001 + 1.0 = 1.001
        assert reward == pytest.approx(1.001)

    def test_defeat_penalty(self):
        rf = OpenRARewardFunction()
        reward = rf.compute(make_obs(done=True, result="lose"))
        # survival + defeat = 0.001 + (-1.0) = -0.999
        assert reward == pytest.approx(-0.999)

    def test_draw_no_terminal_reward(self):
        rf = OpenRARewardFunction()
        reward = rf.compute(make_obs(done=True, result="draw"))
        # Only survival, no terminal reward for draw
        assert reward == pytest.approx(0.001)

    def test_reset_clears_state(self):
        rf = OpenRARewardFunction()
        rf.compute(make_obs(cash=5000, units_killed=10))
        rf.reset()
        # After reset, deltas computed from zero baseline
        reward = rf.compute(make_obs(cash=1000, units_killed=1))
        # survival + econ*(1000/1000) + aggression*1 = 0.001 + 0.01 + 0.1 = 0.111
        assert reward == pytest.approx(0.111)

    def test_custom_weights(self):
        weights = RewardWeights(survival=0.0, aggression=1.0, defense=0.0, victory=100.0)
        rf = OpenRARewardFunction(weights=weights)
        rf.compute(make_obs())
        reward = rf.compute(make_obs(units_killed=5))
        # aggression * 5 = 5.0 (no survival, no defense)
        assert reward == pytest.approx(5.0)

    def test_combined_scenario(self):
        rf = OpenRARewardFunction()
        rf.compute(make_obs(cash=1000))
        # Next tick: gained cash, killed 1 enemy, lost 1 unit
        reward = rf.compute(make_obs(cash=2000, units_killed=1, units_lost=1))
        # survival + econ*(1000/1000) + aggression*1 - defense*1
        # = 0.001 + 0.01 + 0.1 - 0.05 = 0.061
        assert reward == pytest.approx(0.061)

    def test_delta_tracking_across_steps(self):
        rf = OpenRARewardFunction()
        rf.compute(make_obs(units_killed=0))
        rf.compute(make_obs(units_killed=2))  # killed 2
        # Now units_killed is still 2, so delta is 0
        reward = rf.compute(make_obs(units_killed=2))
        assert reward == pytest.approx(0.001)  # only survival

    def test_empty_observation(self):
        rf = OpenRARewardFunction()
        reward = rf.compute({})
        # Should handle missing keys gracefully, just survival
        assert reward == pytest.approx(0.001)
