import numpy as np
import pytest

from ra_rl.league import assign_opponents, compute_harmonic_score, pfsp_probabilities
from ra_rl.ppo import normalize_advantages


def test_harmonic_score_two_and_three_bots():
    assert compute_harmonic_score(0.7, 0.7) == 0.7
    assert compute_harmonic_score(0.8, 0.4) == pytest.approx(0.533, abs=1e-3)
    assert compute_harmonic_score(0.6, 0.6, 0.6) == 0.6
    assert compute_harmonic_score(1.0, 0.8, 0.0) == 0.0


def test_pfsp_prioritizes_weaker_opponents_with_floor():
    opponents = ["hard", "brutal", "turtle"]
    ema_wrs = {"hard": 0.75, "brutal": 0.35, "turtle": 0.90}
    probs = pfsp_probabilities(ema_wrs, opponents, power=1.5, min_prob=0.18)

    assert probs.shape == (3,)
    assert pytest.approx(float(probs.sum()), abs=1e-6) == 1.0
    assert (probs >= 0.18 - 1e-6).all()
    # brutal (lowest WR) gets highest share; turtle (highest WR) gets lowest share
    assert probs[1] > probs[0] > probs[2]

    assigned = assign_opponents(opponents, probs, games=60, rng=np.random.default_rng(42))
    assert len(assigned) == 60
    assert set(assigned) <= set(opponents)


def test_stratified_per_opponent_advantage_normalization():
    # Bot 0 ("turtle") has artificially high raw advantages (+10, +12);
    # Bot 1 ("brutal") has artificially negative raw advantages (-10, -8).
    packed = {
        "adv": np.array([10.0, 12.0, -10.0, -8.0], dtype=np.float32),
        "bot_idx": np.array([2, 2, 1, 1], dtype=np.int64),
    }
    normalize_advantages(packed, per_bot=True)
    adv = packed["adv"]
    # Within each bot, the better trajectory gets +1 and the worse gets -1!
    assert adv[0] < 0 < adv[1]
    assert adv[2] < 0 < adv[3]
    assert pytest.approx(float(adv[:2].mean()), abs=1e-5) == 0.0
    assert pytest.approx(float(adv[2:].mean()), abs=1e-5) == 0.0
