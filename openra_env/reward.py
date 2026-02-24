"""Reward computation for OpenRA-RL.

Two reward systems:

1. **Scalar reward** (OpenRARewardFunction) — Legacy 6-component shaped reward.
   Used when reward_vector.enabled=False (default).

2. **Reward vector** (RewardVectorComputer from openra-rl-util) — 7+1 dimensional
   skill-based signal. Enabled via reward_vector.enabled=True in config.
   Can be collapsed to scalar via configurable weights.
"""

from dataclasses import dataclass
from typing import Optional

from openra_rl_util.reward_vector import RewardVector, RewardVectorComputer


@dataclass
class RewardWeights:
    """Configurable weights for each reward component."""

    survival: float = 0.001          # Per-tick survival bonus
    economic_efficiency: float = 0.01  # Reward for cash/power changes
    aggression: float = 0.1          # Reward for killing enemy units
    defense: float = 0.05            # Penalty for losing units
    victory: float = 1.0             # Terminal reward for winning
    defeat: float = -1.0             # Terminal penalty for losing


@dataclass
class RewardState:
    """Tracks previous observation values for delta computation."""

    prev_cash: int = 0
    prev_army_value: int = 0
    prev_units_killed: int = 0
    prev_units_lost: int = 0
    prev_buildings_killed: int = 0
    prev_buildings_lost: int = 0


class OpenRARewardFunction:
    """Computes shaped rewards from OpenRA game observations.

    Supports two modes:
    - Scalar: weighted sum of 6 simple components (default)
    - Vector: 8-dimensional reward via RewardVectorComputer (when enabled)

    The vector mode provides richer training signal for RL, decomposing
    reward into combat, economy, infrastructure, intelligence, composition,
    tempo, disruption, and outcome dimensions.
    """

    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        vector_enabled: bool = False,
        vector_weights: Optional[dict[str, float]] = None,
    ):
        self.weights = weights or RewardWeights()
        self._state = RewardState()

        # Reward vector mode
        self.vector_enabled = vector_enabled
        self._vector_computer = RewardVectorComputer() if vector_enabled else None
        self._vector_weights = vector_weights

    def reset(self) -> None:
        """Reset tracking state for a new episode."""
        self._state = RewardState()
        if self._vector_computer is not None:
            self._vector_computer.reset()

    def compute(self, obs_dict: dict) -> float:
        """Compute scalar reward from an observation dictionary.

        Args:
            obs_dict: Observation data with economy, military, done, result fields.

        Returns:
            Scalar reward value.
        """
        reward = 0.0

        economy = obs_dict.get("economy", {})
        military = obs_dict.get("military", {})
        done = obs_dict.get("done", False)
        result = obs_dict.get("result", "")

        # Survival reward
        reward += self.weights.survival

        # Economic efficiency (delta cash)
        cash = economy.get("cash", 0)
        cash_delta = cash - self._state.prev_cash
        if cash_delta > 0:
            reward += self.weights.economic_efficiency * (cash_delta / 1000.0)

        # Aggression (enemy kills)
        units_killed = military.get("units_killed", 0)
        buildings_killed = military.get("buildings_killed", 0)
        kills_delta = (units_killed - self._state.prev_units_killed) + (
            buildings_killed - self._state.prev_buildings_killed
        )
        reward += self.weights.aggression * kills_delta

        # Defense (own losses)
        units_lost = military.get("units_lost", 0)
        buildings_lost = military.get("buildings_lost", 0)
        losses_delta = (units_lost - self._state.prev_units_lost) + (
            buildings_lost - self._state.prev_buildings_lost
        )
        reward -= self.weights.defense * losses_delta

        # Terminal rewards
        if done:
            if result == "win":
                reward += self.weights.victory
            elif result == "lose":
                reward += self.weights.defeat

        # Update tracking state
        self._state.prev_cash = cash
        self._state.prev_units_killed = units_killed
        self._state.prev_units_lost = units_lost
        self._state.prev_buildings_killed = buildings_killed
        self._state.prev_buildings_lost = buildings_lost
        self._state.prev_army_value = military.get("army_value", 0)

        return reward

    def compute_vector(self, obs_dict: dict) -> Optional[RewardVector]:
        """Compute multi-dimensional reward vector.

        Returns None if vector mode is not enabled.

        Args:
            obs_dict: Full observation dictionary.

        Returns:
            RewardVector with 8 dimensions, or None if disabled.
        """
        if self._vector_computer is None:
            return None
        return self._vector_computer.compute(obs_dict)

    def compute_all(self, obs_dict: dict) -> tuple[float, Optional[dict[str, float]]]:
        """Compute both scalar reward and optional reward vector dict.

        Convenience method for the environment step() to get both signals.

        Returns:
            (scalar_reward, reward_vector_dict_or_None)
        """
        scalar = self.compute(obs_dict)
        vector = self.compute_vector(obs_dict)
        if vector is not None:
            return scalar, vector.as_dict()
        return scalar, None
