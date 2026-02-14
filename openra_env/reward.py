"""Reward computation for OpenRA-RL.

Configurable multi-component reward function that combines
survival, economic, military, and strategic signals.
"""

from dataclasses import dataclass, field
from typing import Optional


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

    The reward is a weighted sum of:
    - Survival: small positive reward per tick alive
    - Economic efficiency: reward for increasing cash/resources
    - Aggression: reward for destroying enemy units/buildings
    - Defense: penalty for losing own units/buildings
    - Victory/Defeat: large terminal reward
    """

    def __init__(self, weights: Optional[RewardWeights] = None):
        self.weights = weights or RewardWeights()
        self._state = RewardState()

    def reset(self) -> None:
        """Reset tracking state for a new episode."""
        self._state = RewardState()

    def compute(self, obs_dict: dict) -> float:
        """Compute reward from an observation dictionary.

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
