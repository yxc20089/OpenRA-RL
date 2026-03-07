"""Directive adherence metrics for evaluating agent compliance with strategic directives.

Tracks proxy metrics to measure how well an agent follows high-level strategic
directives from human commanders. Uses observable game state to infer adherence.
"""

import re
from typing import Optional

from openra_env.directives import Directive, DirectiveType, DirectivesManager


class DirectiveMetrics:
    """Tracks and evaluates agent adherence to strategic directives.

    Collects observable game state over time and computes adherence scores
    for each directive based on proxy metrics.
    """

    def __init__(self, directives_manager: DirectivesManager):
        """Initialize metrics tracker.

        Parameters
        ----------
        directives_manager : DirectivesManager
            The directives manager containing active directives.
        """
        self.directives_manager = directives_manager
        self.tick_count = 0
        self.observations = []  # Store state observations for analysis

        # Per-tick measurements
        self.harvester_counts = []
        self.defending_unit_counts = []
        self.cash_values = []
        self.ore_values = []
        self.army_values = []
        self.enemy_engagement_ticks = []  # Ticks when enemies were engaged

    def record_state(self, obs: dict, tick: int) -> None:
        """Record game state for a single tick.

        Parameters
        ----------
        obs : dict
            Game state observation (from get_game_state or similar).
        tick : int
            Current game tick.
        """
        self.tick_count += 1

        # Extract economy
        economy = obs.get("economy", {})
        self.cash_values.append(economy.get("cash", 0))
        self.ore_values.append(economy.get("ore", 0))

        # Extract units
        units = obs.get("units_summary", [])

        # Count harvesters
        harvester_count = sum(1 for u in units if u.get("type") == "harv")
        self.harvester_counts.append(harvester_count)

        # Count defending units (units within ~10 cells of base)
        buildings = obs.get("buildings_summary", [])
        if buildings:
            # Approximate base center from buildings
            base_x = sum(b.get("cell_x", 0) for b in buildings) / len(buildings)
            base_y = sum(b.get("cell_y", 0) for b in buildings) / len(buildings)

            # Count combat units near base (within 15 cells)
            defending_count = 0
            for unit in units:
                ux, uy = unit.get("cell_x", 0), unit.get("cell_y", 0)
                distance_sq = (ux - base_x) ** 2 + (uy - base_y) ** 2
                # Combat units have attack range > 0
                if distance_sq < 15 * 15 and unit.get("attack_range", 0) > 0:
                    defending_count += 1
            self.defending_unit_counts.append(defending_count)
        else:
            self.defending_unit_counts.append(0)

        # Track army value
        military = obs.get("military", {})
        self.army_values.append(military.get("army_value", 0))

        # Check if enemies are being engaged (visible enemies with low health or kills this tick)
        enemies = obs.get("visible_enemies_summary", [])
        if enemies:
            # If any enemy is damaged (health < 100%), agent is engaging
            engaged = any(e.get("health_percentage", 100) < 100 for e in enemies)
            if engaged or military.get("kills_cost", 0) > 0:
                self.enemy_engagement_ticks.append(tick)

    def compute_adherence(self) -> dict:
        """Compute adherence scores for all directives.

        Returns
        -------
        dict
            Adherence report with per-directive scores and overall adherence.
            Format: {
                "directives": [
                    {"id": 1, "text": "...", "adherence": 0.85, "details": "..."},
                    ...
                ],
                "overall_adherence": 0.75,
                "observations_count": 1234,
            }
        """
        if self.tick_count == 0:
            return {
                "directives": [],
                "overall_adherence": 0.0,
                "observations_count": 0,
            }

        directive_results = []
        adherence_scores = []

        for directive in self.directives_manager.get_all_directives():
            score, details = self._compute_directive_adherence(directive)
            directive_results.append({
                "id": directive.id,
                "type": directive.type.value,
                "text": directive.text,
                "adherence": score,
                "details": details,
            })
            if score is not None:
                adherence_scores.append(score)

        overall = sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0.0

        return {
            "directives": directive_results,
            "overall_adherence": overall,
            "observations_count": self.tick_count,
        }

    def _compute_directive_adherence(self, directive: Directive) -> tuple[Optional[float], str]:
        """Compute adherence score for a single directive.

        Parameters
        ----------
        directive : Directive
            The directive to evaluate.

        Returns
        -------
        tuple[float or None, str]
            (adherence_score, details_text). Score is 0-1 or None if not measurable.
        """
        text_lower = directive.text.lower()

        # Harvester directives (e.g., "Maintain 2+ harvesters")
        if "harvester" in text_lower:
            target = self._extract_number(text_lower, context="harvester")
            if target and self.harvester_counts:
                compliant_ticks = sum(1 for count in self.harvester_counts if count >= target)
                adherence = compliant_ticks / len(self.harvester_counts)
                avg_harvesters = sum(self.harvester_counts) / len(self.harvester_counts)
                return adherence, f"Target: {target}+, Average: {avg_harvesters:.1f}, Compliance: {adherence:.0%}"
            return None, "Harvester directive detected but target unclear"

        # Defense directives (e.g., "Keep 3+ units defending base")
        if "defend" in text_lower or "defense" in text_lower or "guard" in text_lower:
            target = self._extract_number(text_lower, context="defend")
            if target and self.defending_unit_counts:
                compliant_ticks = sum(1 for count in self.defending_unit_counts if count >= target)
                adherence = compliant_ticks / len(self.defending_unit_counts)
                avg_defenders = sum(self.defending_unit_counts) / len(self.defending_unit_counts)
                return adherence, f"Target: {target}+, Average: {avg_defenders:.1f}, Compliance: {adherence:.0%}"
            return None, "Defense directive detected but target unclear"

        # Economy focus (e.g., "Economy boom", "Focus on resources")
        if any(kw in text_lower for kw in ["economy", "resource", "boom", "cash", "income"]):
            if self.cash_values and self.army_values:
                # Measure if cash accumulation rate is high relative to army spending
                mid_point = len(self.cash_values) // 2
                early_cash_avg = sum(self.cash_values[:mid_point]) / mid_point if mid_point > 0 else 0
                late_cash_avg = sum(self.cash_values[mid_point:]) / (len(self.cash_values) - mid_point) if len(self.cash_values) > mid_point else 0
                cash_growth = late_cash_avg - early_cash_avg

                # Good adherence if cash grew significantly
                if cash_growth > 1000:
                    adherence = min(1.0, cash_growth / 3000)
                else:
                    adherence = 0.5  # Neutral
                return adherence, f"Cash growth: ${cash_growth:.0f} (early avg: ${early_cash_avg:.0f}, late avg: ${late_cash_avg:.0f})"
            return None, "Economy directive detected but insufficient data"

        # Aggression/Rush directives (e.g., "Rush", "Attack aggressively", "Early pressure")
        if any(kw in text_lower for kw in ["rush", "aggressive", "attack", "pressure", "early combat"]):
            if self.tick_count > 0:
                engagement_rate = len(self.enemy_engagement_ticks) / self.tick_count
                # High adherence if agent engages enemies frequently
                adherence = min(1.0, engagement_rate * 5)  # Scale so 20%+ engagement = full adherence
                return adherence, f"Combat engagement rate: {engagement_rate:.1%} of ticks"
            return None, "Aggression directive detected but no data"

        # Turtle/Defensive directives (e.g., "Turtle", "Defend", "Build up defenses")
        if any(kw in text_lower for kw in ["turtle", "defensive", "build up", "fortify"]):
            if self.defending_unit_counts:
                # High adherence if maintaining strong defense
                avg_defenders = sum(self.defending_unit_counts) / len(self.defending_unit_counts)
                adherence = min(1.0, avg_defenders / 5)  # 5+ defenders = full adherence
                return adherence, f"Average defenders: {avg_defenders:.1f}"
            return None, "Defensive directive detected but no data"

        # Generic/unmeasurable directive
        return None, "Directive type not measurable with current metrics"

    def _extract_number(self, text: str, context: str = "") -> Optional[int]:
        """Extract a numeric target from directive text.

        Parameters
        ----------
        text : str
            Directive text (lowercased).
        context : str
            Context hint for extraction (e.g., "harvester").

        Returns
        -------
        int or None
            Extracted number, or None if not found.
        """
        # Look for patterns like "2+", "at least 3", "maintain 2", etc.
        patterns = [
            r"(\d+)\s*\+",  # "2+"
            r"at\s+least\s+(\d+)",  # "at least 3"
            r"maintain\s+(\d+)",  # "maintain 2"
            r"keep\s+(\d+)",  # "keep 3"
            r"(\d+)\s+or\s+more",  # "2 or more"
            r"(\d+)\s+" + context,  # "2 harvesters"
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))

        # Fallback: just look for any number in the text
        match = re.search(r"\b(\d+)\b", text)
        if match:
            return int(match.group(1))

        return None

    def format_report(self) -> str:
        """Generate a human-readable adherence report.

        Returns
        -------
        str
            Formatted adherence report.
        """
        result = self.compute_adherence()

        lines = ["=== DIRECTIVE ADHERENCE REPORT ===", ""]
        lines.append(f"Observations: {result['observations_count']} ticks")
        lines.append(f"Overall Adherence: {result['overall_adherence']:.1%}")
        lines.append("")

        for d in result["directives"]:
            lines.append(f"Directive {d['id']}: {d['text']}")
            if d['adherence'] is not None:
                lines.append(f"  Adherence: {d['adherence']:.1%}")
            lines.append(f"  {d['details']}")
            lines.append("")

        return "\n".join(lines)
