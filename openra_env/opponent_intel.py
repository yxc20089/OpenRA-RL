"""Hardcoded opponent intelligence profiles for OpenRA AI bots.

Provides scouting reports and behavioral profiles based on the AI difficulty
level. These are static assessments based on observed AI behavior patterns.
"""

from typing import Optional


# ── Opponent Profiles ──────────────────────────────────────────────────────

AI_PROFILES: dict[str, dict] = {
    "easy": {
        "difficulty": "Easy",
        "display_name": "Easy AI",
        "aggressiveness": "low",
        "expansion_tendency": "none",
        "unit_diversity": "low",
        "build_order_quality": "poor",
        "estimated_win_rate_vs_new_player": 0.30,
        "typical_first_attack_tick": 3000,
        "behavioral_traits": [
            "Defensive posture — rarely attacks unprovoked",
            "Builds basic units only (infantry, light vehicles)",
            "Stays at starting base, does not expand",
            "Slow economy — builds one refinery, few harvesters",
            "Weak at defending against multi-pronged attacks",
            "Does not rebuild destroyed buildings quickly",
        ],
        "recommended_counters": [
            "Rush with early infantry to overwhelm before defenses go up",
            "Any tank force will crush their army",
            "No need to scout urgently — they won't attack early",
        ],
        "typical_army_composition": {
            "infantry": 0.7,
            "vehicles": 0.2,
            "aircraft": 0.0,
            "ships": 0.1,
        },
        "recent_match_history": [
            {"result": "loss", "duration_ticks": 4500, "score": 1200},
            {"result": "loss", "duration_ticks": 6000, "score": 1800},
            {"result": "win", "duration_ticks": 12000, "score": 3500},
        ],
    },
    "normal": {
        "difficulty": "Normal",
        "display_name": "Normal AI",
        "aggressiveness": "high",
        "expansion_tendency": "high",
        "unit_diversity": "high",
        "build_order_quality": "good",
        "estimated_win_rate_vs_new_player": 0.65,
        "typical_first_attack_tick": 1500,
        "behavioral_traits": [
            "Very aggressive — sends attack waves frequently starting around tick 1500",
            "Masters all different unit types (infantry, tanks, aircraft, ships)",
            "Eager to open a second base near your position or mid-way on the map",
            "Strong economy — builds 2-3 refineries with multiple harvesters",
            "Rebuilds destroyed buildings quickly and adapts composition",
            "Will target your harvesters and exposed, undefended buildings",
            "Uses combined arms effectively (infantry + vehicles + air strikes)",
            "Scouts your base early and adjusts strategy based on what you build",
        ],
        "recommended_counters": [
            "Build early defenses (turrets) at base entrance — first attack comes ~tick 1500",
            "Scout early (by tick 500) to find and deny expansion attempts",
            "Send a small raiding force to destroy their second base before it's established",
            "Maintain power surplus at all times — their attacks exploit brownouts",
            "Build anti-air (SAM/AA Gun) by mid-game to counter their aircraft",
            "Match their economy: build 2+ refineries minimum to keep up",
            "Don't turtle — they will out-expand and out-resource you",
        ],
        "typical_army_composition": {
            "infantry": 0.30,
            "vehicles": 0.45,
            "aircraft": 0.15,
            "ships": 0.10,
        },
        "recent_match_history": [
            {"result": "win", "duration_ticks": 8000, "score": 5200},
            {"result": "win", "duration_ticks": 6500, "score": 4800},
            {"result": "loss", "duration_ticks": 10000, "score": 6100},
            {"result": "win", "duration_ticks": 7200, "score": 5500},
            {"result": "loss", "duration_ticks": 9000, "score": 4000},
        ],
    },
    "hard": {
        "difficulty": "Hard",
        "display_name": "Hard AI",
        "aggressiveness": "very_high",
        "expansion_tendency": "very_high",
        "unit_diversity": "very_high",
        "build_order_quality": "optimal",
        "estimated_win_rate_vs_new_player": 0.85,
        "typical_first_attack_tick": 1000,
        "behavioral_traits": [
            "Extremely aggressive — attacks within first 1000 ticks with combined forces",
            "Optimal build orders — wastes no time or resources, perfect macro",
            "Expands aggressively with multiple bases across the map",
            "Uses superweapons if tech allows (nuclear missile, iron curtain)",
            "Coordinates multi-front attacks simultaneously from different angles",
            "Excellent at resource denial — prioritizes harvesters and refineries",
            "Rapid tech progression to advanced units (Mammoth tanks, MiGs)",
            "Will cheat slightly on resource gathering speed",
        ],
        "recommended_counters": [
            "MUST build defenses immediately — turrets before second refinery",
            "Scout by tick 300 — their expansion is very fast",
            "Deny expansions aggressively or you'll be completely out-resourced",
            "Build multiple production buildings for faster unit output",
            "Never let power go negative — they will exploit it ruthlessly",
            "Mix anti-air into every attack group — they will use aircraft",
            "Prepare for superweapons by mid-game — keep army spread out",
        ],
        "typical_army_composition": {
            "infantry": 0.20,
            "vehicles": 0.45,
            "aircraft": 0.25,
            "ships": 0.10,
        },
        "recent_match_history": [
            {"result": "win", "duration_ticks": 5000, "score": 7200},
            {"result": "win", "duration_ticks": 4500, "score": 6800},
            {"result": "win", "duration_ticks": 6000, "score": 8100},
            {"result": "loss", "duration_ticks": 12000, "score": 9500},
            {"result": "win", "duration_ticks": 5500, "score": 7500},
        ],
    },
}


def get_opponent_profile(difficulty: str) -> Optional[dict]:
    """Get the opponent intelligence profile for a given AI difficulty.

    Args:
        difficulty: One of "easy", "normal", "hard". Also accepts
                   "bot_easy", "bot_normal", "bot_hard" (strips prefix).

    Returns:
        Profile dict or None if not found.
    """
    clean = difficulty.lower().replace("bot_", "")
    return AI_PROFILES.get(clean)


def get_opponent_summary(difficulty: str) -> str:
    """Get a human-readable scouting report for LLM consumption."""
    profile = get_opponent_profile(difficulty)
    if profile is None:
        return f"Unknown AI difficulty: {difficulty}"

    traits = "\n".join(f"  - {t}" for t in profile["behavioral_traits"])
    counters = "\n".join(f"  - {c}" for c in profile["recommended_counters"])

    wins = sum(1 for m in profile["recent_match_history"] if m["result"] == "win")
    total = len(profile["recent_match_history"])
    avg_score = sum(m["score"] for m in profile["recent_match_history"]) // total

    army = profile["typical_army_composition"]
    army_str = ", ".join(f"{k}: {v:.0%}" for k, v in army.items() if v > 0)

    return (
        f"## Opponent Scouting Report: {profile['display_name']}\n"
        f"Aggressiveness: {profile['aggressiveness']}\n"
        f"Expansion tendency: {profile['expansion_tendency']}\n"
        f"Unit diversity: {profile['unit_diversity']}\n"
        f"Build order quality: {profile['build_order_quality']}\n"
        f"Estimated first attack: ~tick {profile['typical_first_attack_tick']}\n"
        f"Win rate vs new players: {profile['estimated_win_rate_vs_new_player']:.0%}\n"
        f"Recent record: {wins}W-{total - wins}L (avg score: {avg_score})\n"
        f"Typical army mix: {army_str}\n"
        f"\nBehavioral traits:\n{traits}\n"
        f"\nRecommended counters:\n{counters}"
    )
