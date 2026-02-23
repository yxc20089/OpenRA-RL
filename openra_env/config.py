"""Unified configuration for OpenRA-RL.

Provides a single YAML-based configuration system with Pydantic validation.
Supports multiple override layers:
  env vars > constructor overrides > config file > built-in defaults

Usage:
    from openra_env.config import load_config
    config = load_config()                         # auto-find config.yaml
    config = load_config("path/to/config.yaml")    # explicit path
    config = load_config(game={"mod": "cnc"})      # with overrides
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# ── Pydantic Config Models ────────────────────────────────────────────


class GameConfig(BaseModel):
    openra_path: str = "/opt/openra"
    mod: str = "ra"
    map_name: str = "singles.oramap"
    grpc_port: int = 9999
    headless: bool = True
    record_replays: bool = False
    seed: Optional[int] = None
    max_ticks: int = 0  # 0 = unlimited
    max_wall_time_s: int = 0  # 0 = unlimited


class OpponentConfig(BaseModel):
    bot_type: str = "normal"
    ai_slot: str = "Multi0"


class PlanningConfig(BaseModel):
    enabled: bool = True
    max_turns: int = 10
    max_time_s: float = 60.0


class RewardConfig(BaseModel):
    survival: float = 0.001
    economic_efficiency: float = 0.01
    aggression: float = 0.1
    defense: float = 0.05
    victory: float = 1.0
    defeat: float = -1.0


class ToolCategoriesConfig(BaseModel):
    read: bool = True
    knowledge: bool = True
    bulk_knowledge: bool = True
    planning: bool = True
    game_control: bool = True
    movement: bool = True
    production: bool = True
    building_actions: bool = True
    placement: bool = True
    unit_groups: bool = True
    compound: bool = True
    utility: bool = True
    terrain: bool = True


class ToolsConfig(BaseModel):
    categories: ToolCategoriesConfig = Field(default_factory=ToolCategoriesConfig)
    disabled: list[str] = Field(default_factory=list)


class AlertsConfig(BaseModel):
    under_attack: bool = True
    damaged_building: bool = True
    low_power: bool = True
    idle_funds: bool = True
    ore_full: bool = True
    idle_production: bool = True
    production_stalled: bool = True
    building_ready: bool = True
    stance_warning: bool = True
    idle_army: bool = True
    no_defenses: bool = True
    no_scouting: bool = True
    loss_tracking: bool = True


class LLMConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key: str = ""
    model: str = "qwen/qwen3-coder-next"
    max_tokens: int = 1500
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    keep_last_messages: int = 40
    max_retries: int = 4
    retry_backoff_s: int = 10
    request_timeout_s: float = 120.0
    extra_headers: dict[str, str] = Field(
        default_factory=lambda: {
            "HTTP-Referer": "https://github.com/openra-rl",
            "X-Title": "OpenRA-RL Agent",
        }
    )


class AgentConfig(BaseModel):
    server_url: str = "http://localhost:8000"
    max_turns: int = 0  # 0 = unlimited
    max_time_s: int = 1800
    verbose: bool = False
    log_file: str = ""


class OpenRARLConfig(BaseModel):
    """Root configuration for the OpenRA-RL system."""

    game: GameConfig = Field(default_factory=GameConfig)
    opponent: OpponentConfig = Field(default_factory=OpponentConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    @model_validator(mode="after")
    def sync_planning_tools(self) -> "OpenRARLConfig":
        """Auto-disable planning tools when planning is disabled."""
        if not self.planning.enabled:
            self.tools.categories.planning = False
        return self


# ── Tool Category Mapping ─────────────────────────────────────────────

TOOL_CATEGORIES: dict[str, str] = {
    # Read
    "get_game_state": "read",
    "get_economy": "read",
    "get_units": "read",
    "get_buildings": "read",
    "get_enemies": "read",
    "get_production": "read",
    "get_map_info": "read",
    "get_exploration_status": "read",
    # Knowledge
    "lookup_unit": "knowledge",
    "lookup_building": "knowledge",
    "lookup_tech_tree": "knowledge",
    "lookup_faction": "knowledge",
    # Bulk Knowledge
    "get_faction_briefing": "bulk_knowledge",
    "get_map_analysis": "bulk_knowledge",
    "batch_lookup": "bulk_knowledge",
    # Planning
    "get_opponent_intel": "planning",
    "start_planning_phase": "planning",
    "end_planning_phase": "planning",
    "get_planning_status": "planning",
    # Game Control
    "advance": "game_control",
    # Movement
    "move_units": "movement",
    "attack_move": "movement",
    "attack_target": "movement",
    "stop_units": "movement",
    # Production
    "build_unit": "production",
    "build_structure": "production",
    "build_and_place": "production",
    # Building/Unit Actions
    "place_building": "building_actions",
    "cancel_production": "building_actions",
    "deploy_unit": "building_actions",
    "sell_building": "building_actions",
    "repair_building": "building_actions",
    "set_rally_point": "building_actions",
    "guard_target": "building_actions",
    "set_stance": "building_actions",
    "harvest": "building_actions",
    "power_down": "building_actions",
    "set_primary": "building_actions",
    # Placement
    "get_valid_placements": "placement",
    # Unit Groups
    "assign_group": "unit_groups",
    "add_to_group": "unit_groups",
    "get_groups": "unit_groups",
    "command_group": "unit_groups",
    # Compound
    "batch": "compound",
    "plan": "compound",
    # Utility
    "get_replay_path": "utility",
    "surrender": "utility",
    # Terrain
    "get_terrain_at": "terrain",
}


# ── Env Var Mapping ───────────────────────────────────────────────────

# Ordered so that more-specific vars (LLM_*) overwrite less-specific (OPENROUTER_*)
_ENV_VAR_MAP: list[tuple[str, str]] = [
    # game
    ("OPENRA_PATH", "game.openra_path"),
    ("RECORD_REPLAYS", "game.record_replays"),
    # opponent
    ("BOT_TYPE", "opponent.bot_type"),
    ("AI_SLOT", "opponent.ai_slot"),
    # planning
    ("PLANNING_ENABLED", "planning.enabled"),
    ("PLANNING_MAX_TURNS", "planning.max_turns"),
    ("PLANNING_MAX_TIME", "planning.max_time_s"),
    # llm — legacy OpenRouter names first, then generic LLM_ names (override)
    ("OPENROUTER_API_KEY", "llm.api_key"),
    ("OPENROUTER_MODEL", "llm.model"),
    ("LLM_BASE_URL", "llm.base_url"),
    ("LLM_API_KEY", "llm.api_key"),
    ("LLM_MODEL", "llm.model"),
    # agent
    ("OPENRA_URL", "agent.server_url"),
    ("MAX_TIME", "agent.max_time_s"),
    ("LLM_AGENT_LOG", "agent.log_file"),
]


# ── Helper Functions ──────────────────────────────────────────────────


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _set_nested(d: dict, path: str, value: object) -> None:
    """Set a value in a nested dict via dotted path (e.g. ``'game.mod'``)."""
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _coerce_value(value: str) -> object:
    """Coerce a string env-var value to bool / int / float / str."""
    lower = value.lower()
    if lower in ("true", "1", "yes"):
        return True
    if lower in ("false", "0", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def should_register_tool(tool_name: str, tools_config: ToolsConfig) -> bool:
    """Return True if *tool_name* should be registered based on config."""
    if tool_name in tools_config.disabled:
        return False
    category = TOOL_CATEGORIES.get(tool_name)
    if category is not None:
        return getattr(tools_config.categories, category, True)
    return True  # unknown tools default to enabled


# ── Config Loading ────────────────────────────────────────────────────


def load_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[dict] = None,
    **overrides: object,
) -> OpenRARLConfig:
    """Load configuration with precedence: CLI > env vars > overrides > file > defaults.

    Parameters
    ----------
    config_path:
        Explicit path to a YAML config file. When ``None``, searches for
        ``config.yaml`` in the current working directory and the project root.
    cli_overrides:
        Dict of overrides from explicit CLI flags. Applied last (highest
        priority), beating even environment variables. Use this for values
        the user typed on the command line.
    **overrides:
        Keyword arguments that are deep-merged on top of the file values.
        Keys should be top-level section names (e.g. ``game={...}``).
    """
    config_dict: dict = {}

    # 1. Load YAML file
    resolved_path = _resolve_config_path(config_path)
    if resolved_path is not None:
        with open(resolved_path, encoding="utf-8") as f:
            file_dict = yaml.safe_load(f) or {}
        _deep_merge(config_dict, file_dict)

    # 2. Apply programmatic overrides (e.g. constructor args)
    if overrides:
        _deep_merge(config_dict, overrides)

    # 3. Apply environment variable overrides
    for env_var, dotted_path in _ENV_VAR_MAP:
        value = os.environ.get(env_var)
        if value is not None:
            _set_nested(config_dict, dotted_path, _coerce_value(value))

    # 4. Apply CLI overrides (highest priority — explicit user intent)
    if cli_overrides:
        _deep_merge(config_dict, cli_overrides)

    # 5. Validate and return
    return OpenRARLConfig(**config_dict)


def _resolve_config_path(config_path: Optional[str]) -> Optional[str]:
    """Find the config file to load, or None if none exists."""
    if config_path is not None:
        p = Path(config_path)
        return str(p) if p.exists() else None

    # Auto-discover: CWD first, then project root
    candidates = [
        Path.cwd() / "config.yaml",
        Path(__file__).resolve().parent.parent / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None
