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
    # bot_type: difficulty tiers (beginner/easy/medium/hard/brutal)
    # or raw OpenRA play styles (rush/normal/turtle/naval)
    # ai_slot: player slot for AI; set to "" to disable enemy spawning
    bot_type: str = "easy"
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


class RewardVectorConfig(BaseModel):
    """Configuration for the multi-dimensional reward vector.

    When enabled, each step returns an 8-dimensional reward vector
    (combat, economy, infrastructure, intelligence, composition,
    tempo, disruption, outcome) alongside the scalar reward.
    """

    enabled: bool = True  # 8-dimensional skill signal (combat, economy, etc.)
    weights: dict[str, float] = Field(default_factory=lambda: {
        "combat": 0.30,
        "economy": 0.15,
        "infrastructure": 0.10,
        "intelligence": 0.10,
        "composition": 0.10,
        "tempo": 0.10,
        "disruption": 0.15,
        "outcome": 1.00,
    })


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
    minimap: bool = True  # Show ASCII minimap in turn briefing
    max_alerts: int = 0  # 0 = unlimited; set >0 to cap alerts per turn


class LLMConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key: str = ""
    model: str = "qwen/qwen3-coder-next"
    max_tokens: int = 1500
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    keep_last_messages: int = 40
    compression_strategy: str = "sliding_window"  # "sliding_window" or "none"
    compression_trigger: int = 0  # 0 = keep_last_messages * 2
    max_retries: int = 4
    retry_backoff_s: int = 10
    request_timeout_s: float = 120.0
    reasoning_effort: Optional[str] = None  # "none", "low", "medium", "high"
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
    bench_url: str = ""  # empty = no auto-upload. Set to HF Space URL to enable.
    system_prompt: str = ""  # deprecated — use prompts.system_prompt
    system_prompt_file: str = ""  # deprecated — use prompts.system_prompt_file


class AlertPromptsConfig(BaseModel):
    """Templates for in-game alert messages.

    All templates use Python str.format() placeholders (e.g. {balance}).
    """

    under_attack: str = "UNDER ATTACK: enemy {type} id={id} near base"
    under_attack_mass: str = "UNDER ATTACK: {count} enemies near base ({breakdown})"
    damaged: str = "DAMAGED: {type} id={id} at {hp} HP"
    low_power: str = "LOW POWER: {balance} — production runs at 1/3 speed"
    power_tight: str = "POWER TIGHT: {balance} surplus — next building may cause low power"
    idle_funds: str = "IDLE FUNDS: ${funds} available, {harvesters} harvester(s)"
    ore_full: str = "ORE FULL: {ore}/{cap} storage — income is being lost"
    idle_production: str = "IDLE PRODUCTION: no active production queue"
    stalled: str = "STALLED: {item}@{progress} — $0 funds, production paused"
    building_stuck: str = "BUILDING STUCK: {building} — auto-placement failing"
    ready_to_place: str = "READY TO PLACE: {building} — completed, awaiting placement"
    stance: str = "STANCE: {count} combat unit(s) on ReturnFire (only fire when fired upon)"
    idle_army: str = "IDLE ARMY: {count} combat units idle"
    no_defenses: str = "NO DEFENSES: no defense structures built"
    no_scouting: str = (
        "NO SCOUTING: enemy not found — {explored} of map explored, "
        "{idle} idle combat units available"
    )


class CompressionConfig(BaseModel):
    """Controls what context is preserved in history compression summaries."""
    include_strategy: bool = True  # Preserve planning strategy
    include_military: bool = True  # Include kill/death counts
    include_production: bool = True  # Track what was produced


class PromptsConfig(BaseModel):
    """All LLM-facing text, configurable for customization.

    Templates use Python str.format() placeholders. Override individual
    fields in config.yaml, or point prompts_file to a YAML with all prompts.
    """

    # ── System prompt ────────────────────────────────────────────────
    system_prompt: str = ""  # inline override (highest priority)
    system_prompt_file: str = ""  # path to .txt file override
    prompts_file: str = ""  # path to YAML with all prompts below

    # ── Planning phase ───────────────────────────────────────────────
    # Variables: {max_turns}, {map_name}, {map_width}, {map_height},
    #   {base_x}, {base_y}, {enemy_x}, {enemy_y}, {faction}, {side},
    #   {opponent_summary}, {planning_nudge}
    planning_prompt: str = (
        "## PRE-GAME PLANNING PHASE\n"
        "You have {max_turns} turns to plan.\n\n"
        "### Map Intel\n"
        "Map: {map_name} ({map_width}x{map_height})\n"
        "Your base: ({base_x}, {base_y})\n"
        "Enemy estimated: ({enemy_x}, {enemy_y})\n"
        "Your faction: {faction} ({side})\n\n"
        "### Opponent Intelligence\n{opponent_summary}\n\n"
        "{planning_nudge}"
    )
    planning_nudge: str = "Call end_planning_phase(strategy='...') when ready to start."
    planning_instructions: str = (
        "Planning phase active. Available tools: get_faction_briefing "
        "(all unit/building stats), get_map_analysis (terrain/resources), "
        "get_opponent_intel (enemy profile), batch_lookup (multi-item queries). "
        "Call end_planning_phase(strategy=...) to begin gameplay."
    )
    planning_complete: str = "Planning complete. Game is now live."

    # ── Game start ───────────────────────────────────────────────────
    # Variables: {strategy_section}, {briefing}, {barracks_type}, {mcv_note}
    game_start: str = (
        "Game started!{strategy_section}\n\n{briefing}\n\n"
        "Your barracks type is '{barracks_type}'.{mcv_note}"
    )

    # ── Agent nudges ─────────────────────────────────────────────────
    no_tool_nudge: str = "No tool was called. A tool call is required each turn."
    continue_nudge: str = "The game is still in progress."
    compression_suffix: str = "Game continues from current state."
    sanitize_bridge: str = "Acknowledged. Continuing."

    # ── Tool warnings ────────────────────────────────────────────────
    # Variables: {building}, {drain}, {balance}
    power_warning: str = (
        "POWER WARNING: {building} drains {drain} power. "
        "Balance will be {balance}."
    )
    # Variables: {available}, {item}, {cost}
    insufficient_funds: str = (
        "Insufficient funds: ${available} available, "
        "{item} costs ${cost}."
    )

    # ── Placement feedback ───────────────────────────────────────────
    placement_success: str = "AUTO-PLACED: {building}"
    placement_failed: str = "PLACEMENT FAILED: {building} — {reason}. Auto-cancelling."
    placement_water: str = "WATER BUILDING: {building} requires water tiles for placement."

    # ── Build confirmations ───────────────────────────────────────────
    # Variables: {building}, {cost}, {ticks}, {seconds}
    build_queued: str = (
        "'{building}' (${cost}) queued, auto-places on completion. "
        "~{ticks} ticks (~{seconds}s)."
    )
    build_structure_queued: str = (
        "'{building}' (${cost}) queued. ~{ticks} ticks (~{seconds}s) to complete."
    )
    # Variables: {count}, {unit}, {cost}, {ticks_each}, {ticks_total}, {seconds_total}
    build_unit_queued: str = (
        "{count}x '{unit}' (${cost} each) queued. "
        "~{ticks_each} ticks per unit, ~{ticks_total} ticks (~{seconds_total}s) total."
    )

    # ── Build guards ──────────────────────────────────────────────────
    # Variables: {building}
    build_already_pending: str = "'{building}' is already queued and pending auto-placement."
    place_auto_managed: str = (
        "'{building}' is queued via build_and_place — placement is automatic."
    )

    # ── Movement feedback ────────────────────────────────────────────
    # Variables: {ticks}, {seconds}
    move_eta: str = "Units moving. Slowest arrives in ~{ticks} ticks (~{seconds}s)."

    # ── Compression ──────────────────────────────────────────────────
    compression: CompressionConfig = Field(default_factory=CompressionConfig)

    # ── Alerts ───────────────────────────────────────────────────────
    alerts: AlertPromptsConfig = Field(default_factory=AlertPromptsConfig)


class OpenRARLConfig(BaseModel):
    """Root configuration for the OpenRA-RL system."""

    game: GameConfig = Field(default_factory=GameConfig)
    opponent: OpponentConfig = Field(default_factory=OpponentConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    reward_vector: RewardVectorConfig = Field(default_factory=RewardVectorConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)

    @model_validator(mode="after")
    def sync_planning_tools(self) -> "OpenRARLConfig":
        """Auto-disable planning tools when planning is disabled."""
        if not self.planning.enabled:
            self.tools.categories.planning = False
        return self

    @model_validator(mode="after")
    def migrate_system_prompt(self) -> "OpenRARLConfig":
        """Backward compat: copy agent.system_prompt* to prompts.* if prompts.* empty."""
        if not self.prompts.system_prompt and self.agent.system_prompt:
            self.prompts.system_prompt = self.agent.system_prompt
        if not self.prompts.system_prompt_file and self.agent.system_prompt_file:
            self.prompts.system_prompt_file = self.agent.system_prompt_file
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
    ("BENCH_URL", "agent.bench_url"),
    ("SYSTEM_PROMPT_FILE", "agent.system_prompt_file"),
    # prompts
    ("SYSTEM_PROMPT_FILE", "prompts.system_prompt_file"),  # also maps to prompts.*
    ("PROMPTS_FILE", "prompts.prompts_file"),
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
