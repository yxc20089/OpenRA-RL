"""Tests for the unified configuration system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from openra_env.config import (
    TOOL_CATEGORIES,
    AlertPromptsConfig,
    AlertsConfig,
    AgentConfig,
    CompressionConfig,
    GameConfig,
    LLMConfig,
    OpenRARLConfig,
    OpponentConfig,
    PlanningConfig,
    PromptsConfig,
    RewardConfig,
    RewardVectorConfig,
    ToolCategoriesConfig,
    ToolsConfig,
    _coerce_value,
    _deep_merge,
    _set_nested,
    load_config,
    should_register_tool,
)


# ── Default Loading ───────────────────────────────────────────────────


class TestDefaults:
    def test_default_config_has_sane_values(self):
        cfg = OpenRARLConfig()
        assert cfg.game.mod == "ra"
        assert cfg.game.grpc_port == 9999
        assert cfg.opponent.bot_type == "easy"
        assert cfg.planning.enabled is True
        assert cfg.reward.victory == 1.0
        assert cfg.llm.model == "qwen/qwen3-coder-next"
        assert cfg.agent.max_time_s == 1800

    def test_all_tool_categories_enabled_by_default(self):
        cfg = OpenRARLConfig()
        cats = cfg.tools.categories
        for field in ToolCategoriesConfig.model_fields:
            assert getattr(cats, field) is True, f"Category {field} should default to True"

    def test_all_alerts_enabled_by_default(self):
        cfg = OpenRARLConfig()
        for field in AlertsConfig.model_fields:
            if field == "max_alerts":
                continue  # max_alerts is an int, not a bool toggle
            assert getattr(cfg.alerts, field) is True, f"Alert {field} should default to True"

    def test_disabled_tools_list_empty_by_default(self):
        cfg = OpenRARLConfig()
        assert cfg.tools.disabled == []

    def test_load_config_no_file_returns_defaults(self):
        """load_config() with no file and no env vars should return defaults."""
        with _clean_env():
            cfg = load_config(config_path="__nonexistent__.yaml")
            assert cfg.game.mod == "ra"
            assert cfg.llm.base_url == "https://openrouter.ai/api/v1/chat/completions"


# ── YAML Loading ──────────────────────────────────────────────────────


class TestYAMLLoading:
    def test_load_from_yaml(self):
        data = {"game": {"mod": "cnc", "grpc_port": 5555}, "opponent": {"bot_type": "hard"}}
        with _temp_yaml(data) as path, _clean_env():
            cfg = load_config(config_path=path)
            assert cfg.game.mod == "cnc"
            assert cfg.game.grpc_port == 5555
            assert cfg.opponent.bot_type == "hard"
            # Unspecified fields keep defaults
            assert cfg.game.map_name == "singles.oramap"

    def test_partial_yaml_merges_with_defaults(self):
        data = {"reward": {"victory": 5.0}}
        with _temp_yaml(data) as path, _clean_env():
            cfg = load_config(config_path=path)
            assert cfg.reward.victory == 5.0
            assert cfg.reward.defeat == -1.0  # default preserved

    def test_empty_yaml_returns_defaults(self):
        with _temp_yaml({}) as path, _clean_env():
            cfg = load_config(config_path=path)
            assert cfg.game.mod == "ra"

    def test_llm_config_from_yaml(self):
        data = {
            "llm": {
                "base_url": "http://localhost:11434/v1/chat/completions",
                "model": "llama3.1:70b",
                "api_key": "",
                "extra_headers": {},
            }
        }
        with _temp_yaml(data) as path, _clean_env():
            cfg = load_config(config_path=path)
            assert cfg.llm.base_url == "http://localhost:11434/v1/chat/completions"
            assert cfg.llm.model == "llama3.1:70b"
            assert cfg.llm.api_key == ""
            assert cfg.llm.extra_headers == {}


# ── Environment Variable Precedence ──────────────────────────────────


class TestEnvVarPrecedence:
    def test_env_var_overrides_yaml(self):
        data = {"opponent": {"bot_type": "easy"}}
        with _temp_yaml(data) as path:
            with _clean_env(BOT_TYPE="hard"):
                cfg = load_config(config_path=path)
                assert cfg.opponent.bot_type == "hard"

    def test_env_var_overrides_default(self):
        with _clean_env(BOT_TYPE="hard"):
            cfg = load_config(config_path="__nonexistent__.yaml")
            assert cfg.opponent.bot_type == "hard"

    def test_openra_path_env(self):
        with _clean_env(OPENRA_PATH="/custom/openra"):
            cfg = load_config(config_path="__nonexistent__.yaml")
            assert cfg.game.openra_path == "/custom/openra"

    def test_planning_enabled_env(self):
        with _clean_env(PLANNING_ENABLED="false"):
            cfg = load_config(config_path="__nonexistent__.yaml")
            assert cfg.planning.enabled is False

    def test_record_replays_env(self):
        with _clean_env(RECORD_REPLAYS="yes"):
            cfg = load_config(config_path="__nonexistent__.yaml")
            assert cfg.game.record_replays is True

    def test_openrouter_api_key_env(self):
        with _clean_env(OPENROUTER_API_KEY="sk-test-123"):
            cfg = load_config(config_path="__nonexistent__.yaml")
            assert cfg.llm.api_key == "sk-test-123"

    def test_llm_api_key_overrides_openrouter(self):
        """LLM_API_KEY should take precedence over OPENROUTER_API_KEY."""
        with _clean_env(OPENROUTER_API_KEY="sk-old", LLM_API_KEY="sk-new"):
            cfg = load_config(config_path="__nonexistent__.yaml")
            assert cfg.llm.api_key == "sk-new"

    def test_llm_base_url_env(self):
        with _clean_env(LLM_BASE_URL="http://localhost:1234/v1/chat/completions"):
            cfg = load_config(config_path="__nonexistent__.yaml")
            assert cfg.llm.base_url == "http://localhost:1234/v1/chat/completions"

    def test_llm_model_env(self):
        with _clean_env(LLM_MODEL="my-local-model"):
            cfg = load_config(config_path="__nonexistent__.yaml")
            assert cfg.llm.model == "my-local-model"

    def test_max_time_env(self):
        with _clean_env(MAX_TIME="3600"):
            cfg = load_config(config_path="__nonexistent__.yaml")
            assert cfg.agent.max_time_s == 3600


# ── Constructor Override Precedence ───────────────────────────────────


class TestOverridePrecedence:
    def test_overrides_beat_yaml(self):
        data = {"game": {"mod": "cnc"}}
        with _temp_yaml(data) as path, _clean_env():
            cfg = load_config(config_path=path, game={"mod": "d2k"})
            assert cfg.game.mod == "d2k"

    def test_env_beats_overrides(self):
        with _clean_env(BOT_TYPE="hard"):
            cfg = load_config(config_path="__nonexistent__.yaml", opponent={"bot_type": "easy"})
            assert cfg.opponent.bot_type == "hard"

    def test_cli_overrides_beat_env(self):
        """Explicit CLI flags should beat environment variables."""
        with _clean_env(OPENROUTER_MODEL="env-model"):
            cfg = load_config(
                config_path="__nonexistent__.yaml",
                cli_overrides={"llm": {"model": "cli-model"}},
            )
            assert cfg.llm.model == "cli-model"

    def test_cli_overrides_beat_yaml_and_env(self):
        data = {"llm": {"model": "yaml-model"}}
        with _temp_yaml(data) as path, _clean_env(LLM_MODEL="env-model"):
            cfg = load_config(
                config_path=path,
                cli_overrides={"llm": {"model": "cli-model"}},
            )
            assert cfg.llm.model == "cli-model"


# ── Boolean Coercion ──────────────────────────────────────────────────


class TestCoercion:
    @pytest.mark.parametrize("val,expected", [
        ("true", True), ("True", True), ("TRUE", True),
        ("1", True), ("yes", True), ("Yes", True),
        ("false", False), ("False", False), ("FALSE", False),
        ("0", False), ("no", False), ("No", False),
    ])
    def test_bool_coercion(self, val, expected):
        assert _coerce_value(val) is expected

    def test_int_coercion(self):
        assert _coerce_value("42") == 42
        assert isinstance(_coerce_value("42"), int)

    def test_float_coercion(self):
        assert _coerce_value("3.14") == 3.14
        assert isinstance(_coerce_value("3.14"), float)

    def test_string_passthrough(self):
        assert _coerce_value("hello") == "hello"


# ── Deep Merge ────────────────────────────────────────────────────────


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 3, "c": 4})
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"game": {"mod": "ra", "port": 9999}}
        _deep_merge(base, {"game": {"mod": "cnc"}})
        assert base == {"game": {"mod": "cnc", "port": 9999}}

    def test_override_replaces_non_dict(self):
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": "flat"})
        assert base == {"a": "flat"}


class TestSetNested:
    def test_single_level(self):
        d: dict = {}
        _set_nested(d, "key", "val")
        assert d == {"key": "val"}

    def test_multi_level(self):
        d: dict = {}
        _set_nested(d, "game.mod", "cnc")
        assert d == {"game": {"mod": "cnc"}}

    def test_preserves_siblings(self):
        d = {"game": {"mod": "ra", "port": 9999}}
        _set_nested(d, "game.mod", "cnc")
        assert d == {"game": {"mod": "cnc", "port": 9999}}


# ── Tool Filtering ────────────────────────────────────────────────────


class TestToolFiltering:
    def test_all_tools_enabled_by_default(self):
        cfg = ToolsConfig()
        for tool_name in TOOL_CATEGORIES:
            assert should_register_tool(tool_name, cfg) is True

    def test_disable_category(self):
        cfg = ToolsConfig(categories=ToolCategoriesConfig(knowledge=False))
        assert should_register_tool("lookup_unit", cfg) is False
        assert should_register_tool("lookup_building", cfg) is False
        assert should_register_tool("lookup_tech_tree", cfg) is False
        assert should_register_tool("lookup_faction", cfg) is False
        # Other categories unaffected
        assert should_register_tool("advance", cfg) is True
        assert should_register_tool("move_units", cfg) is True

    def test_disable_individual_tool(self):
        cfg = ToolsConfig(disabled=["surrender", "sell_building"])
        assert should_register_tool("surrender", cfg) is False
        assert should_register_tool("sell_building", cfg) is False
        # Other utility tools still enabled
        assert should_register_tool("get_replay_path", cfg) is True

    def test_disabled_list_overrides_category_enable(self):
        cfg = ToolsConfig(
            categories=ToolCategoriesConfig(movement=True),
            disabled=["move_units"],
        )
        assert should_register_tool("move_units", cfg) is False
        assert should_register_tool("attack_move", cfg) is True

    def test_unknown_tool_defaults_to_enabled(self):
        cfg = ToolsConfig()
        assert should_register_tool("some_future_tool", cfg) is True

    def test_all_tools_have_categories(self):
        """Every tool in TOOL_CATEGORIES should map to a valid category field."""
        valid_categories = set(ToolCategoriesConfig.model_fields.keys())
        for tool_name, category in TOOL_CATEGORIES.items():
            assert category in valid_categories, f"Tool {tool_name} maps to unknown category {category}"

    def test_tool_count(self):
        """Verify we have all 48 tools mapped."""
        assert len(TOOL_CATEGORIES) == 48


# ── Planning Sync Validator ───────────────────────────────────────────


class TestPlanningSync:
    def test_planning_disabled_auto_disables_planning_tools(self):
        cfg = OpenRARLConfig(planning=PlanningConfig(enabled=False))
        assert cfg.tools.categories.planning is False

    def test_planning_enabled_keeps_planning_tools(self):
        cfg = OpenRARLConfig(planning=PlanningConfig(enabled=True))
        assert cfg.tools.categories.planning is True

    def test_planning_disabled_via_yaml(self):
        data = {"planning": {"enabled": False}}
        with _temp_yaml(data) as path, _clean_env():
            cfg = load_config(config_path=path)
            assert cfg.planning.enabled is False
            assert cfg.tools.categories.planning is False


# ── LLM Config ────────────────────────────────────────────────────────


class TestLLMConfig:
    def test_local_model_no_key(self):
        cfg = LLMConfig(
            base_url="http://localhost:11434/v1/chat/completions",
            api_key="",
            model="llama3.1:70b",
        )
        assert cfg.api_key == ""
        assert "localhost" in cfg.base_url

    def test_remote_model_with_key(self):
        cfg = LLMConfig(api_key="sk-test-123")
        assert cfg.api_key == "sk-test-123"

    def test_extra_headers_default(self):
        cfg = LLMConfig()
        assert "HTTP-Referer" in cfg.extra_headers
        assert "X-Title" in cfg.extra_headers

    def test_extra_headers_empty_for_local(self):
        cfg = LLMConfig(extra_headers={})
        assert cfg.extra_headers == {}

    def test_temperature_default_none(self):
        cfg = LLMConfig()
        assert cfg.temperature is None

    def test_temperature_set(self):
        cfg = LLMConfig(temperature=0.7)
        assert cfg.temperature == 0.7


# ── Alert Config ──────────────────────────────────────────────────────


class TestAlertConfig:
    def test_disable_specific_alerts(self):
        cfg = AlertsConfig(under_attack=False, low_power=False)
        assert cfg.under_attack is False
        assert cfg.low_power is False
        assert cfg.damaged_building is True  # others unchanged


# ── Backwards Compatibility ───────────────────────────────────────────


class TestBackwardsCompat:
    def test_load_config_with_no_args(self):
        """Calling load_config() with no args should not raise."""
        with _clean_env():
            cfg = load_config(config_path="__nonexistent__.yaml")
            assert isinstance(cfg, OpenRARLConfig)

    def test_reward_config_matches_reward_weights(self):
        """RewardConfig fields should match the existing RewardWeights dataclass."""
        from openra_env.reward import RewardWeights

        rw = RewardWeights()
        rc = RewardConfig()
        assert rc.survival == rw.survival
        assert rc.economic_efficiency == rw.economic_efficiency
        assert rc.aggression == rw.aggression
        assert rc.defense == rw.defense
        assert rc.victory == rw.victory
        assert rc.defeat == rw.defeat


class TestRewardVectorConfig:
    """Test reward vector configuration."""

    def test_disabled_by_default(self):
        cfg = RewardVectorConfig()
        assert cfg.enabled is False

    def test_default_weights(self):
        cfg = RewardVectorConfig()
        assert cfg.weights["combat"] == 0.30
        assert cfg.weights["economy"] == 0.15
        assert cfg.weights["outcome"] == 1.00
        assert len(cfg.weights) == 8

    def test_present_in_root_config(self):
        cfg = OpenRARLConfig()
        assert hasattr(cfg, "reward_vector")
        assert isinstance(cfg.reward_vector, RewardVectorConfig)
        assert cfg.reward_vector.enabled is False

    def test_enable_via_yaml(self):
        with _clean_env():
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump({"reward_vector": {"enabled": True}}, f)
                f.flush()
                cfg = load_config(config_path=f.name)
                assert cfg.reward_vector.enabled is True

    def test_custom_weights_via_yaml(self):
        with _clean_env():
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump({"reward_vector": {"enabled": True, "weights": {"combat": 0.5}}}, f)
                f.flush()
                cfg = load_config(config_path=f.name)
                assert cfg.reward_vector.weights["combat"] == 0.5


# ── Validation Errors ─────────────────────────────────────────────────


class TestValidation:
    def test_invalid_grpc_port_type(self):
        with pytest.raises(Exception):  # Pydantic ValidationError
            GameConfig(grpc_port="not_a_number")

    def test_invalid_reward_weight(self):
        with pytest.raises(Exception):
            RewardConfig(victory="not_a_float")


# ── Helpers ───────────────────────────────────────────────────────────

_CONFIG_ENV_VARS = [
    "OPENRA_PATH", "RECORD_REPLAYS", "BOT_TYPE", "AI_SLOT",
    "PLANNING_ENABLED", "PLANNING_MAX_TURNS", "PLANNING_MAX_TIME",
    "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
    "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
    "OPENRA_URL", "MAX_TIME", "LLM_AGENT_LOG",
]


class _clean_env:
    """Context manager that temporarily clears config-related env vars and sets new ones."""

    def __init__(self, **overrides):
        self._overrides = overrides
        self._saved: dict[str, str | None] = {}

    def __enter__(self):
        # Save and clear all config env vars
        for var in _CONFIG_ENV_VARS:
            self._saved[var] = os.environ.pop(var, None)
        # Set overrides
        for key, val in self._overrides.items():
            os.environ[key] = str(val)
        return self

    def __exit__(self, *args):
        # Remove overrides
        for key in self._overrides:
            os.environ.pop(key, None)
        # Restore saved values
        for var, val in self._saved.items():
            if val is not None:
                os.environ[var] = val


def _temp_yaml(data: dict):
    """Context manager that writes *data* to a temp YAML file and yields its path."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            yield path
        finally:
            Path(path).unlink(missing_ok=True)

    return _ctx()


# ── PromptsConfig Tests ──────────────────────────────────────────────


class TestPromptsConfig:
    """Tests for the PromptsConfig system."""

    def test_default_prompts_have_values(self):
        """PromptsConfig defaults should have non-empty values for key fields."""
        p = PromptsConfig()
        assert "end_planning_phase" in p.planning_nudge
        assert "tool" in p.no_tool_nudge.lower()
        assert "{building}" in p.power_warning
        assert "{count}" in p.alerts.idle_army

    def test_prompts_in_root_config(self):
        """OpenRARLConfig should have prompts field with defaults."""
        config = OpenRARLConfig()
        assert isinstance(config.prompts, PromptsConfig)
        assert isinstance(config.prompts.alerts, AlertPromptsConfig)
        assert config.prompts.planning_complete == "Planning complete. Game is now live."

    def test_prompts_from_yaml(self):
        """Override prompts via config YAML."""
        data = {
            "prompts": {
                "no_tool_nudge": "Please call a tool now.",
                "alerts": {
                    "low_power": "Power is low: {balance}",
                },
            },
        }
        with _temp_yaml(data) as path:
            config = load_config(config_path=path)
        assert config.prompts.no_tool_nudge == "Please call a tool now."
        assert config.prompts.alerts.low_power == "Power is low: {balance}"
        # Other fields keep defaults
        assert "combat units idle" in config.prompts.alerts.idle_army

    def test_alert_template_format(self):
        """Alert templates should render with .format()."""
        p = AlertPromptsConfig()
        result = p.low_power.format(balance="-30")
        assert "LOW POWER" in result
        assert "-30" in result

    def test_placement_template_format(self):
        """Placement templates should render with .format()."""
        p = PromptsConfig()
        result = p.placement_failed.format(building="powr", reason="no valid position")
        assert "powr" in result
        assert "no valid position" in result

    def test_planning_prompt_template(self):
        """Planning prompt template should accept all expected variables."""
        p = PromptsConfig()
        result = p.planning_prompt.format(
            max_turns=10, map_name="test", map_width=64, map_height=64,
            base_x=10, base_y=10, enemy_x=50, enemy_y=50,
            faction="russia", side="Soviet",
            opponent_summary="Easy AI", planning_nudge=p.planning_nudge,
        )
        assert "10 turns" in result
        assert "russia" in result
        assert "end_planning_phase" in result

    def test_backward_compat_system_prompt_migration(self):
        """agent.system_prompt should migrate to prompts.system_prompt."""
        config = OpenRARLConfig(agent=AgentConfig(system_prompt="My custom prompt"))
        assert config.prompts.system_prompt == "My custom prompt"

    def test_prompts_system_prompt_takes_precedence(self):
        """prompts.system_prompt should win over agent.system_prompt."""
        config = OpenRARLConfig(
            agent=AgentConfig(system_prompt="agent version"),
            prompts=PromptsConfig(system_prompt="prompts version"),
        )
        assert config.prompts.system_prompt == "prompts version"

    def test_backward_compat_system_prompt_file_migration(self):
        """agent.system_prompt_file should migrate to prompts.system_prompt_file."""
        config = OpenRARLConfig(agent=AgentConfig(system_prompt_file="/tmp/test.txt"))
        assert config.prompts.system_prompt_file == "/tmp/test.txt"

    def test_env_var_prompts_file(self):
        """PROMPTS_FILE env var should set prompts.prompts_file."""
        with patch.dict(os.environ, {"PROMPTS_FILE": "/tmp/prompts.yaml"}, clear=False):
            config = load_config(config_path="/nonexistent/config.yaml")
        assert config.prompts.prompts_file == "/tmp/prompts.yaml"

    def test_game_start_template(self):
        """Game start template should render correctly."""
        p = PromptsConfig()
        result = p.game_start.format(
            strategy_section="\n\nRush strategy",
            briefing="Map: test",
            barracks_type="barr",
            mcv_note=" Your MCV is unit 42.",
        )
        assert "Game started!" in result
        assert "Rush strategy" in result
        assert "barr" in result
        assert "unit 42" in result

    def test_insufficient_funds_template(self):
        """Insufficient funds template should render correctly."""
        p = PromptsConfig()
        result = p.insufficient_funds.format(available=500, item="3tnk", cost=950)
        assert "500" in result
        assert "3tnk" in result
        assert "950" in result

    def test_build_queued_template(self):
        """Build queued template should render correctly."""
        p = PromptsConfig()
        result = p.build_queued.format(building="powr", cost=300, ticks=180, seconds=7.2)
        assert "powr" in result
        assert "300" in result
        assert "180" in result
        assert "auto-places" in result

    def test_build_unit_queued_template(self):
        """Build unit queued template should render correctly."""
        p = PromptsConfig()
        result = p.build_unit_queued.format(
            count=3, unit="e1", cost=100, ticks_each=60,
            ticks_total=180, seconds_total=7.2)
        assert "3x" in result
        assert "e1" in result
        assert "60" in result
        assert "180" in result

    def test_build_already_pending_template(self):
        """Build already pending template should render correctly."""
        p = PromptsConfig()
        result = p.build_already_pending.format(building="powr")
        assert "powr" in result
        assert "already queued" in result

    def test_max_alerts_default(self):
        """AlertsConfig max_alerts should default to 0 (unlimited)."""
        cfg = AlertsConfig()
        assert cfg.max_alerts == 0


# ── Compression Config ───────────────────────────────────────────────


class TestCompressionConfig:
    def test_defaults(self):
        c = CompressionConfig()
        assert c.include_strategy is True
        assert c.include_military is True
        assert c.include_production is True

    def test_disable_strategy(self):
        c = CompressionConfig(include_strategy=False)
        assert c.include_strategy is False

    def test_llm_compression_strategy_default(self):
        llm = LLMConfig()
        assert llm.compression_strategy == "sliding_window"

    def test_llm_compression_trigger_default(self):
        llm = LLMConfig()
        assert llm.compression_trigger == 0

    def test_compression_strategy_none(self):
        llm = LLMConfig(compression_strategy="none")
        assert llm.compression_strategy == "none"

    def test_compression_trigger_custom(self):
        llm = LLMConfig(compression_trigger=60)
        assert llm.compression_trigger == 60

    def test_prompts_compression_field(self):
        p = PromptsConfig()
        assert isinstance(p.compression, CompressionConfig)
        assert p.compression.include_strategy is True

    def test_move_eta_template(self):
        p = PromptsConfig()
        result = p.move_eta.format(ticks=183, seconds=7.3)
        assert "183" in result
        assert "7.3" in result

    def test_full_config_compression_yaml(self):
        """Compression fields round-trip through YAML config loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "llm": {
                    "compression_strategy": "none",
                    "compression_trigger": 60,
                    "keep_last_messages": 20,
                },
                "prompts": {
                    "compression": {
                        "include_strategy": False,
                        "include_military": True,
                        "include_production": False,
                    }
                }
            }, f)
            f.flush()
            cfg = load_config(f.name)
        os.unlink(f.name)
        assert cfg.llm.compression_strategy == "none"
        assert cfg.llm.compression_trigger == 60
        assert cfg.llm.keep_last_messages == 20
        assert cfg.prompts.compression.include_strategy is False
        assert cfg.prompts.compression.include_production is False
        assert cfg.prompts.compression.include_military is True


# ── Opponent Config ──────────────────────────────────────────────────


class TestOpponentConfig:
    def test_default_spawns_enemy(self):
        """Default opponent config spawns an enemy in Multi0."""
        cfg = OpponentConfig()
        assert cfg.ai_slot == "Multi0"
        assert cfg.bot_type == "easy"

    def test_disable_enemy_via_empty_slot(self):
        cfg = OpponentConfig(ai_slot="")
        assert cfg.ai_slot == ""

    def test_custom_bot_type(self):
        cfg = OpponentConfig(bot_type="hard")
        assert cfg.bot_type == "hard"


# ── Bot Type Mapping ─────────────────────────────────────────────────


class TestBotTypeMapping:
    def test_beginner_maps_to_beginner(self):
        from openra_env.server.openra_process import BOT_TYPE_MAP
        assert BOT_TYPE_MAP["beginner"] == "beginner"

    def test_easy_maps_to_easy(self):
        from openra_env.server.openra_process import BOT_TYPE_MAP
        assert BOT_TYPE_MAP["easy"] == "easy"

    def test_medium_maps_to_medium(self):
        from openra_env.server.openra_process import BOT_TYPE_MAP
        assert BOT_TYPE_MAP["medium"] == "medium"

    def test_hard_maps_to_normal(self):
        from openra_env.server.openra_process import BOT_TYPE_MAP
        assert BOT_TYPE_MAP["hard"] == "normal"

    def test_brutal_maps_to_rush(self):
        from openra_env.server.openra_process import BOT_TYPE_MAP
        assert BOT_TYPE_MAP["brutal"] == "rush"

    def test_raw_names_pass_through(self):
        from openra_env.server.openra_process import BOT_TYPE_MAP
        for raw in ["rush", "normal", "turtle", "naval", "beginner", "easy", "medium"]:
            assert BOT_TYPE_MAP.get(raw, raw) == raw

    def test_build_command_maps_hard(self):
        from openra_env.server.openra_process import OpenRAConfig, OpenRAProcessManager
        openra_path = str(Path(__file__).parent.parent / "OpenRA")
        config = OpenRAConfig(openra_path=openra_path, bot_type="hard")
        manager = OpenRAProcessManager(config)
        cmd = manager._build_command()
        bots_arg = [a for a in cmd if "Launch.Bots" in a][0]
        assert "normal" in bots_arg
        assert "hard" not in bots_arg

    def test_build_command_maps_brutal(self):
        from openra_env.server.openra_process import OpenRAConfig, OpenRAProcessManager
        openra_path = str(Path(__file__).parent.parent / "OpenRA")
        config = OpenRAConfig(openra_path=openra_path, bot_type="brutal")
        manager = OpenRAProcessManager(config)
        cmd = manager._build_command()
        bots_arg = [a for a in cmd if "Launch.Bots" in a][0]
        assert "rush" in bots_arg
        assert "brutal" not in bots_arg

    def test_build_command_no_enemy_with_empty_slot(self):
        from openra_env.server.openra_process import OpenRAConfig, OpenRAProcessManager
        openra_path = str(Path(__file__).parent.parent / "OpenRA")
        config = OpenRAConfig(openra_path=openra_path, ai_slot="")
        manager = OpenRAProcessManager(config)
        cmd = manager._build_command()
        bots_arg = [a for a in cmd if "Launch.Bots" in a][0]
        assert bots_arg == "Launch.Bots=Multi1:rl-agent"

    def test_default_config_spawns_enemy(self):
        from openra_env.server.openra_process import OpenRAConfig, OpenRAProcessManager
        openra_path = str(Path(__file__).parent.parent / "OpenRA")
        config = OpenRAConfig(openra_path=openra_path)
        manager = OpenRAProcessManager(config)
        cmd = manager._build_command()
        bots_arg = [a for a in cmd if "Launch.Bots" in a][0]
        # Default should include enemy (Multi0:normal)
        assert "Multi0" in bots_arg
        assert "normal" in bots_arg
