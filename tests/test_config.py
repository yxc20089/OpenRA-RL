"""Tests for the unified configuration system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from openra_env.config import (
    TOOL_CATEGORIES,
    AlertsConfig,
    AgentConfig,
    GameConfig,
    LLMConfig,
    OpenRARLConfig,
    OpponentConfig,
    PlanningConfig,
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
        assert cfg.opponent.bot_type == "normal"
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
