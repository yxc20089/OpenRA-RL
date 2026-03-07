"""Tests for the strategic directives system."""

import tempfile
from pathlib import Path

import pytest
import yaml

from openra_env.config import DirectivesConfig, OpenRARLConfig, load_config
from openra_env.directives import Directive, DirectiveType, DirectivesManager
from openra_env.directive_metrics import DirectiveMetrics


class TestDirectivesConfig:
    """Test DirectivesConfig model."""

    def test_default_disabled(self):
        """Directives should be disabled by default."""
        cfg = DirectivesConfig()
        assert cfg.enabled is False
        assert cfg.pregame_strategy == ""
        assert cfg.standing_orders == []
        assert cfg.midgame_adjustments == []

    def test_enable_with_directives(self):
        """Can enable directives with pregame strategy and orders."""
        cfg = DirectivesConfig(
            enabled=True,
            pregame_strategy="Rush",
            standing_orders=["Maintain 2 harvesters", "Attack early"],
        )
        assert cfg.enabled is True
        assert cfg.pregame_strategy == "Rush"
        assert len(cfg.standing_orders) == 2

    def test_acknowledgment_required_default(self):
        """Acknowledgment should be required by default."""
        cfg = DirectivesConfig()
        assert cfg.acknowledgment_required is True


class TestDirectivesManager:
    """Test DirectivesManager functionality."""

    def test_empty_manager(self):
        """Manager with no directives."""
        cfg = DirectivesConfig(enabled=True)
        manager = DirectivesManager(cfg)
        assert len(manager.get_all_directives()) == 0
        assert manager.format_for_system_prompt() == ""

    def test_pregame_strategy_only(self):
        """Manager with only pregame strategy."""
        cfg = DirectivesConfig(
            enabled=True,
            pregame_strategy="Rush - attack early",
        )
        manager = DirectivesManager(cfg)
        directives = manager.get_all_directives()
        assert len(directives) == 1
        assert directives[0].type == DirectiveType.PREGAME_STRATEGY
        assert directives[0].text == "Rush - attack early"

    def test_standing_orders(self):
        """Manager with standing orders."""
        cfg = DirectivesConfig(
            enabled=True,
            standing_orders=["Maintain 2 harvesters", "Defend base"],
        )
        manager = DirectivesManager(cfg)
        directives = manager.get_all_directives()
        assert len(directives) == 2
        assert all(d.type == DirectiveType.STANDING_ORDER for d in directives)
        assert directives[0].text == "Maintain 2 harvesters"
        assert directives[1].text == "Defend base"

    def test_midgame_adjustments(self):
        """Manager with midgame adjustments."""
        cfg = DirectivesConfig(
            enabled=True,
            midgame_adjustments=["Flank from east", "Counter tanks"],
        )
        manager = DirectivesManager(cfg)
        directives = manager.get_all_directives()
        assert len(directives) == 2
        assert all(d.type == DirectiveType.MIDGAME_ADJUSTMENT for d in directives)

    def test_mixed_directives(self):
        """Manager with all directive types."""
        cfg = DirectivesConfig(
            enabled=True,
            pregame_strategy="Balanced",
            standing_orders=["Keep 2 harvesters"],
            midgame_adjustments=["Scout enemy"],
        )
        manager = DirectivesManager(cfg)
        directives = manager.get_all_directives()
        assert len(directives) == 3
        types = [d.type for d in directives]
        assert DirectiveType.PREGAME_STRATEGY in types
        assert DirectiveType.STANDING_ORDER in types
        assert DirectiveType.MIDGAME_ADJUSTMENT in types

    def test_directive_ids_unique(self):
        """Each directive should have a unique ID."""
        cfg = DirectivesConfig(
            enabled=True,
            standing_orders=["Order 1", "Order 2", "Order 3"],
        )
        manager = DirectivesManager(cfg)
        directives = manager.get_all_directives()
        ids = [d.id for d in directives]
        assert len(ids) == len(set(ids))  # All unique

    def test_acknowledge_directive(self):
        """Can acknowledge directives."""
        cfg = DirectivesConfig(
            enabled=True,
            standing_orders=["Order 1", "Order 2"],
        )
        manager = DirectivesManager(cfg)
        directives = manager.get_all_directives()

        # Initially not acknowledged
        assert not directives[0].acknowledged
        assert not directives[1].acknowledged

        # Acknowledge first directive
        success = manager.acknowledge_directive(directives[0].id)
        assert success is True
        assert directives[0].acknowledged is True
        assert directives[1].acknowledged is False

    def test_acknowledge_invalid_id(self):
        """Acknowledging invalid ID returns False."""
        cfg = DirectivesConfig(enabled=True, standing_orders=["Order 1"])
        manager = DirectivesManager(cfg)
        success = manager.acknowledge_directive(9999)
        assert success is False

    def test_get_unacknowledged_count(self):
        """Can count unacknowledged directives."""
        cfg = DirectivesConfig(
            enabled=True,
            standing_orders=["Order 1", "Order 2", "Order 3"],
        )
        manager = DirectivesManager(cfg)
        directives = manager.get_all_directives()

        assert manager.get_unacknowledged_count() == 3

        manager.acknowledge_directive(directives[0].id)
        assert manager.get_unacknowledged_count() == 2

        manager.acknowledge_directive(directives[1].id)
        assert manager.get_unacknowledged_count() == 1

    def test_format_for_system_prompt_with_pregame(self):
        """System prompt formatting includes pregame strategy."""
        cfg = DirectivesConfig(
            enabled=True,
            pregame_strategy="Rush early",
        )
        manager = DirectivesManager(cfg)
        prompt = manager.format_for_system_prompt()

        assert "STRATEGIC DIRECTIVES FROM COMMAND" in prompt
        assert "PREGAME STRATEGY:" in prompt
        assert "Rush early" in prompt

    def test_format_for_system_prompt_with_standing_orders(self):
        """System prompt formatting includes standing orders."""
        cfg = DirectivesConfig(
            enabled=True,
            standing_orders=["Keep 2 harvesters"],
        )
        manager = DirectivesManager(cfg)
        prompt = manager.format_for_system_prompt()

        assert "STANDING ORDERS:" in prompt
        assert "Keep 2 harvesters" in prompt
        assert "[NEW]" in prompt  # Unacknowledged marker

    def test_format_for_mcp_tool(self):
        """MCP tool formatting."""
        cfg = DirectivesConfig(
            enabled=True,
            pregame_strategy="Balanced",
            standing_orders=["Keep 2 harvesters"],
        )
        manager = DirectivesManager(cfg)
        output = manager.format_for_mcp_tool()

        assert "STRATEGIC DIRECTIVES FROM COMMAND" in output
        assert "Balanced" in output
        assert "Keep 2 harvesters" in output
        assert "awaiting acknowledgment" in output

    def test_get_status_summary(self):
        """Status summary returns correct structure."""
        cfg = DirectivesConfig(
            enabled=True,
            standing_orders=["Order 1", "Order 2"],
        )
        manager = DirectivesManager(cfg)
        directives = manager.get_all_directives()
        manager.acknowledge_directive(directives[0].id)

        status = manager.get_status_summary()
        assert status["total"] == 2
        assert status["acknowledged"] == 1
        assert status["unacknowledged"] == 1
        assert len(status["directives"]) == 2

    def test_load_from_yaml_file(self):
        """Can load directives from external YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = {
                "pregame_strategy": "From file",
                "standing_orders": ["File order 1", "File order 2"],
            }
            yaml.dump(yaml_content, f)
            yaml_path = f.name

        try:
            cfg = DirectivesConfig(
                enabled=True,
                directives_file=yaml_path,
            )
            manager = DirectivesManager(cfg)
            directives = manager.get_all_directives()

            assert len(directives) == 3  # 1 pregame + 2 standing
            assert directives[0].text == "From file"
            assert directives[1].text == "File order 1"
        finally:
            Path(yaml_path).unlink()

    def test_inline_config_overrides_file(self):
        """Inline config takes precedence over file config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = {
                "pregame_strategy": "From file",
            }
            yaml.dump(yaml_content, f)
            yaml_path = f.name

        try:
            cfg = DirectivesConfig(
                enabled=True,
                directives_file=yaml_path,
                pregame_strategy="From inline config",  # Should override file
            )
            manager = DirectivesManager(cfg)
            directives = manager.get_all_directives()

            assert directives[0].text == "From inline config"
        finally:
            Path(yaml_path).unlink()


class TestDirectiveMetrics:
    """Test directive adherence metrics."""

    def test_empty_metrics(self):
        """Metrics with no observations."""
        cfg = DirectivesConfig(enabled=True, standing_orders=["Order 1"])
        manager = DirectivesManager(cfg)
        metrics = DirectiveMetrics(manager)

        result = metrics.compute_adherence()
        assert result["observations_count"] == 0
        assert result["overall_adherence"] == 0.0

    def test_harvester_directive_adherence(self):
        """Measure adherence to harvester count directive."""
        cfg = DirectivesConfig(
            enabled=True,
            standing_orders=["Maintain 2+ harvesters"],
        )
        manager = DirectivesManager(cfg)
        metrics = DirectiveMetrics(manager)

        # Simulate 10 ticks with varying harvester counts
        for i in range(10):
            obs = {
                "economy": {"cash": 1000, "ore": 500},
                "units_summary": [
                    {"type": "harv"} for _ in range(2 if i < 7 else 1)
                ],  # 7/10 ticks compliant
                "buildings_summary": [],
                "visible_enemies_summary": [],
                "military": {"army_value": 0, "kills_cost": 0},
            }
            metrics.record_state(obs, i * 25)

        result = metrics.compute_adherence()
        directive_result = result["directives"][0]

        # Should be ~70% adherent (7 out of 10 ticks)
        assert 0.6 <= directive_result["adherence"] <= 0.8

    def test_defense_directive_adherence(self):
        """Measure adherence to defense directive."""
        cfg = DirectivesConfig(
            enabled=True,
            standing_orders=["Keep 3+ units defending base"],
        )
        manager = DirectivesManager(cfg)
        metrics = DirectiveMetrics(manager)

        # Simulate observations with units near buildings
        for i in range(10):
            obs = {
                "economy": {"cash": 1000},
                "units_summary": [
                    {"cell_x": 50, "cell_y": 50, "attack_range": 5}
                    for _ in range(3 if i < 6 else 2)
                ],
                "buildings_summary": [{"cell_x": 50, "cell_y": 50}],
                "visible_enemies_summary": [],
                "military": {"army_value": 0},
            }
            metrics.record_state(obs, i * 25)

        result = metrics.compute_adherence()
        directive_result = result["directives"][0]

        # Should be ~60% adherent (6 out of 10 ticks)
        assert 0.5 <= directive_result["adherence"] <= 0.7

    def test_economy_directive_adherence(self):
        """Measure adherence to economy focus directive."""
        cfg = DirectivesConfig(
            enabled=True,
            pregame_strategy="Economy boom - focus on resources",
        )
        manager = DirectivesManager(cfg)
        metrics = DirectiveMetrics(manager)

        # Simulate growing economy
        for i in range(20):
            obs = {
                "economy": {"cash": 1000 + i * 200, "ore": 500},  # Cash grows
                "units_summary": [],
                "buildings_summary": [],
                "visible_enemies_summary": [],
                "military": {"army_value": 100},
            }
            metrics.record_state(obs, i * 25)

        result = metrics.compute_adherence()
        directive_result = result["directives"][0]

        # Should have decent adherence due to cash growth
        assert directive_result["adherence"] is not None
        assert directive_result["adherence"] > 0.3

    def test_unmeasurable_directive(self):
        """Directives that can't be measured return None."""
        cfg = DirectivesConfig(
            enabled=True,
            standing_orders=["Use strategic brilliance"],  # Can't measure this
        )
        manager = DirectivesManager(cfg)
        metrics = DirectiveMetrics(manager)

        obs = {
            "economy": {"cash": 1000},
            "units_summary": [],
            "buildings_summary": [],
            "visible_enemies_summary": [],
            "military": {},
        }
        metrics.record_state(obs, 0)

        result = metrics.compute_adherence()
        directive_result = result["directives"][0]

        assert directive_result["adherence"] is None
        assert "not measurable" in directive_result["details"]


class TestConfigIntegration:
    """Test directives integration with config system."""

    def test_directives_disabled_by_default(self):
        """Directives should be disabled in default config."""
        cfg = OpenRARLConfig()
        assert cfg.directives.enabled is False

    def test_sync_directive_tools_when_disabled(self):
        """Directive tools should be disabled when directives are disabled."""
        cfg = OpenRARLConfig(
            directives={"enabled": False},
            tools={"categories": {"directives": True}},
        )
        # Model validator should disable directive tools
        assert cfg.tools.categories.directives is False

    def test_directive_tools_enabled_when_directives_enabled(self):
        """Directive tools should remain enabled when directives are enabled."""
        cfg = OpenRARLConfig(
            directives={"enabled": True},
            tools={"categories": {"directives": True}},
        )
        assert cfg.tools.categories.directives is True

    def test_load_directives_from_yaml_config(self):
        """Can load directives configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = {
                "directives": {
                    "enabled": True,
                    "pregame_strategy": "Rush strategy",
                    "standing_orders": ["Order 1", "Order 2"],
                }
            }
            yaml.dump(yaml_content, f)
            config_path = f.name

        try:
            cfg = load_config(config_path=config_path)
            assert cfg.directives.enabled is True
            assert cfg.directives.pregame_strategy == "Rush strategy"
            assert len(cfg.directives.standing_orders) == 2
        finally:
            Path(config_path).unlink()

    def test_env_var_override_for_directives(self):
        """Environment variables can override directive settings."""
        import os

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = {"directives": {"enabled": False}}
            yaml.dump(yaml_content, f)
            config_path = f.name

        try:
            os.environ["DIRECTIVES_ENABLED"] = "true"
            cfg = load_config(config_path=config_path)
            assert cfg.directives.enabled is True
        finally:
            Path(config_path).unlink()
            del os.environ["DIRECTIVES_ENABLED"]
