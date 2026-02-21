"""Tests for the openra-rl CLI package."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ── Console ─────────────────────────────────────────────────────────

class TestConsole:
    def test_info(self, capsys):
        from openra_env.cli.console import info
        info("hello")
        assert "hello" in capsys.readouterr().out

    def test_success(self, capsys):
        from openra_env.cli.console import success
        success("done")
        assert "done" in capsys.readouterr().out

    def test_error(self, capsys):
        from openra_env.cli.console import error
        error("fail")
        assert "fail" in capsys.readouterr().err

    def test_warn(self, capsys):
        from openra_env.cli.console import warn
        warn("caution")
        assert "caution" in capsys.readouterr().out

    def test_step(self, capsys):
        from openra_env.cli.console import step
        step("pulling...")
        assert "pulling..." in capsys.readouterr().out

    def test_header(self, capsys):
        from openra_env.cli.console import header
        header("Title")
        assert "Title" in capsys.readouterr().out

    def test_dim(self, capsys):
        from openra_env.cli.console import dim
        dim("faint text")
        assert "faint text" in capsys.readouterr().out


# ── Docker Manager ──────────────────────────────────────────────────

class TestDockerManager:
    @patch("openra_env.cli.docker_manager.shutil.which", return_value=None)
    def test_check_docker_not_installed(self, mock_which):
        from openra_env.cli.docker_manager import check_docker
        assert check_docker() is False

    @patch("openra_env.cli.docker_manager.shutil.which", return_value="/usr/bin/docker")
    @patch("openra_env.cli.docker_manager._run")
    def test_check_docker_daemon_not_running(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=1)
        from openra_env.cli.docker_manager import check_docker
        assert check_docker() is False

    @patch("openra_env.cli.docker_manager.shutil.which", return_value="/usr/bin/docker")
    @patch("openra_env.cli.docker_manager._run")
    def test_check_docker_ok(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0)
        from openra_env.cli.docker_manager import check_docker
        assert check_docker() is True

    @patch("openra_env.cli.docker_manager._run")
    def test_is_running_false(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        from openra_env.cli.docker_manager import is_running
        assert is_running() is False

    @patch("openra_env.cli.docker_manager._run")
    def test_is_running_true(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="openra-rl-server\n")
        from openra_env.cli.docker_manager import is_running
        assert is_running() is True

    @patch("openra_env.cli.docker_manager._run")
    def test_image_exists_false(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        from openra_env.cli.docker_manager import image_exists
        assert image_exists() is False

    @patch("openra_env.cli.docker_manager._run")
    def test_image_exists_true(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n")
        from openra_env.cli.docker_manager import image_exists
        assert image_exists() is True

    @patch("openra_env.cli.docker_manager.is_running", return_value=True)
    def test_start_server_already_running(self, mock_running):
        from openra_env.cli.docker_manager import start_server
        assert start_server() is True

    @patch("openra_env.cli.docker_manager.is_running", return_value=False)
    def test_stop_server_not_running(self, mock_running):
        from openra_env.cli.docker_manager import stop_server
        assert stop_server() is True

    @patch("openra_env.cli.docker_manager.is_running", return_value=True)
    @patch("openra_env.cli.docker_manager._run")
    def test_stop_server_ok(self, mock_run, mock_running):
        mock_run.return_value = MagicMock(returncode=0)
        from openra_env.cli.docker_manager import stop_server
        assert stop_server() is True

    @patch("openra_env.cli.docker_manager._run")
    def test_server_status_not_running(self, mock_run):
        from openra_env.cli.docker_manager import server_status, is_running
        with patch("openra_env.cli.docker_manager.is_running", return_value=False):
            assert server_status() is None

    @patch("openra_env.cli.docker_manager.is_running", return_value=True)
    @patch("openra_env.cli.docker_manager._run")
    def test_server_status_running(self, mock_run, mock_running):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Up 5 minutes\t0.0.0.0:8000->8000/tcp"
        )
        from openra_env.cli.docker_manager import server_status
        status = server_status()
        assert status is not None
        assert "Up" in status["status"]

    def test_image_constant(self):
        from openra_env.cli.docker_manager import IMAGE
        assert "ghcr.io" in IMAGE

    def test_container_name(self):
        from openra_env.cli.docker_manager import CONTAINER_NAME
        assert CONTAINER_NAME == "openra-rl-server"


# ── Wizard ──────────────────────────────────────────────────────────

class TestWizard:
    def test_config_path(self):
        from openra_env.cli.wizard import CONFIG_DIR, CONFIG_PATH
        assert CONFIG_DIR == Path.home() / ".openra-rl"
        assert CONFIG_PATH == Path.home() / ".openra-rl" / "config.yaml"

    def test_providers_defined(self):
        from openra_env.cli.wizard import PROVIDERS
        assert "openrouter" in PROVIDERS
        assert "ollama" in PROVIDERS
        assert "lmstudio" in PROVIDERS

    def test_provider_openrouter_needs_key(self):
        from openra_env.cli.wizard import PROVIDERS
        assert PROVIDERS["openrouter"]["needs_key"] is True

    def test_provider_ollama_no_key(self):
        from openra_env.cli.wizard import PROVIDERS
        assert PROVIDERS["ollama"]["needs_key"] is False

    def test_provider_lmstudio_no_key(self):
        from openra_env.cli.wizard import PROVIDERS
        assert PROVIDERS["lmstudio"]["needs_key"] is False

    def test_has_saved_config_false(self, tmp_path):
        from openra_env.cli import wizard
        with patch.object(wizard, "CONFIG_PATH", tmp_path / "nonexistent.yaml"):
            assert wizard.has_saved_config() is False

    def test_save_and_load_config(self, tmp_path):
        from openra_env.cli import wizard
        cfg_path = tmp_path / "config.yaml"
        with patch.object(wizard, "CONFIG_PATH", cfg_path), \
             patch.object(wizard, "CONFIG_DIR", tmp_path):
            wizard.save_config({"llm": {"model": "test-model"}})
            loaded = wizard.load_saved_config()
            assert loaded["llm"]["model"] == "test-model"

    def test_merge_cli_into_config_provider(self):
        from openra_env.cli.wizard import merge_cli_into_config
        config = {"llm": {"model": "old"}}
        result = merge_cli_into_config(config, provider="ollama")
        assert "localhost:11434" in result["llm"]["base_url"]
        assert result["provider"] == "ollama"

    def test_merge_cli_into_config_model(self):
        from openra_env.cli.wizard import merge_cli_into_config
        config = {}
        result = merge_cli_into_config(config, model="new-model")
        assert result["llm"]["model"] == "new-model"

    def test_merge_cli_into_config_api_key(self):
        from openra_env.cli.wizard import merge_cli_into_config
        config = {}
        result = merge_cli_into_config(config, api_key="sk-test")
        assert result["llm"]["api_key"] == "sk-test"

    def test_merge_cli_preserves_existing(self):
        from openra_env.cli.wizard import merge_cli_into_config
        config = {"llm": {"model": "existing", "base_url": "http://test"}}
        result = merge_cli_into_config(config, api_key="sk-new")
        assert result["llm"]["model"] == "existing"
        assert result["llm"]["base_url"] == "http://test"
        assert result["llm"]["api_key"] == "sk-new"


# ── Commands ────────────────────────────────────────────────────────

class TestCommands:
    def test_cmd_version(self, capsys):
        from openra_env.cli.commands import cmd_version
        cmd_version()
        out = capsys.readouterr().out
        assert "openra-rl" in out

    @patch("openra_env.cli.commands.docker")
    def test_cmd_server_status_not_running(self, mock_docker, capsys):
        mock_docker.server_status.return_value = None
        from openra_env.cli.commands import cmd_server_status
        cmd_server_status()
        assert "not running" in capsys.readouterr().out

    @patch("openra_env.cli.commands.docker")
    def test_cmd_server_status_running(self, mock_docker, capsys):
        mock_docker.server_status.return_value = {
            "status": "Up 5 minutes",
            "ports": "0.0.0.0:8000->8000/tcp",
        }
        from openra_env.cli.commands import cmd_server_status
        cmd_server_status()
        assert "running" in capsys.readouterr().out

    @patch("openra_env.cli.commands.docker")
    def test_cmd_server_stop(self, mock_docker):
        from openra_env.cli.commands import cmd_server_stop
        cmd_server_stop()
        mock_docker.stop_server.assert_called_once()

    @patch("openra_env.cli.commands.docker")
    def test_cmd_server_logs(self, mock_docker):
        from openra_env.cli.commands import cmd_server_logs
        cmd_server_logs(follow=True)
        mock_docker.get_logs.assert_called_once_with(follow=True)

    @patch("openra_env.cli.commands.docker.check_docker", return_value=False)
    def test_cmd_play_no_docker(self, mock_check):
        from openra_env.cli.commands import cmd_play
        with pytest.raises(SystemExit):
            cmd_play()


# ── Main Entry Point ───────────────────────────────────────────────

class TestMain:
    def test_main_no_args_shows_help(self, capsys):
        from openra_env.cli.main import main
        with patch("sys.argv", ["openra-rl"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_version_flag(self, capsys):
        from openra_env.cli.main import main
        with patch("sys.argv", ["openra-rl", "--version"]):
            main()
        assert "openra-rl" in capsys.readouterr().out

    def test_main_version_subcommand(self, capsys):
        from openra_env.cli.main import main
        with patch("sys.argv", ["openra-rl", "version"]):
            main()
        assert "openra-rl" in capsys.readouterr().out

    @patch("openra_env.cli.commands.cmd_doctor")
    def test_main_doctor(self, mock_doctor):
        from openra_env.cli.main import main
        with patch("sys.argv", ["openra-rl", "doctor"]):
            main()
        mock_doctor.assert_called_once()

    @patch("openra_env.cli.commands.cmd_config")
    def test_main_config(self, mock_config):
        from openra_env.cli.main import main
        with patch("sys.argv", ["openra-rl", "config"]):
            main()
        mock_config.assert_called_once()

    @patch("openra_env.cli.commands.cmd_server_stop")
    def test_main_server_stop(self, mock_stop):
        from openra_env.cli.main import main
        with patch("sys.argv", ["openra-rl", "server", "stop"]):
            main()
        mock_stop.assert_called_once()

    @patch("openra_env.cli.commands.cmd_server_status")
    def test_main_server_status(self, mock_status):
        from openra_env.cli.main import main
        with patch("sys.argv", ["openra-rl", "server", "status"]):
            main()
        mock_status.assert_called_once()

    @patch("openra_env.cli.commands.cmd_server_logs")
    def test_main_server_logs_follow(self, mock_logs):
        from openra_env.cli.main import main
        with patch("sys.argv", ["openra-rl", "server", "logs", "--follow"]):
            main()
        mock_logs.assert_called_once_with(follow=True)

    @patch("openra_env.cli.commands.cmd_play")
    def test_main_play_with_flags(self, mock_play):
        from openra_env.cli.main import main
        with patch("sys.argv", [
            "openra-rl", "play",
            "--provider", "ollama",
            "--model", "qwen3:32b",
            "--verbose",
            "--port", "9000",
        ]):
            main()
        mock_play.assert_called_once_with(
            provider="ollama",
            model="qwen3:32b",
            api_key=None,
            difficulty="normal",
            verbose=True,
            port=9000,
            server_url=None,
            local=False,
            image_version=None,
        )

    @patch("openra_env.cli.commands.cmd_mcp_server")
    def test_main_mcp_server(self, mock_mcp):
        from openra_env.cli.main import main
        with patch("sys.argv", ["openra-rl", "mcp-server", "--port", "9000"]):
            main()
        mock_mcp.assert_called_once_with(server_url=None, port=9000)


# ── MCP Server ──────────────────────────────────────────────────────

class TestMCPServer:
    def test_mcp_server_module_imports(self):
        from openra_env.mcp_server import mcp
        assert mcp.name == "openra-rl"

    def test_format_dict(self):
        from openra_env.mcp_server import _format
        result = _format({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_format_string(self):
        from openra_env.mcp_server import _format
        assert _format("hello") == "hello"

    def test_server_url_default(self):
        from openra_env.mcp_server import _server_url
        assert _server_url == "http://localhost:8000"

    def test_all_tools_registered(self):
        from openra_env.mcp_server import mcp
        # The FastMCP instance should have tools registered
        # We can check by looking at the _tool_manager
        tools = mcp._tool_manager._tools if hasattr(mcp, '_tool_manager') else {}
        # At minimum these core tools should exist
        expected = [
            "start_game", "get_game_state", "advance",
            "build_unit", "build_structure", "move_units",
            "attack_move", "deploy_unit", "surrender",
        ]
        for name in expected:
            assert name in tools, f"Tool {name} not registered"

    def test_tool_count(self):
        from openra_env.mcp_server import mcp
        tools = mcp._tool_manager._tools if hasattr(mcp, '_tool_manager') else {}
        # We have 48 tools defined in the server
        assert len(tools) >= 40, f"Expected 40+ tools, got {len(tools)}"
