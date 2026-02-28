"""Tests for llm_agent helper functions."""

import pytest

from openra_env.agent import _format_llm_api_error, _sanitize_messages
from openra_env.config import LLMConfig


class TestSanitizeMessages:
    """Tests for _sanitize_messages — merges consecutive same-role messages."""

    def test_empty(self):
        assert _sanitize_messages([]) == []

    def test_no_merge_needed(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _sanitize_messages(msgs)
        assert len(result) == 3
        assert [m["role"] for m in result] == ["system", "user", "assistant"]

    def test_consecutive_user_merged(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        result = _sanitize_messages(msgs)
        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert "first" in result[1]["content"]
        assert "second" in result[1]["content"]

    def test_three_consecutive_user_merged(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        result = _sanitize_messages(msgs)
        assert len(result) == 2
        assert result[1]["content"] == "a\n\nb\n\nc"

    def test_does_not_mutate_original(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        _sanitize_messages(msgs)
        # Original messages should be untouched
        assert msgs[0]["content"] == "first"
        assert msgs[1]["content"] == "second"
        assert len(msgs) == 2

    def test_mixed_roles_preserved(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a2"},
        ]
        result = _sanitize_messages(msgs)
        assert [m["role"] for m in result] == ["system", "user", "assistant", "user", "assistant"]
        assert result[3]["content"] == "u2\n\nu3"

    def test_tool_then_user_gets_bridge_assistant(self):
        """Mistral requires tool → assistant → user, not tool → user."""
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "result1", "tool_call_id": "1"},
            {"role": "user", "content": "briefing"},
        ]
        result = _sanitize_messages(msgs)
        assert len(result) == 4
        assert [m["role"] for m in result] == ["assistant", "tool", "assistant", "user"]
        assert result[2]["content"]  # bridge message is non-empty

    def test_tool_then_assistant_no_extra_bridge(self):
        """When tool → assistant already exists, no bridge is inserted."""
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "result1", "tool_call_id": "1"},
            {"role": "assistant", "content": "Got the result."},
        ]
        result = _sanitize_messages(msgs)
        assert len(result) == 3
        assert [m["role"] for m in result] == ["assistant", "tool", "assistant"]

    def test_real_world_scenario(self):
        """Simulates: nudge (user) → next turn briefing (user) → should merge."""
        msgs = [
            {"role": "system", "content": "You are playing Red Alert."},
            {"role": "user", "content": "STRATEGIC BRIEFING: ..."},
            {"role": "assistant", "content": "I will deploy the MCV."},
            {"role": "user", "content": "Continue playing. Use game tools."},
            {"role": "user", "content": "TURN BRIEFING: Funds 5000, ..."},
        ]
        result = _sanitize_messages(msgs)
        assert len(result) == 4
        roles = [m["role"] for m in result]
        assert roles == ["system", "user", "assistant", "user"]
        assert "Continue playing" in result[3]["content"]
        assert "TURN BRIEFING" in result[3]["content"]

    def test_game_loop_tool_then_briefing(self):
        """Real scenario: tool results from turn N, then briefing user msg for turn N+1."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "initial briefing"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "content": '{"ok": true}', "tool_call_id": "c1"},
            {"role": "user", "content": "TURN BRIEFING: tick 500"},
        ]
        result = _sanitize_messages(msgs)
        roles = [m["role"] for m in result]
        assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]
        assert result[4]["role"] == "assistant"  # bridge
        assert result[5]["content"] == "TURN BRIEFING: tick 500"


class TestFormatLLMApiError:
    """Tests for provider error mapping helper."""

    def test_openrouter_tool_route_error_has_actionable_hint(self):
        cfg = LLMConfig(
            base_url="https://openrouter.ai/api/v1/chat/completions",
            model="liquid/lfm-2.5-1.2b-thinking:free",
        )
        msg = _format_llm_api_error(
            404,
            (
                '{"error":{"message":"No endpoints found that support tool use.'
                ' To learn more about provider routing","code":404}}'
            ),
            cfg,
        )
        assert "supports tool calling" in msg
        assert "OpenRA-RL requires tool-calling models" in msg
        assert "not ':free'" in msg

    def test_auth_error_message_preserved(self):
        cfg = LLMConfig(model="foo/bar")
        msg = _format_llm_api_error(401, "unauthorized", cfg)
        assert "Authentication failed (401)" in msg


class TestToolCallingPreflight:
    """Tests for startup preflight capability checks."""

    @pytest.mark.asyncio
    async def test_openrouter_unsupported_tools_is_blocked(self, monkeypatch):
        from openra_env import agent as agent_mod

        cfg = LLMConfig(
            base_url="https://openrouter.ai/api/v1/chat/completions",
            model="liquid/lfm-2.5-1.2b-thinking:free",
        )

        async def _fake_chat_completion(*args, **kwargs):
            raise RuntimeError("No endpoints found that support tool use.")

        monkeypatch.setattr(agent_mod, "chat_completion", _fake_chat_completion)
        ok, err = await agent_mod._preflight_tool_calling_support(cfg)
        assert ok is False
        assert "support tool use" in err.lower()

    @pytest.mark.asyncio
    async def test_non_openrouter_skips_preflight_call(self, monkeypatch):
        from openra_env import agent as agent_mod

        cfg = LLMConfig(
            base_url="http://localhost:11434/v1/chat/completions",
            model="qwen3:4b",
        )
        called = False

        async def _fake_chat_completion(*args, **kwargs):
            nonlocal called
            called = True
            return {}

        monkeypatch.setattr(agent_mod, "chat_completion", _fake_chat_completion)
        ok, err = await agent_mod._preflight_tool_calling_support(cfg)
        assert ok is True
        assert err == ""
        assert called is False
