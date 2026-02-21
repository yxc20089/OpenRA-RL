"""Tests for llm_agent helper functions."""

from openra_env.agent import _sanitize_messages


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
