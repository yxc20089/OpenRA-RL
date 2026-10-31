import pytest

from ra_env import GameSession, MatchConfig, SessionPoisoned, pb
from ra_env.determinism import first_divergence
from ra_env.probe import deploy_mcv_commands, obs_digest


class FakeBridge:
    """Advances a fake tick counter; optionally ends or poisons the game."""

    address = "fake:0"

    def __init__(self, done_at: int | None = None, poison_at: int | None = None):
        self.done_at = done_at
        self.poison_at = poison_at
        self.tick = 0
        self.created: list[dict] = []
        self.destroyed: list[str] = []
        self.advanced: list[int] = []

    def create_session(self, **kw) -> str:
        self.created.append(kw)
        self.tick = 0
        return f"s{len(self.created)}"

    def wait_ready(self, session_id: str) -> bool:
        return True

    def destroy_session(self, session_id: str) -> None:
        self.destroyed.append(session_id)

    def fast_advance(self, session_id: str, ticks: int, decisions=None, commands=None) -> pb.GameObservation:
        if self.poison_at is not None and self.tick + ticks >= self.poison_at:
            raise SessionPoisoned(session_id)
        self.advanced.append(ticks)
        self.tick += ticks
        done = self.done_at is not None and self.tick >= self.done_at
        return pb.GameObservation(tick=self.tick, done=done, result="win" if done else "")


def test_bots_spec_maps_friendly_names():
    assert MatchConfig(bot="hard").bots_spec() == "Multi1:rl-agent,Multi0:normal"
    assert MatchConfig(bot="brutal").bots_spec() == "Multi1:rl-agent,Multi0:rush"
    with pytest.raises(ValueError):
        MatchConfig(bot="nope").bots_spec()


def test_reset_passes_match_and_seed():
    bridge = FakeBridge()
    s = GameSession(bridge, MatchConfig(map_name="m.oramap", bot="easy"))
    step = s.reset(seed=7)
    assert step.tick == 1
    assert bridge.created[0]["seed"] == 7
    assert bridge.created[0]["map_name"] == "m.oramap"
    assert bridge.created[0]["player_faction"] == "RandomAllies"


def test_step_before_reset_fails():
    with pytest.raises(RuntimeError):
        GameSession(FakeBridge(), MatchConfig()).step([], 10)


def test_truncation_clamps_last_step_to_cap():
    bridge = FakeBridge()
    s = GameSession(bridge, MatchConfig(max_ticks=250))
    s.reset(seed=1)
    steps = [s.step([], 100) for _ in range(3)]
    assert [st.tick for st in steps] == [101, 201, 250]
    assert bridge.advanced[-1] == 49
    assert steps[-1].truncated and not steps[-1].done
    assert not steps[0].truncated


def test_game_over_reports_result():
    s = GameSession(FakeBridge(done_at=150), MatchConfig())
    s.reset(seed=1)
    step = s.step([], 100)
    assert not step.done
    step = s.step([], 100)
    assert step.done and step.result == "win" and not step.truncated


def test_poisoned_session_is_never_reused():
    bridge = FakeBridge(poison_at=150)
    s = GameSession(bridge, MatchConfig())
    s.reset(seed=1)
    with pytest.raises(SessionPoisoned):
        s.step([], 200)
    assert s.session_id == ""
    s.close()
    assert bridge.destroyed == []  # the daemon already destroyed it


def test_reset_closes_previous_session():
    bridge = FakeBridge()
    s = GameSession(bridge, MatchConfig())
    s.reset(seed=1)
    s.reset(seed=2)
    assert bridge.destroyed == ["s1"]


def test_deploy_mcv_commands_targets_only_mcvs():
    obs = pb.GameObservation(units=[pb.RlUnitInfo(actor_id=5, type="mcv"), pb.RlUnitInfo(actor_id=6, type="e1")])
    cmds = deploy_mcv_commands(obs)
    assert [(c.action, c.actor_id) for c in cmds] == [(pb.DEPLOY, 5)]


def test_obs_digest_is_order_independent_and_state_sensitive():
    a = pb.GameObservation(tick=10, units=[pb.RlUnitInfo(actor_id=1, type="e1"), pb.RlUnitInfo(actor_id=2, type="e3")])
    b = pb.GameObservation(tick=10, units=[pb.RlUnitInfo(actor_id=2, type="e3"), pb.RlUnitInfo(actor_id=1, type="e1")])
    c = pb.GameObservation(tick=10, units=[pb.RlUnitInfo(actor_id=1, type="e1", cell_x=3)])
    assert obs_digest(a) == obs_digest(b)
    assert obs_digest(a) != obs_digest(c)


def test_first_divergence():
    assert first_divergence(["a", "b"], ["a", "b"]) == -1
    assert first_divergence(["a", "b"], ["a", "x"]) == 1
    assert first_divergence(["a", "b"], ["a"]) == 1


def test_step_passes_decisions_and_returns_pending_and_teacher():
    class DecisionBridge(FakeBridge):
        def fast_advance(self, session_id, ticks, decisions=None, commands=None):
            self.sent = list(decisions or [])
            o = super().fast_advance(session_id, ticks)
            o.pending_decisions.add(id=7, kind=pb.DECIDE_BUILD, queue="Building", options=["powr"])
            o.teacher_decisions.add().decision.choice = "tent"
            return o

    bridge = DecisionBridge()
    s = GameSession(bridge, MatchConfig(brain="builtin"))
    s.reset(seed=1)
    step = s.step([pb.Decision(request_id=3, choice="proc")], 50)
    assert [d.choice for d in bridge.sent] == ["proc"]
    assert [r.id for r in step.pending] == [7]
    assert [t.decision.choice for t in step.teacher] == ["tent"]
    assert bridge.created[0]["agent_brain"] == "builtin"


def test_unknown_brain_rejected():
    with pytest.raises(ValueError):
        GameSession(FakeBridge(), MatchConfig(brain="nope"))
