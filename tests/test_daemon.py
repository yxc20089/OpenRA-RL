"""Integration tests against a real daemon: docker compose up -d, then
    pytest -m daemon
Address override: RA_DAEMON=host:port.
"""

import os

import pytest

from ra_env import Bridge, GameSession, MatchConfig, pb
from ra_env.determinism import describe, trace
from ra_env.probe import deploy_mcv_commands

pytestmark = pytest.mark.daemon

ADDRESS = os.environ.get("RA_DAEMON", "localhost:9999")


@pytest.fixture(scope="module")
def bridge():
    b = Bridge(ADDRESS)
    yield b
    b.close()


def test_game_starts_with_an_mcv_that_deploys(bridge):
    s = GameSession(bridge, MatchConfig(bot="easy", max_ticks=2000))
    try:
        step = s.reset(seed=42)
        cmds = deploy_mcv_commands(step.obs)
        assert cmds, "agent should start with an MCV"
        step = s.step([], 200, cmds)
        assert any(b.type == "fact" for b in step.obs.buildings)
        assert step.obs.map_info.width > 0
    finally:
        s.close()


def test_same_seed_same_game(bridge):
    match = MatchConfig(bot="hard", max_ticks=4000)
    a = trace(bridge, match, 99, 100)
    b = trace(bridge, match, 99, 100)
    assert describe(a, b) == "identical"


def test_builtin_brain_reports_teacher_decisions(bridge):
    s = GameSession(bridge, MatchConfig(bot="easy", brain="builtin", max_ticks=3000))
    try:
        step = s.reset(seed=5)
        teacher = []
        while not (step.done or step.truncated):
            step = s.step([], 500)
            assert not step.pending
            for t in step.teacher:
                assert 0 <= step.tick - t.request.tick <= 2  # paired with the state it was taken in
                if t.request.kind != pb.DECIDE_ARMY:
                    # production starts on the next step, as for an external agent
                    assert not any(p.queue_type == t.request.queue for p in step.obs.production) or not t.decision.choice
            teacher += step.teacher
        kinds = {t.request.kind for t in teacher}
        assert {pb.DECIDE_BUILD, pb.DECIDE_TRAIN, pb.DECIDE_ARMY} <= kinds
        assert any(t.decision.choice == "powr" for t in teacher)
        assert sum(b.type == "powr" for b in step.obs.buildings) >= 1
    finally:
        s.close()


def test_external_brain_stops_for_decisions_and_obeys(bridge):
    s = GameSession(bridge, MatchConfig(bot="easy", brain="external", max_ticks=4000))
    try:
        step = s.reset(seed=5)
        answer, asked = [], 0
        while not (step.done or step.truncated):
            step = s.step(answer, 1000)
            answer = []
            for r in step.pending:
                asked += 1
                assert 0 <= step.tick - r.tick <= 2  # the advance stopped right away
                if r.kind == pb.DECIDE_BUILD and "powr" in r.options:
                    answer.append(pb.Decision(request_id=r.id, choice="powr"))
        assert asked > 0
        assert sum(b.type == "powr" for b in step.obs.buildings) >= 2
    finally:
        s.close()
