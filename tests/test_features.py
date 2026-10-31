import json

import numpy as np
import pytest

from ra_env import StepResult, pb
from ra_rl.features import (
    CAND_DIM,
    FEATURE_VERSION,
    GLOBAL_DIM,
    HOLD,
    KEEP,
    MAX_TARGETS,
    N_ARMY_ACTIONS,
    N_ITEM_ACTIONS,
    PRIV_DIM,
    WAIT,
    Features,
    privileged_features,
    state_potential,
)
from ra_rl.policies.net import RandomPolicy
from ra_rl.policies.scripted import Params, ScriptedPolicy
from ra_rl.record import GameBuffer
from ra_rl.vocab import ITEM_INDEX


def game_obs(tick=100) -> pb.GameObservation:
    o = pb.GameObservation(
        tick=tick,
        buildings=[pb.RlBuildingInfo(type="fact", cell_x=10, cell_y=10), pb.RlBuildingInfo(type="powr")],
        units=[pb.RlUnitInfo(type="2tnk", cell_x=12, cell_y=10), pb.RlUnitInfo(type="2tnk", cell_x=60, cell_y=60)],
        visible_enemies=[pb.RlUnitInfo(actor_id=7, type="3tnk", cell_x=62, cell_y=60)],
        visible_enemy_buildings=[pb.RlBuildingInfo(actor_id=9, type="barr", cell_x=70, cell_y=70)],
        production=[pb.RlProductionInfo(queue_type="Vehicle", item="2tnk")],
    )
    o.economy.cash, o.economy.power_provided, o.economy.power_drained = 1500, 100, 20
    return o


def army_req(*kinds, rid=3) -> pb.DecisionRequest:
    return pb.DecisionRequest(id=rid, kind=pb.DECIDE_ARMY, targets=[
        pb.TargetCandidate(kind=k, cell_x=60 + i, cell_y=60, type="barr" if k == "enemy_building" else "")
        for i, k in enumerate(kinds)])


def test_global_vector_is_deterministic_and_ignores_the_spectator_summary():
    assert FEATURE_VERSION == 2
    a, b = Features(), Features()
    a.observe(game_obs())
    o = game_obs()
    o.global_summary.enemy.cash = 99999
    o.global_summary.enemy.unit_value = 99999
    b.observe(o)
    assert a.g.shape == (GLOBAL_DIM,) and a.g.dtype == np.float32
    assert np.array_equal(a.g, b.g)
    assert np.isfinite(a.g).all()


def test_privileged_features_and_state_potential():
    o = game_obs()
    assert state_potential(o) == 0.0  # empty global_summary -> 0.0

    o.global_summary.own.unit_value = 2000
    o.global_summary.enemy.unit_value = 2000
    assert state_potential(o) == pytest.approx(0.0, abs=1e-6)  # symmetric start -> 0.0

    o.global_summary.own.building_value = 5000
    o.global_summary.enemy.building_value = 1000
    phi = state_potential(o)
    assert 0.2 < phi <= 1.0

    p_hard = privileged_features(o, "hard")
    p_brutal = privileged_features(o, "brutal")
    assert p_hard.shape == (PRIV_DIM,) and np.isfinite(p_hard).all()
    assert not np.array_equal(p_hard, p_brutal)  # opponent one-hot differs


def test_enemy_units_are_forgotten_buildings_remembered_longer_but_defenses_persist():
    f = Features()
    o = game_obs(tick=100)
    o.visible_enemy_buildings.append(pb.RlBuildingInfo(actor_id=42, type="tsla", cell_x=72, cell_y=70))
    f.observe(o)
    later = pb.GameObservation(tick=5000)
    f.observe(later)
    assert not f.enemy_units and not f.enemy_buildings
    assert 42 in f.seen_enemy_def_ids  # persistent turtle archetype memory


def test_item_mask_label_and_decision_round_trip():
    f = Features()
    req = pb.DecisionRequest(id=1, kind=pb.DECIDE_TRAIN, queue="Vehicle", options=["harv", "2tnk", "zzz"])
    mask = f.item_mask(req)
    assert mask.shape == (N_ITEM_ACTIONS,) and mask.sum() == 3 and mask[WAIT]
    y = f.label(req, pb.Decision(request_id=1, choice="2tnk"))
    assert y == ITEM_INDEX["2tnk"] and f.decision(req, y).choice == "2tnk"
    assert f.label(req, pb.Decision(request_id=1, choice="")) == WAIT
    assert f.decision(req, WAIT).choice == ""
    assert f.label(req, pb.Decision(request_id=1, choice="zzz")) is None


def test_item_mask_sanity_caps_prevent_harvester_and_barracks_spam():
    f = Features()
    o = pb.GameObservation(
        tick=2000,
        buildings=[pb.RlBuildingInfo(type="proc")] + [pb.RlBuildingInfo(type="tent")] * 4,
        units=[pb.RlUnitInfo(type="harv")] * 4,  # 1 proc -> cap is 4 harvesters
    )
    f.observe(o)
    veh_req = pb.DecisionRequest(id=1, kind=pb.DECIDE_TRAIN, queue="Vehicle", options=["harv", "2tnk"])
    veh_mask = f.item_mask(veh_req)
    assert not veh_mask[ITEM_INDEX["harv"]]
    assert veh_mask[ITEM_INDEX["2tnk"]] and veh_mask[WAIT]

    bld_req = pb.DecisionRequest(id=2, kind=pb.DECIDE_BUILD, queue="Building", options=["tent", "weap"])
    bld_mask = f.item_mask(bld_req)
    assert not bld_mask[ITEM_INDEX["tent"]]
    assert bld_mask[ITEM_INDEX["weap"]] and bld_mask[WAIT]


def test_consec_build_waits_breaks_fixed_point_state():
    f = Features()
    o = game_obs(tick=500)
    f.observe(o)
    g0 = f.g.copy()
    bld_req = pb.DecisionRequest(id=1, kind=pb.DECIDE_BUILD, queue="Building", options=["weap"])
    f.record(bld_req, pb.Decision(request_id=1, choice=""))
    f.observe(o)
    g1 = f.g.copy()
    assert f.consec_build_waits == 1
    assert not np.array_equal(g0, g1)

    f.record(bld_req, pb.Decision(request_id=1, choice="weap"))
    assert f.consec_build_waits == 0 and f.last_build_tick == 500


def test_army_actions_and_memory():
    f = Features()
    f.observe(game_obs())
    req = army_req("enemy_spawn", "enemy_building")
    cand, mask = f.candidates(req)
    assert cand.shape == (MAX_TARGETS, CAND_DIM) and mask.shape == (N_ARMY_ACTIONS,)
    assert mask[:2].all() and not mask[2:HOLD].any() and mask[HOLD] and mask[KEEP]
    assert f.label(req, None) == KEEP and f.decision(req, KEEP) is None
    assert f.label(req, pb.Decision(target_index=-1)) == HOLD and f.decision(req, HOLD).target_index == -1

    f.record(req, f.decision(req, 1))
    assert f.army.attacking and f.army.target == ("enemy_building", 61, 60)
    f.observe(game_obs(tick=400))
    cand, _ = f.candidates(req)
    assert cand[1, -1] == 1 and cand[0, -1] == 0  # "is the current target" flag
    f.record(req, None)
    assert f.army.aim_age == 1
    f.record(req, f.decision(req, HOLD))
    assert not f.army.attacking and f.army.target is None


def test_random_policy_answers_validly():
    p = RandomPolicy()
    p.reset(3)
    build = pb.DecisionRequest(id=1, kind=pb.DECIDE_BUILD, queue="Building", options=["powr", "tent"])
    for _ in range(20):
        step = StepResult(obs=game_obs(), tick=1, done=False, truncated=False, result="",
                          pending=[build, army_req("enemy_spawn")])
        for d in p.decide(step):
            if d.request_id == 1:
                assert d.choice in ("", "powr", "tent")
            else:
                assert d.target_index in (-1, 0)


def test_scripted_sync_army_follows_the_executed_order():
    p = ScriptedPolicy(Params(attack_value=100))
    req = army_req("enemy_spawn", "unexplored")
    o = pb.GameObservation(units=[pb.RlUnitInfo(type="2tnk")])
    before = p.army_state()
    [d] = p.decide(StepResult(obs=o, tick=1, done=False, truncated=False, result="", pending=[req]))
    assert d.target_index == 0 and p.attacking
    p.sync_army(before, req, None, o)          # someone else kept: nothing was launched
    assert not p.attacking and p.target is None
    p.sync_army(p.army_state(), req, 1, o)     # someone else attacked target 1
    assert p.attacking and p.target == ("unexplored", 61, 60)


def test_game_buffer_round_trip(tmp_path):
    f = Features()
    f.observe(game_obs())
    buf = GameBuffer()
    build = pb.DecisionRequest(id=1, kind=pb.DECIDE_BUILD, queue="Building", options=["powr"])
    buf.add(f, build, ITEM_INDEX["powr"])
    buf.add(f, army_req("enemy_spawn"), KEEP)
    path = tmp_path / "g000001.npz"
    buf.save(path, {"seed": 1, "outcome": "win"})
    z = np.load(path)
    assert z["prod_g"].shape == (1, GLOBAL_DIM) and z["prod_y"][0] == ITEM_INDEX["powr"]
    assert z["army_cand"].shape == (1, MAX_TARGETS, CAND_DIM) and z["army_y"][0] == KEEP
    assert json.loads(str(z["meta"]))["outcome"] == "win"


def test_model_respects_masks_and_bc_loads(tmp_path):
    torch = pytest.importorskip("torch")
    from ra_rl.bc import load
    from ra_rl.model import CommanderNet

    f = Features()
    f.observe(game_obs())
    for seed in (1, 20):
        buf = GameBuffer()
        buf.add(f, pb.DecisionRequest(id=1, kind=pb.DECIDE_BUILD, queue="Building", options=["powr"]),
                ITEM_INDEX["powr"])
        buf.add(f, army_req("enemy_spawn"), 0)
        buf.save(tmp_path / f"g{seed:06d}.npz", {"seed": seed, "outcome": "win"})
    train, val, stats = load([str(tmp_path)])
    assert stats["games"] == 2 and len(train["prod_y"]) == 1 and len(val["army_y"]) == 1

    net = CommanderNet(hidden=32, cand_hidden=16)
    g = torch.from_numpy(train["prod_g"]).float()
    logits = net.item_logits(g, torch.from_numpy(train["prod_queue"]).long(), torch.from_numpy(train["prod_mask"]))
    assert logits.shape == (1, N_ITEM_ACTIONS) and int(logits.argmax()) in (ITEM_INDEX["powr"], WAIT)
    a = net.army_logits(g, torch.from_numpy(train["army_cand"]).float(), torch.from_numpy(train["army_mask"]))
    assert a.shape == (1, N_ARMY_ACTIONS) and int(a.argmax()) in (0, HOLD, KEEP)

    net.save(str(tmp_path / "m.pt"))
    from ra_rl.policies import make_policy
    for spec in (f"net:{tmp_path / 'm.pt'}", f"net-t0.25:{tmp_path / 'm.pt'}"):
        p = make_policy(spec)
        p.reset(1)
        step = StepResult(obs=game_obs(), tick=1, done=False, truncated=False, result="", pending=[army_req("stale")])
        assert all(d.target_index in (-1, 0) for d in p.decide(step))
