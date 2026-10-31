from ra_env import StepResult, pb
from ra_rl.eval import wilson
from ra_rl.policies import BuiltinPolicy, make_policy
from ra_rl.policies.scripted import Params, ScriptedPolicy


def obs(buildings=(), units=(), power=(100, 0), tick=100) -> pb.GameObservation:
    o = pb.GameObservation(
        tick=tick,
        buildings=[pb.RlBuildingInfo(type=t, cell_x=10, cell_y=10) for t in buildings],
        units=[pb.RlUnitInfo(type=t, cell_x=12, cell_y=10) for t in units],
    )
    o.economy.power_provided, o.economy.power_drained = power
    return o


def step(o: pb.GameObservation, *requests: pb.DecisionRequest) -> StepResult:
    return StepResult(obs=o, tick=o.tick, done=False, truncated=False, result="", pending=list(requests))


def build_req(*options, queue="Building", rid=1):
    return pb.DecisionRequest(id=rid, kind=pb.DECIDE_BUILD, queue=queue, options=list(options))


def train_req(queue, *options, rid=2):
    return pb.DecisionRequest(id=rid, kind=pb.DECIDE_TRAIN, queue=queue, options=list(options))


def army_req(*targets, rid=3):
    return pb.DecisionRequest(id=rid, kind=pb.DECIDE_ARMY, targets=[
        pb.TargetCandidate(kind=k, cell_x=i, cell_y=i) for i, k in enumerate(targets)])


def test_make_policy():
    assert make_policy("builtin").brain == "builtin"
    assert make_policy("scripted").brain == "external"
    assert BuiltinPolicy().decide(step(obs(), build_req("powr"))) == []


def test_build_follows_order_and_answers_every_request():
    p = ScriptedPolicy()
    [d] = p.decide(step(obs(buildings=["fact"]), build_req("powr", "tent", "proc")))
    assert (d.request_id, d.choice) == (1, "powr")
    [d] = p.decide(step(obs(buildings=["fact", "powr"]), build_req("powr", "tent", "proc")))
    assert d.choice == "tent"


def test_build_powers_up_before_a_consumer():
    p = ScriptedPolicy()
    [d] = p.decide(step(obs(buildings=["fact", "powr", "tent"], power=(100, 80)), build_req("powr", "proc", "tent")))
    assert d.choice == "powr"


def test_defense_waits_for_war_factory_unless_early_rush_with_refinery():
    p = ScriptedPolicy()
    [d] = p.decide(step(obs(buildings=["fact"]), build_req("pbox", queue="Defense")))
    assert d.choice == ""
    [d] = p.decide(step(obs(buildings=["fact", "weap"]), build_req("pbox", queue="Defense")))
    assert d.choice == "pbox"

    # Under early rush with >=1 refinery, emergency pbox unlocks before weap
    p2 = ScriptedPolicy()
    rush_obs = obs(buildings=["fact", "powr", "tent", "proc"], tick=2500)
    rush_obs.visible_enemies.extend([pb.RlUnitInfo(type="e1", cell_x=14, cell_y=10)] * 8)
    [d] = p2.decide(step(rush_obs, build_req("pbox", queue="Defense")))
    assert d.choice == "pbox"


def test_train_harvesters_then_tanks_then_artillery_and_siege_adaptation():
    p = ScriptedPolicy(Params(harv_per_proc=2, arty_every=4, siege_arty_every=2, siege_def_threshold=3))
    opts = ("harv", "1tnk", "2tnk", "arty")
    [d] = p.decide(step(obs(buildings=["proc"], units=["harv"]), train_req("Vehicle", *opts)))
    assert d.choice == "harv"
    [d] = p.decide(step(obs(buildings=["proc"], units=["harv"] * 2), train_req("Vehicle", *opts)))
    assert d.choice == "2tnk"
    [d] = p.decide(step(obs(buildings=["proc"], units=["harv"] * 2 + ["2tnk"] * 4), train_req("Vehicle", *opts)))
    assert d.choice == "arty"
    [d] = p.decide(step(obs(), train_req("Aircraft", "heli")))
    assert d.choice == ""

    # Spotting >=3 static defenses activates siege artillery ratio (1 arty per 2 tanks)
    turtle_obs = obs(buildings=["proc"], units=["harv"] * 2 + ["2tnk"] * 2)
    for aid in (10, 11, 12):
        turtle_obs.visible_enemy_buildings.append(pb.RlBuildingInfo(actor_id=aid, type="tsla"))
    [d] = p.decide(step(turtle_obs, train_req("Vehicle", *opts)))
    assert d.choice == "arty"


def test_army_attacks_keeps_reinforces_and_recalls():
    p = ScriptedPolicy(Params(attack_value=1000, reinforce_value=800, recall_fraction=0.5))
    req = army_req("enemy_spawn", "enemy_building")
    away = [pb.RlUnitInfo(type="2tnk", cell_x=90, cell_y=10)] * 2
    yard = [pb.RlBuildingInfo(type="fact", cell_x=10, cell_y=40)]

    def at(units):
        return pb.GameObservation(buildings=yard, units=units)

    assert p.decide(step(at(away[:1]), req)) == []                              # too small: keep
    [d] = p.decide(step(at(away), req))
    assert d.target_index == 1                                                  # nearest building
    assert p.decide(step(at(away), req)) == []                                  # keep going
    home = [pb.RlUnitInfo(type="2tnk", cell_x=12, cell_y=40)] * 1
    [d] = p.decide(step(at(away + home), req))
    assert d.target_index == 1                                                  # reinforce
    [d] = p.decide(step(at(away[:1]), req))
    assert d.target_index == -1                                                 # bled out: recall


def test_army_defaults_to_enemy_spawn():
    p = ScriptedPolicy(Params(attack_value=100))
    [d] = p.decide(step(obs(units=["2tnk"]), army_req("enemy_spawn")))
    assert d.target_index == 0


def test_wilson_interval():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(60, 100)
    assert 0.50 < lo < 0.51 and 0.69 < hi < 0.70


def test_army_moves_on_when_target_disappears_and_hunts_unexplored():
    p = ScriptedPolicy(Params(attack_value=100, reinforce_value=10**9))
    [d] = p.decide(step(obs(units=["2tnk"]), army_req("enemy_spawn", "unexplored")))
    assert d.target_index == 0
    assert p.decide(step(obs(units=["2tnk"]), army_req("enemy_spawn", "unexplored"))) == []
    # spawn cleared: only unexplored sectors are left
    [d] = p.decide(step(obs(units=["2tnk"]), army_req("unexplored")))
    assert d.target_index == 0 and p.target[0] == "unexplored"


def test_army_gives_up_on_a_stuck_target_and_cycles_cleanly():
    p = ScriptedPolicy(Params(attack_value=100, reinforce_value=10**9, max_aim_decisions=2))
    req = army_req("unexplored", "unexplored")  # cells (0,0) and (1,1)
    [d] = p.decide(step(obs(units=["2tnk"]), req))
    assert d.target_index == 0
    assert p.decide(step(obs(units=["2tnk"]), req)) == []
    [d] = p.decide(step(obs(units=["2tnk"]), req))
    assert d.target_index == 1  # (0,0) never cleared: skip it
    # If (1,1) also gets skipped, skipped resets cleanly rather than crashing
    assert p.decide(step(obs(units=["2tnk"]), req)) == []
    [d] = p.decide(step(obs(units=["2tnk"]), req))
    assert d.target_index == 0
