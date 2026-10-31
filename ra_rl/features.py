"""Observation -> network inputs, and decisions <-> network actions.

The same code encodes the teacher's decisions when recording and the policy's
inputs when playing, so both see exactly the same features. Only the agent's own
knowledge is used in `Features.g`: its units and buildings, what it sees, what it
remembers having seen, and its own past production/army orders. The spectator
summary (`global_summary.enemy`) is NEVER read by `Features.observe` or `Features.g`.

For Centralized Training with Decentralized Execution (CTDE) in RL (Phases 3-4),
`privileged_features()` and `state_potential()` separately read `global_summary`
for the training-only critic and exact Potential-Based Reward Shaping (PBRS).

Per game:  feats.reset() once, then each step  feats.observe(step.obs)  before
encoding that step's requests, and  feats.record(req, decision)  for every
decision actually executed (an ARMY request left unanswered is decision None).
"""

from dataclasses import dataclass

import numpy as np

from ra_env import pb
from ra_rl.vocab import (
    BUILDING_INDEX,
    BUILDINGS,
    COST,
    ITEM_INDEX,
    ITEMS,
    KIND_INDEX,
    QUEUE_INDEX,
    QUEUES,
    TARGET_KINDS,
    UNIT_INDEX,
    UNITS,
    base_type,
    combat_value,
)

FEATURE_VERSION = 2

MAX_TARGETS = 12               # RLCommander.MaxTargets
WAIT = len(ITEMS)              # item head: "produce nothing for now"
N_ITEM_ACTIONS = len(ITEMS) + 1
HOLD = MAX_TARGETS             # army head: call off the attack
KEEP = MAX_TARGETS + 1         # army head: no answer, keep the current orders
N_ARMY_ACTIONS = MAX_TARGETS + 2

HOME_CELLS = 20                # around the base center counts as "home"
NEAR_CELLS = 10                # around a target counts as "at the target"
ENEMY_UNIT_MEMORY = 750        # ticks a seen enemy unit is still counted
ENEMY_BUILDING_MEMORY = 3000   # ticks a seen enemy building is still counted

STATIC_DEFENSES = frozenset({"pbox", "hbox", "gun", "ftur", "tsla", "agun", "sam"})
HEAVY_ARMOR = frozenset({"2tnk", "3tnk", "4tnk", "v2rl", "ttnk", "ctnk", "qtnk", "stnk"})

# Sanity caps for non-combat / redundant spam in external commander mode.
MAX_HARVESTERS = 10
MIN_HARVESTER_CAP = 4
MAX_BARRACKS = 4
MAX_PILLBOXES = 6
MAX_MCV_TOTAL = 2

N_SCALARS = 31
N_MEMORY = 15
GLOBAL_DIM = (N_SCALARS + len(QUEUES) + len(BUILDINGS) + len(UNITS) + len(ITEMS)
              + len(BUILDINGS) + len(UNITS) + N_MEMORY)
CAND_DIM = len(TARGET_KINDS) + len(BUILDINGS) + 8

# Training-only asymmetric critic inputs (never used by the actor or at eval time).
OPPONENTS = ("hard", "brutal", "turtle", "medium", "easy", "other")
OPPONENT_INDEX = {
    "hard": 0, "normal": 0,
    "brutal": 1, "rush": 1,
    "turtle": 2,
    "medium": 3,
    "easy": 4, "beginner": 4,
    "other": 5, "naval": 5,
}
N_PRIV_SCALARS = 14
PRIV_DIM = N_PRIV_SCALARS + len(OPPONENTS)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def opponent_id(bot_name: str) -> int:
    return OPPONENT_INDEX.get(bot_name, OPPONENT_INDEX["other"])


def privileged_features(obs: pb.GameObservation, bot_name: str = "other") -> np.ndarray:
    """Training-only spectator features for the asymmetric critic V(g, priv)."""
    own = obs.global_summary.own
    ene = obs.global_summary.enemy

    own_u = float(max(0, own.unit_value))
    own_b = float(max(0, own.building_value))
    own_nb = float(max(0, own.n_buildings))
    own_earn = float(max(0, own.earned))

    ene_cash = float(max(0, ene.cash))
    ene_u = float(max(0, ene.unit_value))
    ene_b = float(max(0, ene.building_value))
    ene_nb = float(max(0, ene.n_buildings))
    ene_earn = float(max(0, ene.earned))

    own_tot = own_u + own_b
    ene_tot = ene_u + ene_b

    scalars = np.asarray([
        np.clip(ene_cash / 5000.0, 0.0, 4.0),
        np.log1p(ene_cash) / 10.0,
        np.clip(ene_u / 10000.0, 0.0, 4.0),
        np.clip(ene_b / 10000.0, 0.0, 4.0),
        np.clip(ene_nb / 20.0, 0.0, 3.0),
        np.clip(ene_earn / 20000.0, 0.0, 4.0),
        np.clip(own_earn / 20000.0, 0.0, 4.0),
        np.clip((own_u - ene_u) / 10000.0, -3.0, 3.0),
        own_u / max(500.0, own_u + ene_u),
        np.clip((own_b - ene_b) / 10000.0, -3.0, 3.0),
        own_b / max(500.0, own_b + ene_b),
        np.clip((own_nb - ene_nb) / 15.0, -3.0, 3.0),
        np.clip((own_earn - ene_earn) / 15000.0, -3.0, 3.0),
        own_tot / max(1000.0, own_tot + ene_tot),
    ], dtype=np.float32)

    opp_oh = np.zeros(len(OPPONENTS), dtype=np.float32)
    opp_oh[opponent_id(bot_name)] = 1.0
    out = np.concatenate([scalars, opp_oh]).astype(np.float32)
    assert out.shape == (PRIV_DIM,)
    return out


def state_potential(obs: pb.GameObservation) -> float:
    """Bounded potential Phi(s) in [-1, 1] for exact telescoping PBRS (Ng et al., 1999).

    Emphasizes structural dominance (65%) + total military/structural share (35%) so
    passive harvester/cash hoarding at home yields no net potential gain, while
    destroying enemy refineries/factories immediately increases Phi(s).
    """
    own = obs.global_summary.own
    ene = obs.global_summary.enemy
    own_b = float(max(0, own.building_value))
    ene_b = float(max(0, enemy_b_val := ene.building_value))
    own_u = float(max(0, own.unit_value))
    ene_u = float(max(0, ene.unit_value))

    if own_b == 0.0 and ene_b == 0.0 and own_u == 0.0 and ene_u == 0.0:
        return 0.0

    bld_adv = (own_b - ene_b) / max(1500.0, own_b + ene_b)
    own_tot = own_b + own_u
    ene_tot = enemy_b_val + ene_u
    tot_adv = (own_tot - max(0.0, ene_tot)) / max(2000.0, own_tot + max(0.0, ene_tot))
    return float(np.clip(0.65 * bld_adv + 0.35 * tot_adv, -1.0, 1.0))


@dataclass
class ArmyMemory:
    """The agent's own army orders (what it decided, not what the bot did)."""

    attacking: bool = False
    attack_tick: int = 0
    order_tick: int = 0
    peak: int = 0
    target: tuple[str, int, int] | None = None
    aim_age: int = 0
    last: int = 0  # last ARMY answer: 0 keep, 1 hold, 2 attack


class Features:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.enemy_units: dict[int, tuple[str, int, tuple[int, int]]] = {}
        self.enemy_buildings: dict[int, tuple[str, int]] = {}
        self.seen_enemy_def_ids: set[int] = set()
        self.seen_enemy_heavy_ids: set[int] = set()
        self.early_rush_seen: bool = False
        self.max_early_threat: float = 0.0

        self.army = ArmyMemory()
        self.last_build_tick: int = 0
        self.consec_build_waits: int = 0
        self.ready_idle_tick: int = 0

        self.tick = 0
        self.g = np.zeros(GLOBAL_DIM, np.float32)
        self._own_b = np.zeros(len(BUILDINGS), np.float32)
        self._own_u = np.zeros(len(UNITS), np.float32)
        self._producing = np.zeros(len(ITEMS), np.float32)
        self.base: tuple[float, float] = (0.0, 0.0)
        self.away_center: tuple[float, float] | None = None
        self.own_units: list[tuple[int, int, int]] = []   # (x, y, combat value)
        self.army_value = 0

    # ── per step ─────────────────────────────────────────────────────────────

    def observe(self, obs: pb.GameObservation) -> None:
        self.tick = tick = obs.tick
        for u in obs.visible_enemies:
            ut = base_type(u.type)
            self.enemy_units[u.actor_id] = (ut, tick, (u.cell_x, u.cell_y))
            if ut in HEAVY_ARMOR:
                self.seen_enemy_heavy_ids.add(u.actor_id)
        for b in obs.visible_enemy_buildings:
            bt = base_type(b.type)
            self.enemy_buildings[b.actor_id] = (bt, tick)
            if bt in STATIC_DEFENSES:
                self.seen_enemy_def_ids.add(b.actor_id)
        self.enemy_units = {k: v for k, v in self.enemy_units.items() if tick - v[1] <= ENEMY_UNIT_MEMORY}
        self.enemy_buildings = {k: v for k, v in self.enemy_buildings.items()
                                if tick - v[1] <= ENEMY_BUILDING_MEMORY}

        yards = [(b.cell_x, b.cell_y) for b in obs.buildings if base_type(b.type) == "fact"]
        if yards:
            self.base = yards[0]
        elif obs.buildings:
            self.base = (float(np.mean([b.cell_x for b in obs.buildings])),
                         float(np.mean([b.cell_y for b in obs.buildings])))
        elif obs.units:
            self.base = (float(np.mean([u.cell_x for u in obs.units])), float(np.mean([u.cell_y for u in obs.units])))

        self.own_units = [(u.cell_x, u.cell_y, combat_value(u.type)) for u in obs.units]
        self.army_value = sum(v for _, _, v in self.own_units)
        home = sum(v for x, y, v in self.own_units if _dist((x, y), self.base) <= HOME_CELLS)
        away = [(x, y, v) for x, y, v in self.own_units if v and _dist((x, y), self.base) > HOME_CELLS]
        self.away_center = None
        if away:
            w = sum(v for _, _, v in away)
            self.away_center = (sum(x * v for x, _, v in away) / w, sum(y * v for _, y, v in away) / w)
        if self.army.attacking:
            self.army.peak = max(self.army.peak, self.army_value)
        if self.army.attacking or self.army_value < 4000:
            self.ready_idle_tick = tick

        visible_value = sum(combat_value(u.type) for u in obs.visible_enemies)
        threat = sum(combat_value(u.type) for u in obs.visible_enemies
                     if _dist((u.cell_x, u.cell_y), self.base) <= HOME_CELLS)
        recent_value = sum(combat_value(t) for t, _, _ in self.enemy_units.values())

        if tick <= 4500 and threat > 0:
            self.early_rush_seen = True
        if tick <= 6000:
            self.max_early_threat = max(self.max_early_threat, float(threat))

        busy = np.zeros(len(QUEUES), np.float32)
        producing = np.zeros(len(ITEMS), np.float32)
        for p in obs.production:
            if p.queue_type in QUEUE_INDEX:
                busy[QUEUE_INDEX[p.queue_type]] = 1
            i = ITEM_INDEX.get(base_type(p.item))
            if i is not None:
                producing[i] += 1

        own_b = np.zeros(len(BUILDINGS), np.float32)
        for b in obs.buildings:
            i = BUILDING_INDEX.get(base_type(b.type))
            if i is not None:
                own_b[i] += 1
        own_u = np.zeros(len(UNITS), np.float32)
        for u in obs.units:
            i = UNIT_INDEX.get(base_type(u.type))
            if i is not None:
                own_u[i] += 1
        enemy_b = np.zeros(len(BUILDINGS), np.float32)
        for t, _ in self.enemy_buildings.values():
            i = BUILDING_INDEX.get(t)
            if i is not None:
                enemy_b[i] += 1
        enemy_u = np.zeros(len(UNITS), np.float32)
        for t, _, _ in self.enemy_units.values():
            i = UNIT_INDEX.get(t)
            if i is not None:
                enemy_u[i] += 1

        self._own_b = own_b
        self._own_u = own_u
        self._producing = producing

        own_proc = float(own_b[BUILDING_INDEX["proc"]])
        own_harv_total = float(own_u[UNIT_INDEX["harv"]] + producing[ITEM_INDEX["harv"]])
        has_weap = float(own_b[BUILDING_INDEX["weap"]] + producing[ITEM_INDEX["weap"]] > 0)
        has_radar = float(own_b[BUILDING_INDEX["dome"]] > 0)
        has_tech = float(own_b[BUILDING_INDEX["atek"]] + own_b[BUILDING_INDEX["stek"]] > 0)

        e, m = obs.economy, obs.military
        scalars = [
            tick / 20000, e.cash / 3000, e.ore / 3000, np.log1p(max(0, e.cash + e.ore)) / 10,
            e.power_provided / 500, e.power_drained / 500,
            np.clip((e.power_provided - e.power_drained) / 200, -3, 3),
            e.resource_capacity / 5000, e.harvester_count / 8, obs.explored_percent / 100,
            m.kills_cost / 20000, m.deaths_cost / 20000,
            self.army_value / 10000, home / 10000, (self.army_value - home) / 10000,
            len(obs.visible_enemies) / 20, visible_value / 10000, threat / 5000, recent_value / 10000,
            float(bool(yards)), len(obs.buildings) / 30,
            # v2 economy/saturation, tech gates & persistent fog-honest scouting
            np.clip(own_harv_total / max(1.0, 2.0 * own_proc), 0.0, 3.0),
            np.clip((own_harv_total - 2.0 * own_proc) / 4.0, -2.0, 2.0),
            has_weap,
            has_radar,
            has_tech,
            float(self.early_rush_seen),
            min(2.0, self.max_early_threat / 5000.0),
            min(3.0, len(self.seen_enemy_def_ids) / 6.0),
            min(3.0, len(self.seen_enemy_heavy_ids) / 10.0),
            np.clip((m.kills_cost - m.deaths_cost) / 10000.0, -2.0, 2.0),
        ]

        a = self.army
        ready_idle = (
            min(2.0, max(0, tick - self.ready_idle_tick) / 2500.0)
            if (not a.attacking and self.army_value >= 4000)
            else 0.0
        )
        memory = [
            float(a.attacking), min(2.0, (tick - a.attack_tick) / 5000) if a.attacking else 0.0,
            min(2.0, (tick - a.order_tick) / 2000), a.peak / 10000,
            self.army_value / a.peak if a.attacking and a.peak else 0.0, min(2.0, a.aim_age / 8),
            *(float(a.target is not None and a.target[0] == k) for k in TARGET_KINDS),
            *(float(a.last == i) for i in (1, 2)),
            # v2 anti-deadlock stagnation counters
            min(2.0, max(0, tick - self.last_build_tick) / 2000.0),
            min(2.0, self.consec_build_waits / 5.0),
            ready_idle,
        ]

        self.g = np.concatenate([
            np.asarray(scalars, np.float32), busy,
            np.minimum(own_b, 20) / 4, np.minimum(own_u, 40) / 8, producing,
            np.minimum(enemy_b, 20) / 4, np.minimum(enemy_u, 40) / 8,
            np.asarray(memory, np.float32),
        ]).astype(np.float32)
        assert self.g.shape == (GLOBAL_DIM,)

    # ── per request ──────────────────────────────────────────────────────────

    def item_mask(self, req: pb.DecisionRequest) -> np.ndarray:
        mask = np.zeros(N_ITEM_ACTIONS, bool)
        for o in req.options:
            i = ITEM_INDEX.get(base_type(o))
            if i is not None:
                mask[i] = True

        # Sanity caps on non-combat / redundant spam (external mode bypasses YAML UnitLimits).
        own_proc = int(self._own_b[BUILDING_INDEX["proc"]])
        harv_cap = min(MAX_HARVESTERS, max(MIN_HARVESTER_CAP, 2 * own_proc + 2))
        if self._own_u[UNIT_INDEX["harv"]] + self._producing[ITEM_INDEX["harv"]] >= harv_cap:
            mask[ITEM_INDEX["harv"]] = False

        mcv_total = (self._own_b[BUILDING_INDEX["fact"]]
                     + self._own_u[UNIT_INDEX["mcv"]]
                     + self._producing[ITEM_INDEX["mcv"]])
        if mcv_total >= MAX_MCV_TOTAL:
            mask[ITEM_INDEX["mcv"]] = False

        barr_total = (self._own_b[BUILDING_INDEX["tent"]] + self._own_b[BUILDING_INDEX["barr"]]
                      + self._producing[ITEM_INDEX["tent"]] + self._producing[ITEM_INDEX["barr"]])
        if barr_total >= MAX_BARRACKS:
            mask[ITEM_INDEX["tent"]] = False
            mask[ITEM_INDEX["barr"]] = False

        pbox_total = (self._own_b[BUILDING_INDEX["pbox"]] + self._own_b[BUILDING_INDEX["hbox"]]
                      + self._producing[ITEM_INDEX["pbox"]] + self._producing[ITEM_INDEX["hbox"]])
        if pbox_total >= MAX_PILLBOXES:
            mask[ITEM_INDEX["pbox"]] = False
            mask[ITEM_INDEX["hbox"]] = False

        mask[WAIT] = True
        return mask

    def candidates(self, req: pb.DecisionRequest) -> tuple[np.ndarray, np.ndarray]:
        cand = np.zeros((MAX_TARGETS, CAND_DIM), np.float32)
        mask = np.zeros(N_ARMY_ACTIONS, bool)
        mask[HOLD] = mask[KEEP] = True
        rank: dict[str, int] = {}
        visible = [(c, combat_value(t)) for t, _, c in self.enemy_units.values()]
        for k, t in enumerate(req.targets[:MAX_TARGETS]):
            mask[k] = True
            cell = (t.cell_x, t.cell_y)
            row = cand[k]
            if t.kind in KIND_INDEX:
                row[KIND_INDEX[t.kind]] = 1
            o = len(TARGET_KINDS)
            bt = base_type(t.type)
            if bt in BUILDING_INDEX:
                row[o + BUILDING_INDEX[bt]] = 1
            o += len(BUILDINGS)
            r = rank.get(t.kind, 0)
            rank[t.kind] = r + 1
            row[o:o + 8] = [
                COST.get(bt, 0) / 2000,
                _dist(cell, self.base) / 100,
                _dist(cell, self.away_center) / 100 if self.away_center else 1.0,
                r / MAX_TARGETS,
                k / MAX_TARGETS,
                sum(v for c, v in visible if _dist(c, cell) <= NEAR_CELLS) / 5000,
                sum(v for x, y, v in self.own_units if _dist((x, y), cell) <= NEAR_CELLS) / 5000,
                float(self.army.target == (t.kind, t.cell_x, t.cell_y)),
            ]
        return cand, mask

    # ── actions ──────────────────────────────────────────────────────────────

    @staticmethod
    def label(req: pb.DecisionRequest, decision: pb.Decision | None) -> int | None:
        """Network action for a decision; None if it is outside the vocabulary."""
        if req.kind == pb.DECIDE_ARMY:
            if decision is None:
                return KEEP
            if decision.target_index < 0:
                return HOLD
            return decision.target_index if decision.target_index < min(MAX_TARGETS, len(req.targets)) else None
        # A choice the queue does not offer starts nothing (the builtin bot asks for an
        # MCV before it can build one): that is a wait.
        if decision is None or decision.choice not in req.options:
            return WAIT
        return ITEM_INDEX.get(base_type(decision.choice))

    @staticmethod
    def decision(req: pb.DecisionRequest, action: int) -> pb.Decision | None:
        """Inverse of label(): the Decision to send (None = leave the ARMY request unanswered)."""
        if req.kind == pb.DECIDE_ARMY:
            if action == KEEP:
                return None
            return pb.Decision(request_id=req.id, target_index=-1 if action == HOLD else action)
        if action == WAIT:
            return pb.Decision(request_id=req.id, choice="")
        name = ITEMS[action]
        choice = next((o for o in req.options if base_type(o) == name), name)
        return pb.Decision(request_id=req.id, choice=choice)

    def record(self, req: pb.DecisionRequest, decision: pb.Decision | None) -> None:
        """Remember an executed decision (Building waits + ARMY orders feed memory)."""
        if req.kind == pb.DECIDE_BUILD and req.queue == "Building":
            if decision is None or not decision.choice:
                self.consec_build_waits += 1
            else:
                self.consec_build_waits = 0
                self.last_build_tick = self.tick
            return

        if req.kind != pb.DECIDE_ARMY:
            return
        a = self.army
        if decision is None:
            a.last = 0
            if a.attacking:
                a.aim_age += 1
            return
        a.order_tick = self.tick
        if decision.target_index < 0:
            a.attacking, a.target, a.last = False, None, 1
            return
        t = req.targets[decision.target_index]
        new = (t.kind, t.cell_x, t.cell_y)
        if not a.attacking:
            a.attacking, a.attack_tick, a.peak = True, self.tick, self.army_value
        a.aim_age = a.aim_age + 1 if new == a.target else 0
        a.target, a.last = new, 2
