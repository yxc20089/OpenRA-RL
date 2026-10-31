"""Hand-written multi-archetype commander policy (Phase 1 / Phase 2.5 teacher).

Allied plan with fog-honest reactive adaptations against rush (`brutal`), macro
(`hard`), and fortified (`turtle`) opponents:
  - Economy first, power kept ahead of demand, extra factories when cash banks up.
  - Anti-rush: if early enemy combat forces enter our base perimeter, unlock
    emergency pillbox/turret defense (once >=1 refinery exists), keep infantry
    training active, and avoid marching defenders away while our yard is contested.
  - Anti-turtle / anti-air: remember spotted static defenses (`tsla`/`gun`/`ftur`/
    `pbox`) and enemy aircraft; shift vehicle production toward siege artillery
    (`1 arty : 2 tanks`), add AA guns (`agun`), and scale push threshold upward
    after recalled pushes instead of trickling small squads into Tesla coils.
  - Only the agent's own fog-of-war observation is used (never `global_summary`).

ARMY contract: -1 = hold (recall), k = attack targets[k], no answer = keep going.
"""

from collections import Counter
from dataclasses import dataclass

from ra_env import StepResult, pb
from ra_rl.vocab import base_type

# Structures in the order we want them, with how many of each by then.
BUILD_ORDER = [
    ("powr", 1), ("tent", 1), ("proc", 1), ("proc", 2), ("powr", 2), ("weap", 1),
    ("apwr", 1), ("dome", 1), ("proc", 3), ("fix", 1), ("apwr", 2), ("weap", 2),
    ("atek", 1), ("apwr", 3), ("proc", 4),
]
DEFENSES = [("pbox", 2), ("gun", 3)]
RUSH_DEFENSES = [("pbox", 2), ("gun", 1)]
POWER = {"powr": 100, "apwr": 200}

STATIC_DEFENSE_TYPES = frozenset({"pbox", "hbox", "gun", "ftur", "tsla"})
AIR_TYPES = frozenset({"heli", "hind", "mh60", "yak", "mig"})
AIR_BUILDING_TYPES = frozenset({"hpad", "afld"})

COMBAT_COST = {
    "e1": 100, "e2": 150, "e3": 300, "e4": 300, "e7": 1200, "shok": 350, "medi": 200,
    "jeep": 600, "1tnk": 700, "2tnk": 850, "3tnk": 1150, "4tnk": 2000,
    "arty": 600, "v2rl": 900, "apc": 800, "ctnk": 1350, "stnk": 1350, "ttnk": 1350,
    "ftrk": 600, "mgg": 1200, "heli": 1200, "hind": 1500, "mh60": 1000, "yak": 1350, "mig": 2000,
}
HOME_RADIUS = 20  # cells around the construction yard that count as "home"


@dataclass
class Params:
    power_margin: int = 40
    rich_bank: int = 3000        # with this much banked, add factories
    max_weap: int = 3
    max_tent: int = 2
    max_infantry: int = 40
    infantry_bank: int = 0       # train infantry only with this much banked
    rifles_per_rocket: int = 2
    harv_per_proc: int = 2
    max_harv: int = 8
    arty_every: int = 4          # 1 artillery per this many tanks (standard)
    siege_arty_every: int = 2    # 1 artillery per this many tanks vs fortified turtle
    siege_def_threshold: int = 3 # spotted static defenses that trigger siege mode
    attack_value: int = 6000     # push when the army is worth this much
    siege_step_value: int = 1500 # extra army value required per recalled siege push
    max_attack_value: int = 9500 # cap on scaled siege push threshold
    reinforce_value: int = 2000  # while attacking, send the home group once it is worth this
    recall_fraction: float = 0.35  # recall when the army falls below this share of its peak
    max_aim_decisions: int = 8   # give up on a target still listed after this many army decisions
    rush_threat_threshold: int = 600  # enemy combat cost in home radius to trigger anti-rush


def counts(obs: pb.GameObservation) -> tuple[Counter, Counter]:
    return Counter(base_type(b.type) for b in obs.buildings), Counter(base_type(u.type) for u in obs.units)


def army_value(units: Counter) -> int:
    return sum(COMBAT_COST.get(t, 0) * n for t, n in units.items())


def bank(obs: pb.GameObservation) -> int:
    return obs.economy.cash + obs.economy.ore


def base_center(obs: pb.GameObservation) -> tuple[int, int] | None:
    yards = [(b.cell_x, b.cell_y) for b in obs.buildings if base_type(b.type) == "fact"]
    if yards:
        return yards[0]
    if obs.buildings:
        bx = int(sum(b.cell_x for b in obs.buildings) / len(obs.buildings))
        by = int(sum(b.cell_y for b in obs.buildings) / len(obs.buildings))
        return bx, by
    return None


def home_army_value(obs: pb.GameObservation) -> int:
    yards = [(b.cell_x, b.cell_y) for b in obs.buildings if base_type(b.type) == "fact"]
    if not yards:
        return 0
    hx, hy = yards[0]
    return sum(COMBAT_COST.get(base_type(u.type), 0) for u in obs.units
               if (u.cell_x - hx) ** 2 + (u.cell_y - hy) ** 2 <= HOME_RADIUS ** 2)


def home_threat_value(obs: pb.GameObservation) -> int:
    center = base_center(obs)
    if center is None:
        return 0
    hx, hy = center
    return sum(COMBAT_COST.get(base_type(u.type), 0) for u in obs.visible_enemies
               if (u.cell_x - hx) ** 2 + (u.cell_y - hy) ** 2 <= HOME_RADIUS ** 2)


class ScriptedPolicy:
    brain = "external"

    def __init__(self, params: Params | None = None):
        self.p = params or Params()
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.attacking = False
        self.peak = 0
        self.target: tuple[str, int, int] | None = None
        self.aim_age = 0
        self.skipped: set[tuple[str, int, int]] = set()
        self.recalls = 0
        self.seen_defenses: set[int] = set()
        self.seen_air = False
        self.rush_detected = False

    def _observe_intel(self, obs: pb.GameObservation, have: Counter) -> int:
        for b in obs.visible_enemy_buildings:
            bt = base_type(b.type)
            if bt in STATIC_DEFENSE_TYPES:
                self.seen_defenses.add(b.actor_id)
            elif bt in AIR_BUILDING_TYPES:
                self.seen_air = True
        for u in obs.visible_enemies:
            if base_type(u.type) in AIR_TYPES:
                self.seen_air = True
        threat = home_threat_value(obs)
        if threat >= self.p.rush_threat_threshold and (obs.tick <= 5000 or have["weap"] == 0):
            self.rush_detected = True
        return threat

    def decide(self, step: StepResult) -> list[pb.Decision]:
        buildings, units = counts(step.obs)
        threat = self._observe_intel(step.obs, buildings)
        out = []
        for req in step.pending:
            if req.kind == pb.DECIDE_BUILD:
                out.append(pb.Decision(request_id=req.id, choice=self.build(req, step.obs, buildings, threat)))
            elif req.kind == pb.DECIDE_TRAIN:
                out.append(pb.Decision(request_id=req.id, choice=self.train(req, step.obs, buildings, units, threat)))
            elif req.kind == pb.DECIDE_ARMY:
                index = self.army(req, step.obs, units, threat)
                if index is not None:
                    out.append(pb.Decision(request_id=req.id, target_index=index))
        return out

    def army_state(self) -> tuple:
        return self.attacking, self.peak, self.target, self.aim_age, set(self.skipped), self.recalls

    def sync_army(self, before: tuple, req: pb.DecisionRequest, index: int | None, obs: pb.GameObservation) -> None:
        """Another policy executed `index` instead of our answer (noise, DAgger):
        rewind the army state to `before` and apply what was really ordered."""
        self.attacking, self.peak, self.target, self.aim_age, self.skipped, self.recalls = before
        if index is None:
            if self.attacking:
                self.aim_age += 1
            return
        if index < 0:
            if self.attacking:
                self.recalls += 1
            self.attacking, self.target = False, None
            return
        if not self.attacking:
            self.attacking, self.peak = True, army_value(counts(obs)[1])
        t = req.targets[index]
        new = (t.kind, t.cell_x, t.cell_y)
        self.aim_age = self.aim_age + 1 if new == self.target else 0
        self.target = new

    def power_item(self, options: set[str]) -> str:
        return "apwr" if "apwr" in options else ("powr" if "powr" in options else "")

    def build(self, req: pb.DecisionRequest, obs: pb.GameObservation, have: Counter, threat: int = 0) -> str:
        options = {base_type(o) for o in req.options}
        raw_map = {base_type(o): o for o in req.options}

        def pick(item: str) -> str:
            return raw_map.get(item, "")

        if req.queue == "Defense":
            if have["weap"] == 0:
                if self.rush_detected and have["proc"] >= 1:
                    for item, n in RUSH_DEFENSES:
                        if item in options and have[item] < n:
                            return pick(item)
                return ""
            for item, n in DEFENSES:
                if item in options and have[item] < n:
                    return pick(item)
            if self.seen_air and "agun" in options and have["agun"] < 2:
                return pick("agun")
            return ""

        spare = obs.economy.power_provided - obs.economy.power_drained
        if spare < self.p.power_margin and have["fact"]:
            p_item = self.power_item(options)
            return pick(p_item) if p_item else ""
        for item, n in BUILD_ORDER:
            if have[item] < n and item in options:
                return pick(item)
        if bank(obs) >= self.p.rich_bank:
            if "weap" in options and have["weap"] < self.p.max_weap:
                return pick("weap")
            if "tent" in options and have["tent"] < self.p.max_tent:
                return pick("tent")
        return ""

    def train(self, req: pb.DecisionRequest, obs: pb.GameObservation, have: Counter, units: Counter,
              threat: int = 0) -> str:
        options = {base_type(o) for o in req.options}
        raw_map = {base_type(o): o for o in req.options}

        def pick(item: str) -> str:
            return raw_map.get(item, "")

        if req.queue == "Infantry":
            infantry = units["e1"] + units["e3"]
            under_attack = threat >= self.p.rush_threat_threshold
            if infantry >= self.p.max_infantry:
                return ""
            if not under_attack and bank(obs) < self.p.infantry_bank and have["weap"]:
                return ""
            ratio = 1 if self.seen_air else self.p.rifles_per_rocket
            if "e3" in options and units["e3"] * ratio < units["e1"]:
                return pick("e3")
            return pick("e1") if "e1" in options else ""

        if req.queue == "Vehicle":
            if "harv" in options and units["harv"] < min(self.p.max_harv, self.p.harv_per_proc * have["proc"]):
                return pick("harv")
            tanks = units["2tnk"] + units["1tnk"]
            siege_mode = len(self.seen_defenses) >= self.p.siege_def_threshold
            arty_ratio = self.p.siege_arty_every if siege_mode else self.p.arty_every
            if "arty" in options and tanks >= arty_ratio * (units["arty"] + 1):
                return pick("arty")
            for tank in ("2tnk", "1tnk"):
                if tank in options:
                    return pick(tank)
            return ""
        return ""  # aircraft / naval: not part of this plan

    def army(self, req: pb.DecisionRequest, obs: pb.GameObservation, units: Counter, threat: int = 0) -> int | None:
        value = army_value(units)
        if not req.targets:
            return None
        if not self.attacking:
            push_threshold = self.p.attack_value
            if len(self.seen_defenses) >= self.p.siege_def_threshold and self.recalls > 0:
                push_threshold = min(
                    self.p.max_attack_value,
                    self.p.attack_value + self.p.siege_step_value * min(3, self.recalls),
                )
            if threat >= 1500:
                push_threshold = max(push_threshold, int(1.3 * threat))
            if value >= push_threshold:
                self.attacking, self.peak = True, value
                return self.aim(req)
            return None
        self.peak = max(self.peak, value)
        if value < self.p.recall_fraction * self.peak:
            self.attacking, self.target = False, None
            self.recalls += 1
            return -1
        self.aim_age += 1
        if self.target not in {(t.kind, t.cell_x, t.cell_y) for t in req.targets}:
            return self.aim(req)  # target destroyed / area explored: next one
        if self.aim_age >= self.p.max_aim_decisions:
            self.skipped.add(self.target)  # stuck (unreachable, or nothing there): move on
            return self.aim(req)
        if home_army_value(obs) >= self.p.reinforce_value:
            return self.aim(req)  # fold the home group into the push
        return None  # keep going

    def aim(self, req: pb.DecisionRequest) -> int:
        i = self.pick_target(req, self.skipped)
        if i is None:
            # Every listed target was previously skipped in a long game; reset and re-sweep.
            self.skipped.clear()
            i = self.pick_target(req, self.skipped) or 0
        t = req.targets[i]
        new = (t.kind, t.cell_x, t.cell_y)
        if new != self.target:
            self.aim_age = 0
        self.target = new
        return i

    @staticmethod
    def pick_target(req: pb.DecisionRequest, skipped: set = frozenset()) -> int | None:
        # Targets come sorted nearest-first within each kind: known buildings, then the
        # enemy spawn, then unexplored and stale (not seen lately) sectors where an expansion may hide.
        for kind in ("enemy_building", "enemy_spawn", "unexplored", "stale"):
            for i, t in enumerate(req.targets):
                if t.kind == kind and (t.kind, t.cell_x, t.cell_y) not in skipped:
                    return i
        return 0 if not skipped else None
