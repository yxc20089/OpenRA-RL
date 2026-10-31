"""One game (episode) against a scripted OpenRA bot, in lock-step.

The game only advances inside step(): the agent decides, the engine runs up to
`ticks` ticks headless, the observation comes back. No wall-clock pacing.

The agent slot is a commander (v3): the normal AI's bot modules execute, and the
strategic decisions (BUILD / TRAIN / ARMY) come from
  - brain "external": the agent. step() returns as soon as new decisions are
    pending; answer them in the next step() (unanswered = wait).
  - brain "builtin": the modules' own heuristics. step() returns as soon as they
    decide something and reports it in `teacher` (imitation data), with the
    state an external agent would have seen; the choice starts next step.
"""

from dataclasses import dataclass, field

from ra_env._proto import rl_bridge_pb2 as pb
from ra_env.bridge import Bridge, SessionPoisoned

# Friendly difficulty -> OpenRA bot type (mods/ra/rules/ai.yaml).
BOT_TYPES = {
    "beginner": "beginner",
    "easy": "easy",
    "medium": "medium",
    "hard": "normal",
    "brutal": "rush",
    "normal": "normal",
    "rush": "rush",
    "turtle": "turtle",
    "naval": "naval",
}

BRAINS = ("external", "builtin")

AGENT_SLOT = "Multi1"
BOT_SLOT = "Multi0"


@dataclass(frozen=True)
class MatchConfig:
    map_name: str = "singles.oramap"
    bot: str = "hard"
    player_faction: str = "RandomAllies"
    enemy_faction: str = "Random"
    max_ticks: int = 30000  # truncation cap (~20 min of game time at 25 t/s)
    brain: str = "external"

    def bots_spec(self) -> str:
        bot_type = BOT_TYPES.get(self.bot)
        if bot_type is None:
            raise ValueError(f"unknown bot {self.bot!r}; expected one of {sorted(BOT_TYPES)}")
        return f"{AGENT_SLOT}:rl-agent,{BOT_SLOT}:{bot_type}"


@dataclass
class StepResult:
    obs: pb.GameObservation
    tick: int
    done: bool        # game over (win / lose / draw)
    truncated: bool   # hit match.max_ticks without a result
    result: str       # "win" | "lose" | "draw" | ""
    pending: list[pb.DecisionRequest] = field(default_factory=list)   # answer these next step
    teacher: list[pb.TeacherDecision] = field(default_factory=list)   # brain "builtin" only


class GameSession:
    def __init__(self, bridge: Bridge, match: MatchConfig):
        if match.brain not in BRAINS:
            raise ValueError(f"unknown brain {match.brain!r}; expected one of {BRAINS}")
        self.bridge = bridge
        self.match = match
        self.session_id = ""
        self.tick = 0

    def reset(self, seed: int) -> StepResult:
        """Start a fresh game. seed=0 asks the engine for a random seed."""
        self.close()
        self.session_id = self.bridge.create_session(
            map_name=self.match.map_name,
            bots=self.match.bots_spec(),
            seed=seed,
            player_faction=self.match.player_faction,
            enemy_faction=self.match.enemy_faction,
            agent_brain=self.match.brain,
        )
        if not self.bridge.wait_ready(self.session_id):
            self.close()
            raise RuntimeError(f"session on {self.bridge.address} never became ready")
        self.tick = 0
        return self.step([], ticks=1)

    def step(self, decisions: list[pb.Decision], ticks: int,
             commands: list[pb.Command] | None = None) -> StepResult:
        """Answer pending decisions and advance up to `ticks` ticks.

        `commands` are raw unit orders, kept only for probes and debugging.
        """
        if not self.session_id:
            raise RuntimeError("step() before reset()")
        ticks = max(1, min(ticks, self.match.max_ticks - self.tick))
        try:
            obs = self.bridge.fast_advance(self.session_id, ticks, decisions, commands)
        except SessionPoisoned:
            self.session_id = ""  # the daemon already destroyed it
            raise
        self.tick = obs.tick
        done = bool(obs.done)
        return StepResult(
            obs=obs,
            tick=obs.tick,
            done=done,
            truncated=not done and obs.tick >= self.match.max_ticks,
            result=obs.result if done else "",
            pending=list(obs.pending_decisions),
            teacher=list(obs.teacher_decisions),
        )

    def close(self) -> None:
        if self.session_id:
            self.bridge.destroy_session(self.session_id)
            self.session_id = ""
