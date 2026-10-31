"""Check that a seed reproduces the same game.

Plays the same seed twice (and one control seed) with the probe policy and
compares a state digest after every decision. Exits 1 on divergence.

Games run one after another by default: that is the supported mode (one
session per daemon). --concurrent runs the three on the same daemon at once,
which currently diverges because sessions share engine state.

    python -m ra_env.determinism --daemon localhost:9999 --seed 123 --ticks 12000
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

from ra_env.bridge import Bridge
from ra_env.probe import deploy_mcv_commands, obs_digest
from ra_env.session import GameSession, MatchConfig


def trace(bridge: Bridge, match: MatchConfig, seed: int, decision_ticks: int) -> list[tuple[int, str]]:
    """(tick, state digest without tick) after every decision."""
    session = GameSession(bridge, match)
    try:
        step = session.reset(seed)
        out = [(step.tick, obs_digest(step.obs, include_tick=False))]
        commands = deploy_mcv_commands(step.obs)
        while not (step.done or step.truncated):
            step = session.step([], decision_ticks, commands)
            out.append((step.tick, obs_digest(step.obs, include_tick=False)))
            commands = []
        return out
    finally:
        session.close()


def describe(a: list[tuple[int, str]], b: list[tuple[int, str]]) -> str:
    state = first_divergence([d for _, d in a], [d for _, d in b])
    ticks = first_divergence([t for t, _ in a], [t for t, _ in b])
    if state < 0 and ticks < 0:
        return "identical"
    if state < 0:
        return f"same states, reported tick differs at decision {ticks} ({a[ticks][0]} vs {b[ticks][0]})"
    return f"DIVERGES at decision {state} (tick {a[state][0]} vs {b[state][0]})"


def first_divergence(a: list[str], b: list[str]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daemon", default="localhost:9999")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--ticks", type=int, default=12000, help="game length to compare")
    ap.add_argument("--decision-ticks", type=int, default=100)
    ap.add_argument("--map", default=MatchConfig.map_name)
    ap.add_argument("--bot", default=MatchConfig.bot)
    ap.add_argument("--brain", default="builtin")
    ap.add_argument("--concurrent", action="store_true", help="run the three games at once on one daemon")
    args = ap.parse_args()

    bridge = Bridge(args.daemon)
    match = MatchConfig(map_name=args.map, bot=args.bot, max_ticks=args.ticks, brain=args.brain)
    seeds = [args.seed, args.seed, args.seed + 1]
    play = lambda s: trace(bridge, match, s, args.decision_ticks)
    if args.concurrent:
        with ThreadPoolExecutor(max_workers=3) as pool:
            a, b, control = pool.map(play, seeds)
    else:
        a, b, control = map(play, seeds)
    bridge.close()

    same = describe(a, b)
    ctrl = describe(a, control)
    print(f"seed {args.seed} vs itself: {same}  [{len(a)} decisions, final tick {a[-1][0]}]")
    print(f"seed {args.seed} vs {args.seed + 1} (control): "
          + ("identical (seed has no effect!)" if ctrl == "identical" else "differs (ok)"))
    return 0 if same == "identical" and ctrl != "identical" else 1


if __name__ == "__main__":
    sys.exit(main())
