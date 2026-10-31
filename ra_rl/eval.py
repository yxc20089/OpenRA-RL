"""Standard evaluation: N games with fixed seeds, one session per daemon.

    python -m ra_rl.eval --policy scripted --bot hard --games 100 \
        --daemons localhost:9999,localhost:10000,localhost:10001,localhost:10002,localhost:10003,localhost:10004

Win rate counts only real wins; truncated games (max ticks) are not wins.
Daemon containers are automatically recycled every --recycle-every games and
retried once on transient container OOM / poisoned sessions so evaluations are clean.
"""

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import grpc

from ra_env import Bridge, GameSession, MatchConfig, daemons as daemon_ctl
from ra_rl.policies import make_policy

DEFAULT_DAEMONS = ",".join(f"localhost:{p}" for p in range(9999, 10005))


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def play_game(bridge: Bridge, policy_name: str, match: MatchConfig, seed: int, step_ticks: int) -> dict:
    policy = make_policy(policy_name)
    session = GameSession(bridge, MatchConfig(**{**match.__dict__, "brain": policy.brain}))
    t0 = time.perf_counter()
    steps = decisions = teacher = 0
    try:
        policy.reset(seed)
        step = session.reset(seed)
        while not (step.done or step.truncated):
            answer = policy.decide(step)
            decisions += len(answer)
            step = session.step(answer, step_ticks)
            steps += 1
            teacher += len(step.teacher)
        outcome, error = (step.result or "truncated"), ""
    except (grpc.RpcError, RuntimeError) as e:
        outcome, error = "error", repr(e)
    finally:
        session.close()
    return {
        "seed": seed, "outcome": outcome, "ticks": session.tick, "steps": steps,
        "decisions": decisions, "teacher_decisions": teacher,
        "wall_s": round(time.perf_counter() - t0, 1), "error": error,
    }


def evaluate(daemons: list[str], policy: str, match: MatchConfig, games: int, seed_base: int,
             step_ticks: int, log=print, recycle_every: int = 50, pre_recycle: bool = False) -> dict:
    lock = threading.Lock()
    next_game = [0]
    results: list[dict] = []

    def worker(addr: str) -> None:
        if pre_recycle:
            daemon_ctl.restart(addr)
        bridge = Bridge(addr)
        played = 0
        try:
            while True:
                with lock:
                    g = next_game[0]
                    next_game[0] += 1
                if g >= games:
                    return
                if recycle_every and played >= recycle_every:
                    bridge.close()
                    daemon_ctl.restart(addr)
                    bridge = Bridge(addr)
                    played = 0

                seed = seed_base + g
                r = play_game(bridge, policy, match, seed, step_ticks)
                played += 1

                if r["outcome"] == "error":
                    # Recover once from container OOM / SessionPoisoned so eval seeds are not dropped.
                    bridge.close()
                    if daemon_ctl.restart(addr):
                        bridge = Bridge(addr)
                        played = 0
                        r = play_game(bridge, policy, match, seed, step_ticks)
                        played += 1
                    else:
                        bridge = Bridge(addr)

                with lock:
                    results.append(r)
                    log(json.dumps(r))
        finally:
            bridge.close()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, len(daemons))) as pool:
        list(pool.map(worker, daemons))

    outcomes: dict[str, int] = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    finished = [r for r in results if r["outcome"] != "error"]
    wins = outcomes.get("win", 0)
    lo, hi = wilson(wins, len(finished))
    return {
        "policy": policy, "bot": match.bot, "map": match.map_name, "games": len(finished),
        "outcomes": outcomes, "win_rate": round(wins / max(1, len(finished)), 3),
        "win_rate_ci95": [round(lo, 3), round(hi, 3)],
        "mean_ticks": round(sum(r["ticks"] for r in finished) / max(1, len(finished))),
        "mean_decisions": round(sum(r["decisions"] for r in finished) / max(1, len(finished)), 1),
        "mean_teacher_decisions": round(sum(r["teacher_decisions"] for r in finished) / max(1, len(finished)), 1),
        "wall_s": round(time.perf_counter() - t0, 1),
        "errors": [r["error"] for r in results if r["error"]][:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="scripted",
                    help="builtin | scripted | random | net:<path> | net-sample:<path> | net-t0.25:<path>")
    ap.add_argument("--daemons", default=DEFAULT_DAEMONS)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--bot", default="hard")
    ap.add_argument("--map", default=MatchConfig.map_name)
    ap.add_argument("--max-ticks", type=int, default=54000, help="truncate games after this (~36 min)")
    ap.add_argument("--step-ticks", type=int, default=500, help="max ticks per step")
    ap.add_argument("--seed-base", type=int, default=10000)
    ap.add_argument("--recycle-every", type=int, default=50, help="restart each daemon every N games (0=never)")
    ap.add_argument("--pre-recycle", action="store_true", help="restart daemons before starting evaluation")
    ap.add_argument("--out", default="", help="append per-game lines and the summary to this jsonl")
    args = ap.parse_args()

    def log(line: str) -> None:
        print(line, flush=True)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    match = MatchConfig(map_name=args.map, bot=args.bot, max_ticks=args.max_ticks)
    daemons = [d.strip() for d in args.daemons.split(",") if d.strip()]
    summary = evaluate(
        daemons, args.policy, match, args.games, args.seed_base, args.step_ticks, log,
        recycle_every=args.recycle_every, pre_recycle=args.pre_recycle,
    )
    log(json.dumps({"summary": summary}))


if __name__ == "__main__":
    main()
