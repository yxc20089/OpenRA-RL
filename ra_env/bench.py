"""Throughput benchmark: how many games per hour can this machine simulate?

Runs full games (probe policy vs a scripted bot) on one or more daemons, for
several decision intervals, and reports aggregate ticks/s and games/h.

    python -m ra_env.bench --daemons localhost:9999,localhost:10000 --sessions 4 \
        --decision-ticks 25,100,250 --games 16 --bot hard
"""

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import grpc

from ra_env.bridge import Bridge
from ra_env.probe import deploy_mcv_commands
from ra_env.session import GameSession, MatchConfig

REF_GAME_TICKS = 20000  # games/h is also reported normalised to this length


def play_one(bridge: Bridge, match: MatchConfig, seed: int, decision_ticks: int) -> dict:
    session = GameSession(bridge, match)
    t0 = time.perf_counter()
    rpc_s = 0.0
    decisions = 0
    try:
        step = session.reset(seed)
        commands = deploy_mcv_commands(step.obs)
        while not (step.done or step.truncated):
            t_rpc = time.perf_counter()
            step = session.step([], decision_ticks, commands)
            rpc_s += time.perf_counter() - t_rpc
            decisions += 1
            commands = []
        outcome = step.result or "truncated"
        ticks = step.tick
        error = ""
    except (grpc.RpcError, RuntimeError) as e:  # keep benchmarking the rest; report failures
        outcome, ticks, error = "error", session.tick, repr(e)
    finally:
        session.close()
    return {
        "seed": seed, "ticks": ticks, "decisions": decisions, "outcome": outcome,
        "wall_s": time.perf_counter() - t0, "rpc_s": rpc_s, "error": error,
    }


def run_config(bridges: list[Bridge], sessions: int, match: MatchConfig,
               decision_ticks: int, games: int, seed_base: int) -> dict:
    workers = len(bridges) * sessions
    lock = threading.Lock()
    next_game = [0]
    results: list[dict] = []

    def worker(i: int) -> None:
        bridge = bridges[i % len(bridges)]
        while True:
            with lock:
                g = next_game[0]
                next_game[0] += 1
            if g >= games:
                return
            r = play_one(bridge, match, seed_base + g, decision_ticks)
            with lock:
                results.append(r)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, range(workers)))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["outcome"] != "error"]
    ticks = sum(r["ticks"] for r in ok)
    decisions = sum(r["decisions"] for r in ok)
    outcomes: dict[str, int] = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    return {
        "decision_ticks": decision_ticks,
        "daemons": len(bridges),
        "sessions_per_daemon": sessions,
        "games": len(results),
        "errors": [r["error"] for r in results if r["error"]],
        "outcomes": outcomes,
        "wall_s": round(wall, 1),
        "ticks_per_s": round(ticks / wall, 1),
        "decisions_per_s": round(decisions / wall, 1),
        "mean_game_ticks": round(statistics.mean(r["ticks"] for r in ok)) if ok else 0,
        "rpc_ms_mean": round(1000 * sum(r["rpc_s"] for r in ok) / max(1, decisions), 1),
        "games_per_h": round(3600 * len(ok) / wall, 1),
        "games_per_h_at_ref": round(3600 * ticks / wall / REF_GAME_TICKS, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daemons", default="localhost:9999", help="comma-separated host:port list")
    ap.add_argument("--sessions", type=int, default=4, help="concurrent sessions per daemon")
    ap.add_argument("--decision-ticks", default="25,100,250", help="comma-separated intervals to compare")
    ap.add_argument("--games", type=int, default=8, help="games per decision interval")
    ap.add_argument("--map", default=MatchConfig.map_name)
    ap.add_argument("--bot", default=MatchConfig.bot)
    ap.add_argument("--brain", default="builtin", help="agent commander brain (builtin plays a full game)")
    ap.add_argument("--max-ticks", type=int, default=MatchConfig.max_ticks)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--out", default="", help="append one JSON line per config to this file")
    args = ap.parse_args()

    bridges = [Bridge(a.strip()) for a in args.daemons.split(",") if a.strip()]
    match = MatchConfig(map_name=args.map, bot=args.bot, max_ticks=args.max_ticks, brain=args.brain)
    print(f"bench: {len(bridges)} daemon(s) x {args.sessions} sessions, map={match.map_name}, "
          f"bot={match.bot}, brain={match.brain}, max_ticks={match.max_ticks}", flush=True)
    for dt in (int(x) for x in args.decision_ticks.split(",")):
        row = run_config(bridges, args.sessions, match, dt, args.games, args.seed_base)
        print(json.dumps(row), flush=True)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    for b in bridges:
        b.close()


if __name__ == "__main__":
    main()
