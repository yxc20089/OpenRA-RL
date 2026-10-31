"""Record imitation data: (features, teacher action) for every decision, one .npz per game.

    # the engine's own bot logic as teacher (brain builtin)
    python -m ra_rl.record --teacher builtin --games 2000 --out data/builtin
    # the hand-written policy as teacher, executing a random action 15% of the time
    python -m ra_rl.record --teacher scripted --bots hard,brutal,turtle --eps 0.15 --games 2000 --out data/scripted_v2
    # DAgger: the network plays, the scripted policy labels the states it reaches
    python -m ra_rl.record --teacher scripted --bots hard,brutal,turtle --executor net:runs/phase2_5/bc_s0/model.pt --out data/dagger1_v2

Opponents cycle through --bots by seed. Seeds 10000-10099 (eval) and 20000-20199
(selection) are reserved and refused. Existing game files are skipped, so a run
can be resumed. Daemon memory grows per game, so each daemon's container is
restarted every --recycle-every games (needs the docker CLI).
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import numpy as np

from ra_env import Bridge, GameSession, MatchConfig, daemons, pb
from ra_rl.eval import DEFAULT_DAEMONS
from ra_rl.features import FEATURE_VERSION, WAIT, Features
from ra_rl.policies import make_policy
from ra_rl.policies.scripted import ScriptedPolicy
from ra_rl.vocab import QUEUE_INDEX

EVAL_SEEDS = range(10000, 10100)
SELECT_SEEDS = range(20000, 20200)


class GameBuffer:
    def __init__(self) -> None:
        self.prod: dict[str, list] = {"g": [], "queue": [], "mask": [], "y": []}
        self.army: dict[str, list] = {"g": [], "cand": [], "mask": [], "y": []}

    def add(self, feats: Features, req: pb.DecisionRequest, y: int) -> None:
        if req.kind == pb.DECIDE_ARMY:
            cand, mask = feats.candidates(req)
            if not mask[y]:
                return
            for k, v in (("g", feats.g.copy()), ("cand", cand), ("mask", mask), ("y", y)):
                self.army[k].append(v)
        else:
            mask = feats.item_mask(req)
            if not mask[y]:
                y = WAIT
            for k, v in (("g", feats.g.copy()), ("queue", QUEUE_INDEX.get(req.queue, 0)),
                         ("mask", mask), ("y", y)):
                self.prod[k].append(v)

    def save(self, path: Path, meta: dict) -> None:
        def arr(rows: list, dtype) -> np.ndarray:
            return np.asarray(rows, dtype) if rows else np.zeros((0,), dtype)

        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            prod_g=arr(self.prod["g"], np.float16), prod_queue=arr(self.prod["queue"], np.int8),
            prod_mask=arr(self.prod["mask"], bool), prod_y=arr(self.prod["y"], np.int16),
            army_g=arr(self.army["g"], np.float16), army_cand=arr(self.army["cand"], np.float16),
            army_mask=arr(self.army["mask"], bool), army_y=arr(self.army["y"], np.int16),
            meta=np.asarray(json.dumps(meta)),
        )
        tmp.replace(path)


def record_game(bridge: Bridge, args, seed: int, bot: str) -> dict:
    builtin = args.teacher == "builtin"
    match = MatchConfig(map_name=args.map, bot=bot, max_ticks=args.max_ticks,
                        brain="builtin" if builtin else "external")
    session = GameSession(bridge, match)
    rng = random.Random(seed)
    feats = Features()
    expert = None if builtin else ScriptedPolicy()
    executor = make_policy(args.executor) if args.executor else None
    if executor is not None:
        executor.reset(seed)
    buf = GameBuffer()
    t0 = time.perf_counter()
    try:
        step = session.reset(seed)
        while not (step.done or step.truncated):
            feats.observe(step.obs)
            answer: list[pb.Decision] = []
            if builtin:
                labeled = [(td.request, td.decision) for td in step.teacher]
                for req, d in labeled:
                    y = feats.label(req, d)
                    if y is not None:
                        buf.add(feats, req, y)
                for req, d in labeled:
                    feats.record(req, d)
            else:
                before = expert.army_state()
                labels = {d.request_id: d for d in expert.decide(step)}
                chosen = executor.actions(step) if executor is not None else None
                for i, req in enumerate(step.pending):
                    teacher = labels.get(req.id)
                    y = feats.label(req, teacher)
                    if y is not None:
                        buf.add(feats, req, y)
                    army = req.kind == pb.DECIDE_ARMY
                    a = chosen[i] if chosen is not None else y
                    if rng.random() < (args.eps_army if army else args.eps):
                        mask = feats.candidates(req)[1] if army else feats.item_mask(req)
                        a = rng.choice(np.flatnonzero(mask).tolist())
                    d = teacher if a is None or a == y else feats.decision(req, a)
                    feats.record(req, d)
                    if executor is not None:
                        executor.feats.record(req, d)
                    if army and a != y:
                        expert.sync_army(before, req, None if d is None else d.target_index, step.obs)
                    if d is not None:
                        answer.append(d)
            step = session.step(answer, args.step_ticks)
        outcome, error = (step.result or "truncated"), ""
    except (grpc.RpcError, RuntimeError) as e:
        outcome, error = "error", repr(e)
    finally:
        session.close()

    meta = {
        "seed": seed, "bot": bot, "teacher": args.teacher, "executor": args.executor, "eps": args.eps,
        "eps_army": args.eps_army, "outcome": outcome, "ticks": session.tick,
        "prod_rows": len(buf.prod["y"]), "army_rows": len(buf.army["y"]),
        "feature_version": FEATURE_VERSION,
        "wall_s": round(time.perf_counter() - t0, 1), "error": error,
    }
    if not error:
        buf.save(Path(args.out) / f"g{seed:06d}.npz", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", choices=("builtin", "scripted"), required=True)
    ap.add_argument("--executor", default="", help="policy that plays instead of the teacher (DAgger), e.g. net:<path>")
    ap.add_argument("--eps", type=float, default=0.0, help="chance of a random BUILD/TRAIN action (scripted)")
    ap.add_argument("--eps-army", type=float, default=0.0, help="chance of a random ARMY action (scripted)")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--seed-base", type=int, default=1)
    ap.add_argument("--bots", default="hard,brutal,turtle", help="opponents, cycled by seed")
    ap.add_argument("--daemons", default=DEFAULT_DAEMONS)
    ap.add_argument("--map", default=MatchConfig.map_name)
    ap.add_argument("--max-ticks", type=int, default=54000)
    ap.add_argument("--step-ticks", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recycle-every", type=int, default=60,
                    help="restart each daemon (docker) after this many games; 0 = never")
    args = ap.parse_args()
    if args.teacher == "builtin" and (args.executor or args.eps or args.eps_army):
        ap.error("the builtin teacher plays its own games (no --executor / --eps)")

    seeds = [args.seed_base + i for i in range(args.games)]
    if any(s in EVAL_SEEDS or s in SELECT_SEEDS or s <= 0 for s in seeds):
        ap.error(
            f"seeds must be > 0 and outside reserved eval ({EVAL_SEEDS.start}-{EVAL_SEEDS.stop - 1}) "
            f"and selection ({SELECT_SEEDS.start}-{SELECT_SEEDS.stop - 1}) ranges"
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seeds if not (out / f"g{s:06d}.npz").exists()]
    bots = [b.strip() for b in args.bots.split(",") if b.strip()]
    print(f"{len(todo)} games to record ({len(seeds) - len(todo)} already there)", flush=True)

    lock = threading.Lock()
    log = (out / "games.jsonl").open("a", encoding="utf-8")

    def worker(address: str) -> None:
        bridge = Bridge(address)
        played = 0
        try:
            while True:
                with lock:
                    if not todo:
                        return
                    seed = todo.pop(0)
                if args.recycle_every and played >= args.recycle_every:
                    bridge.close()
                    daemons.restart(address)
                    bridge = Bridge(address)
                    played = 0
                bot_name = bots[seed % len(bots)]
                meta = record_game(bridge, args, seed, bot_name)
                played += 1
                if meta["error"]:
                    # Out of memory, poisoned session, or wedged: restart the daemon and retry once.
                    bridge.close()
                    daemons.restart(address)
                    bridge = Bridge(address)
                    played = 0
                    meta = record_game(bridge, args, seed, bot_name)
                    played += 1
                with lock:
                    log.write(json.dumps(meta) + "\n")
                    log.flush()
                    print(json.dumps(meta), flush=True)
        finally:
            bridge.close()

    addresses = [d.strip() for d in args.daemons.split(",") if d.strip()]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(addresses)) as pool:
        list(pool.map(worker, addresses))
    log.close()
    print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
