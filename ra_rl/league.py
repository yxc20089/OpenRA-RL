r"""Phase 4 v2: PFSP Multi-Bot League Training for the Commander Network.

    python -m ra_rl.league --init runs/phase2_5/bc_s3/model.pt \
        --ref runs/phase2_5/bc_s3/model.pt --out runs/phase4/league_v2

Key safeguards against multi-opponent collapse & composition drift:
  - Reuses Phase 3 v2 SMDP joint-step GAE (continuous tick discounting + exact PBRS)
    and decoupled Asymmetric Critic V(g, priv) conditioned on true opponent archetype.
  - Stratified per-opponent advantage normalization (adv normalized within each bot type,
    preventing easy wins vs one archetype from biasing gradients against another).
  - Prioritized Fictitious Self-Play (PFSP) dynamic opponent weighting: automatically
    allocates more rollout games to opponents with lower EMA win rate while keeping a
    minimum sampling floor per archetype.
  - Masked KL anchor (--kl-coef) toward the multi-bot BC reference teacher (--ref).
  - Clean multi-bot evaluation (hard, brutal, turtle) with automatic daemon recycling.
"""

import argparse
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    SummaryWriter = None
    HAS_TB = False

from ra_env import Bridge, MatchConfig, daemons
from ra_rl.eval import DEFAULT_DAEMONS, evaluate
from ra_rl.model import CommanderNet
from ra_rl.ppo import (
    ARMY_KEYS,
    PROD_KEYS,
    STEP_KEYS,
    collect_game,
    compute_behavior_stats,
    normalize_advantages,
    pack,
    policy_spec,
    ppo_update,
    to_torch_dict,
)


def compute_harmonic_score(*wrs: float) -> float:
    """Generalized harmonic mean over >=1 win rates; collapses to 0 if any is 0."""
    if not wrs:
        return 0.0
    if any(w <= 0.0 for w in wrs):
        return 0.0
    denom = sum(1.0 / max(1e-8, float(w)) for w in wrs)
    return round(len(wrs) / denom, 3)


def pfsp_probabilities(
    ema_wrs: dict[str, float],
    opponents: list[str],
    power: float = 1.5,
    min_prob: float = 0.18,
) -> np.ndarray:
    """PFSP opponent distribution focusing compute on lower-win-rate archetypes."""
    n = len(opponents)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.ones((1,), dtype=np.float64)

    floor = min(min_prob, 1.0 / n)
    raw = np.array([max(0.05, 1.0 - float(ema_wrs.get(o, 0.5))) ** power for o in opponents], dtype=np.float64)
    raw /= raw.sum()
    probs = floor + (1.0 - n * floor) * raw
    return probs / probs.sum()


def assign_opponents(opponents: list[str], probs: np.ndarray, games: int, rng: np.random.Generator) -> list[str]:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    counts = rng.multinomial(games, probs)
    bot_assignments: list[str] = []
    for bot_name, count in zip(opponents, counts):
        bot_assignments.extend([bot_name] * int(count))
    while len(bot_assignments) < games:
        bot_assignments.append(opponents[0])
    rng.shuffle(bot_assignments)
    return bot_assignments[:games]


def collect_league_rollouts(
    addresses: list[str],
    net: CommanderNet,
    games: int,
    seed_base: int,
    opponents: list[str],
    probs: np.ndarray,
    map_name: str,
    max_ticks: int,
    gamma: float = 0.995,
    gae_lambda: float = 0.98,
    recycle_every: int = 50,
    step_ticks: int = 500,
    gamma_ticks: float = 250.0,
    pbrs_coef: float = 0.15,
    rng: np.random.Generator | None = None,
) -> tuple[dict, dict, dict, dict]:
    net_cpu = copy.deepcopy(net).cpu().eval()
    rng = rng or np.random.default_rng(seed_base)
    bot_assignments = assign_opponents(opponents, probs, games, rng)

    job_list = [(seed_base + i, bot_assignments[i]) for i in range(games)]
    lock = threading.Lock()
    todo = list(job_list)

    all_prod: dict[str, list] = {k: [] for k in PROD_KEYS}
    all_army: dict[str, list] = {k: [] for k in ARMY_KEYS}
    all_step: dict[str, list] = {k: [] for k in STEP_KEYS}
    metas: list[dict] = []

    def worker(addr: str) -> None:
        bridge = Bridge(addr)
        played = 0
        try:
            while True:
                with lock:
                    if not todo:
                        return
                    seed, bot_name = todo.pop(0)

                if recycle_every and played >= recycle_every:
                    bridge.close()
                    daemons.restart(addr)
                    bridge = Bridge(addr)
                    played = 0

                match = MatchConfig(map_name=map_name, bot=bot_name, max_ticks=max_ticks, brain="external")
                meta, p_data, a_data, s_data = collect_game(
                    bridge, net_cpu, match, seed, gamma, gae_lambda, step_ticks, gamma_ticks, pbrs_coef,
                )
                played += 1

                if meta["error"]:
                    bridge.close()
                    daemons.restart(addr)
                    bridge = Bridge(addr)
                    played = 0
                    meta, p_data, a_data, s_data = collect_game(
                        bridge, net_cpu, match, seed, gamma, gae_lambda, step_ticks, gamma_ticks, pbrs_coef,
                    )
                    played += 1

                with lock:
                    metas.append(meta)
                    if not meta["error"]:
                        for k in PROD_KEYS:
                            all_prod[k].extend(p_data[k])
                        for k in ARMY_KEYS:
                            all_army[k].extend(a_data[k])
                        for k in STEP_KEYS:
                            all_step[k].extend(s_data[k])
        finally:
            bridge.close()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(addresses)) as pool:
        list(pool.map(worker, addresses))

    packed_prod = pack(all_prod, PROD_KEYS)
    packed_army = pack(all_army, ARMY_KEYS)
    packed_step = pack(all_step, STEP_KEYS)

    # Stratified normalization per opponent prevents cross-archetype baseline bias.
    normalize_advantages(packed_prod, per_bot=True)
    normalize_advantages(packed_army, per_bot=True)
    beh = compute_behavior_stats(packed_prod, packed_army, packed_step)

    valid_metas = [m for m in metas if not m["error"]]
    outcomes: dict[str, int] = {}
    bot_stats: dict[str, dict] = {}

    for bot in opponents:
        bot_metas = [m for m in valid_metas if m["bot"] == bot]
        w = sum(1 for m in bot_metas if m["outcome"] == "win")
        bot_stats[bot] = {
            "games": len(bot_metas),
            "wins": w,
            "win_rate": round(w / max(1, len(bot_metas)), 3),
        }

    for m in valid_metas:
        outcomes[m["outcome"]] = outcomes.get(m["outcome"], 0) + 1

    stats = {
        "games_collected": len(valid_metas),
        "games_failed": len(metas) - len(valid_metas),
        "outcomes": outcomes,
        "win_rate": round(outcomes.get("win", 0) / max(1, len(valid_metas)), 3),
        "mean_ticks": round(sum(m["ticks"] for m in valid_metas) / max(1, len(valid_metas))),
        "mean_decisions": round(sum(m["decisions"] for m in valid_metas) / max(1, len(valid_metas)), 1),
        "n_prod": len(packed_prod.get("g", [])),
        "n_army": len(packed_army.get("g", [])),
        "n_steps": len(packed_step.get("g", [])),
        "collect_s": round(time.perf_counter() - t0, 1),
        "bot_stats": bot_stats,
        "opp_probs": {o: round(float(p), 3) for o, p in zip(opponents, probs)},
        **beh,
    }

    return stats, packed_prod, packed_army, packed_step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="runs/phase2_5/bc_s3/model.pt", help="starting checkpoint")
    ap.add_argument("--ref", default="", help="frozen KL anchor checkpoint (defaults to --init)")
    ap.add_argument("--out", required=True, help="output directory for checkpoints and logs")
    ap.add_argument("--iterations", type=int, default=80, help="total league training iterations")
    ap.add_argument("--games-per-iter", type=int, default=48, help="number of games collected per iteration")
    ap.add_argument("--opponents", default="hard,brutal,turtle", help="comma-separated opponent bot types")
    ap.add_argument("--opponent-weights", default="0.4,0.3,0.3", help="static weights when --no-pfsp is set")
    ap.add_argument("--pfsp", action=argparse.BooleanOptionalAction, default=True, help="adaptive PFSP opponent focus")
    ap.add_argument("--pfsp-power", type=float, default=1.5)
    ap.add_argument("--min-opp-prob", type=float, default=0.18)
    ap.add_argument("--eval-bots", default="hard,brutal,turtle", help="bots evaluated every --eval-every iterations")
    ap.add_argument("--map", default=MatchConfig.map_name)
    ap.add_argument("--max-ticks", type=int, default=54000)
    ap.add_argument("--daemons", default=DEFAULT_DAEMONS)
    ap.add_argument("--seed-base", type=int, default=300000, help="starting seed for rollouts (never overlaps with eval)")
    ap.add_argument("--lr", type=float, default=4e-5, help="actor learning rate")
    ap.add_argument("--critic-lr", type=float, default=3e-4, help="decoupled critic learning rate")
    ap.add_argument("--critic-warmup-iters", type=int, default=6, help="critic warmup iterations on multi-bot mix")
    ap.add_argument("--reset-critic", action="store_true", help="re-initialize critic weights on load")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995, help="discount factor per --gamma-ticks ticks")
    ap.add_argument("--gamma-ticks", type=float, default=250.0)
    ap.add_argument("--gae-lambda", type=float, default=0.98)
    ap.add_argument("--pbrs-coef", type=float, default=0.15)
    ap.add_argument("--kl-coef", type=float, default=0.05, help="KL penalty toward --ref teacher")
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--ent-decay", type=float, default=0.995)
    ap.add_argument("--ent-min", type=float, default=0.001)
    ap.add_argument("--army-weight", type=float, default=1.0)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.025)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games-per-bot", type=int, default=50)
    ap.add_argument("--eval-seeds-base", type=int, default=20000)
    ap.add_argument("--eval-temp", type=float, default=0.0, help="evaluation softmax temperature (0.0 = argmax)")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--recycle-every", type=int, default=50)
    ap.add_argument("--tb", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    daemons_list = [d.strip() for d in args.daemons.split(",") if d.strip()]
    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    eval_bots = [o.strip() for o in args.eval_bots.split(",") if o.strip()]
    static_weights = np.array([float(w.strip()) for w in args.opponent_weights.split(",") if w.strip()], dtype=np.float64)
    static_weights /= static_weights.sum()

    writer = SummaryWriter(log_dir=str(out / "tb")) if (args.tb and HAS_TB) else None
    ref_path = args.ref or args.init

    print(json.dumps({
        "msg": "Starting Phase 4 v2 PFSP League Training",
        "init": args.init, "ref": ref_path, "out": str(out),
        "device": device, "daemons": len(daemons_list),
        "opponents": opponents, "eval_bots": eval_bots, "pfsp": args.pfsp,
        "kl_coef": args.kl_coef, "ent_coef": args.ent_coef, "pbrs_coef": args.pbrs_coef,
        "critic_warmup": args.critic_warmup_iters, "reset_critic": args.reset_critic,
        "games_per_iter": args.games_per_iter,
    }), flush=True)

    net = CommanderNet.load(args.init, device=device)
    if args.reset_critic:
        net.reset_critic()
    ref_net = CommanderNet.load(ref_path, device=device).eval() if args.kl_coef > 0.0 else None
    if ref_net is not None:
        ref_net.requires_grad_(False)

    actor_opt = torch.optim.AdamW(list(net.actor_parameters()), lr=args.lr, weight_decay=args.weight_decay)
    critic_opt = torch.optim.AdamW(list(net.critic_parameters()), lr=args.critic_lr, weight_decay=args.weight_decay)

    cur_seed = args.seed_base
    cur_ent = args.ent_coef
    best_harmonic_score = -1.0
    ema_wrs: dict[str, float] = {o: 0.5 for o in opponents}
    rng = np.random.default_rng(args.seed_base)
    log = (out / "metrics.jsonl").open("a", encoding="utf-8")

    t_start = time.perf_counter()

    for it in range(1, args.iterations + 1):
        t_it = time.perf_counter()
        in_warmup = it <= args.critic_warmup_iters

        probs = (
            pfsp_probabilities(ema_wrs, opponents, power=args.pfsp_power, min_prob=args.min_opp_prob)
            if args.pfsp else static_weights
        )

        collect_stats, prod_np, army_np, step_np = collect_league_rollouts(
            addresses=daemons_list,
            net=net,
            games=args.games_per_iter,
            seed_base=cur_seed,
            opponents=opponents,
            probs=probs,
            map_name=args.map,
            max_ticks=args.max_ticks,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            recycle_every=args.recycle_every,
            gamma_ticks=args.gamma_ticks,
            pbrs_coef=args.pbrs_coef,
            rng=rng,
        )
        cur_seed += args.games_per_iter

        for bot, bstat in collect_stats["bot_stats"].items():
            if bstat["games"] > 0:
                ema_wrs[bot] = 0.7 * ema_wrs[bot] + 0.3 * float(bstat["win_rate"])

        prod_t = to_torch_dict(prod_np, device)
        army_t = to_torch_dict(army_np, device)
        step_t = to_torch_dict(step_np, device)

        update_metrics = ppo_update(
            net=net,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            prod=prod_t,
            army=army_t,
            steps=step_t,
            ref_net=ref_net,
            critic_only=in_warmup,
            epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            clip_eps=args.clip_eps,
            ent_coef=cur_ent,
            kl_coef=args.kl_coef,
            army_weight=args.army_weight,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
        )

        if not in_warmup:
            cur_ent = max(args.ent_min, cur_ent * args.ent_decay)

        net.save(str(out / "checkpoint_latest.pt"), iteration=it, stats=collect_stats)
        if args.save_every > 0 and it % args.save_every == 0:
            net.save(str(out / f"checkpoint_iter_{it:04d}.pt"), iteration=it, stats=collect_stats)

        it_record = {
            "iteration": it,
            "ent_coef": round(cur_ent, 5),
            "ema_wrs": {k: round(v, 3) for k, v in ema_wrs.items()},
            "it_s": round(time.perf_counter() - t_it, 1),
            "total_s": round(time.perf_counter() - t_start, 1),
            **collect_stats,
            **update_metrics,
        }

        if args.eval_every > 0 and it % args.eval_every == 0:
            print(f"[Iter {it}] Running League Evaluation (vs {', '.join(eval_bots)})...", flush=True)
            eval_results = {}
            total_eval_errors = 0
            for test_bot in eval_bots:
                eval_match = MatchConfig(map_name=args.map, bot=test_bot, max_ticks=args.max_ticks)
                eval_summary = evaluate(
                    daemons=daemons_list,
                    policy=policy_spec(out / "checkpoint_latest.pt", args.eval_temp),
                    match=eval_match,
                    games=args.eval_games_per_bot,
                    seed_base=args.eval_seeds_base,
                    step_ticks=500,
                    log=lambda _: None,
                    recycle_every=args.recycle_every,
                    pre_recycle=True,
                )
                total_eval_errors += eval_summary["outcomes"].get("error", 0)
                eval_results[test_bot] = {
                    "win_rate": eval_summary["win_rate"],
                    "ci95": eval_summary["win_rate_ci95"],
                    "outcomes": eval_summary["outcomes"],
                }

            harmonic_score = compute_harmonic_score(*(eval_results[b]["win_rate"] for b in eval_bots))
            it_record["eval"] = {
                **eval_results,
                "harmonic_score": harmonic_score,
                "total_errors": total_eval_errors,
            }

            max_allowed_errors = max(3, (len(eval_bots) * args.eval_games_per_bot) // 10)
            if harmonic_score > best_harmonic_score and total_eval_errors <= max_allowed_errors:
                best_harmonic_score = harmonic_score
                net.save(str(out / "checkpoint_best.pt"), iteration=it, eval=it_record["eval"])
                it_record["new_best"] = True

        log.write(json.dumps(it_record) + "\n")
        log.flush()
        print(json.dumps(it_record), flush=True)

        if writer is not None:
            writer.add_scalar("rollout/win_rate_total", collect_stats["win_rate"], it)
            for bot, bstat in collect_stats.get("bot_stats", {}).items():
                writer.add_scalar(f"rollout/win_rate_{bot}", bstat["win_rate"], it)
                writer.add_scalar(f"pfsp/prob_{bot}", collect_stats["opp_probs"].get(bot, 0.0), it)

            writer.add_scalar("rollout/mean_ticks", collect_stats["mean_ticks"], it)
            writer.add_scalar("rollout/mean_decisions", collect_stats["mean_decisions"], it)
            writer.add_scalar("rollout/explained_var", collect_stats["explained_var"], it)
            writer.add_scalar("behavior/veh_harv_pct", collect_stats["beh_veh_harv_pct"], it)
            writer.add_scalar("behavior/veh_2tnk_pct", collect_stats["beh_veh_2tnk_pct"], it)
            writer.add_scalar("behavior/bld_wait_pct", collect_stats["beh_bld_wait_pct"], it)
            writer.add_scalar("behavior/army_atk_pct", collect_stats["beh_army_atk_pct"], it)

            writer.add_scalar("loss/policy_prod", update_metrics["pi_loss_prod"], it)
            writer.add_scalar("loss/policy_army", update_metrics["pi_loss_army"], it)
            writer.add_scalar("loss/value", update_metrics["val_loss"], it)
            writer.add_scalar("policy/entropy_prod", update_metrics["ent_prod"], it)
            writer.add_scalar("policy/entropy_army", update_metrics["ent_army"], it)
            writer.add_scalar("policy/kl_ref_prod", update_metrics["kl_ref_prod"], it)
            writer.add_scalar("policy/kl_ref_army", update_metrics["kl_ref_army"], it)

            if "eval" in it_record:
                ev = it_record["eval"]
                for b in eval_bots:
                    writer.add_scalar(f"eval/win_rate_{b}", ev[b]["win_rate"], it)
                writer.add_scalar("eval/harmonic_score", ev["harmonic_score"], it)

            writer.flush()

    log.close()
    if writer is not None:
        writer.close()
    print(f"League training complete in {time.perf_counter() - t_start:.0f}s. Best harmonic score: {best_harmonic_score}", flush=True)


if __name__ == "__main__":
    main()
