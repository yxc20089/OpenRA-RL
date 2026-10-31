"""PPO training for the commander network (Phase 3 v2).

    python -m ra_rl.ppo --init runs/phase2_5/bc_s3/model.pt --out runs/phase3/ppo_v2

Key architectural & algorithmic safeguards:
- Decoupled Asymmetric Critic V(g, priv) (CTDE): separate network trunk reading
  spectator summary + opponent identity during training only; zero gradient
  interference on the actor trunk, plus initial critic warmup (--critic-warmup-iters).
- Joint-Step SMDP GAE: simultaneous decisions at the same tick share a single
  temporal StepNode (no fake intra-tick decay), discounted continuously by elapsed
  game ticks gamma^(dt / gamma_ticks).
- Exact telescoping Potential-Based Reward Shaping (PBRS, Ng et al. 1999) with
  Phi(s_terminal) = 0, preserving optimal-policy invariance.
- Masked KL anchor to frozen BC reference model (--kl-coef), preventing catastrophic
  forgetting and unit-composition drift (harvester/pillbox spam, WAIT/KEEP deadlocks).
- Behavioral telemetry logged every iteration + clean daemon recycling in eval.
"""

import argparse
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import grpc
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    SummaryWriter = None
    HAS_TB = False

from ra_env import Bridge, GameSession, MatchConfig, daemons, pb
from ra_rl.eval import DEFAULT_DAEMONS, evaluate
from ra_rl.features import (
    HOLD,
    KEEP,
    MAX_TARGETS,
    WAIT,
    Features,
    opponent_id,
    privileged_features,
    state_potential,
)
from ra_rl.model import NEG, CommanderNet
from ra_rl.vocab import ITEM_INDEX, QUEUE_INDEX

TERMINAL_REWARDS = {
    "win": 1.0,
    "lose": -1.0,
    "draw": 0.0,
    "truncated": -0.5,
}

PROD_KEYS = ("g", "queue", "mask", "action", "old_lp", "adv", "bot_idx")
ARMY_KEYS = ("g", "cand", "mask", "action", "old_lp", "adv", "bot_idx")
STEP_KEYS = ("g", "priv", "ret", "val", "bot_idx")


def pack(d: dict[str, list], keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    res = {}
    for k in keys:
        if not d.get(k):
            dtype = (
                bool if k == "mask"
                else np.int64 if k in ("queue", "action", "bot_idx")
                else np.float32
            )
            res[k] = np.zeros((0,), dtype=dtype)
        elif k in ("g", "priv", "cand", "adv", "ret", "val", "old_lp"):
            res[k] = np.asarray(d[k], dtype=np.float32)
        elif k in ("queue", "action", "bot_idx"):
            res[k] = np.asarray(d[k], dtype=np.int64)
        elif k == "mask":
            res[k] = np.asarray(d[k], dtype=bool)
    return res


@dataclass
class ProdDecision:
    queue: int
    mask: np.ndarray
    action: int
    log_prob: float


@dataclass
class ArmyDecision:
    cand: np.ndarray
    mask: np.ndarray
    action: int
    log_prob: float


@dataclass
class StepNode:
    """All commander decisions issued simultaneously at a single game tick."""

    tick: int
    g: np.ndarray
    priv: np.ndarray
    val: float
    phi: float = 0.0
    bot_idx: int = 0
    items: list[ProdDecision] = field(default_factory=list)
    armies: list[ArmyDecision] = field(default_factory=list)

    def add_item(self, queue: int, mask: np.ndarray, action: int, log_prob: float) -> None:
        self.items.append(ProdDecision(queue=queue, mask=mask, action=action, log_prob=log_prob))

    def add_army(self, cand: np.ndarray, mask: np.ndarray, action: int, log_prob: float) -> None:
        self.armies.append(ArmyDecision(cand=cand, mask=mask, action=action, log_prob=log_prob))


class GameTrajectory:
    """SMDP trajectory of joint decision ticks with continuous tick discounting + PBRS."""

    def __init__(self) -> None:
        self.steps: list[StepNode] = []

    def begin_step(self, tick: int, g: np.ndarray, priv: np.ndarray, val: float,
                   phi: float = 0.0, bot_idx: int = 0) -> StepNode:
        node = StepNode(tick=tick, g=g.copy(), priv=priv.copy(), val=val, phi=phi, bot_idx=bot_idx)
        self.steps.append(node)
        return node

    def finish(
        self,
        terminal_reward: float,
        end_tick: int,
        gamma: float = 0.995,
        gae_lambda: float = 0.98,
        gamma_ticks: float = 250.0,
        pbrs_coef: float = 0.15,
    ) -> tuple[dict[str, list], dict[str, list], dict[str, list]]:
        prod_data: dict[str, list] = {k: [] for k in PROD_KEYS}
        army_data: dict[str, list] = {k: [] for k in ARMY_KEYS}
        step_data: dict[str, list] = {k: [] for k in STEP_KEYS}

        K = len(self.steps)
        if K == 0:
            return prod_data, army_data, step_data

        ticks = np.array([s.tick for s in self.steps], dtype=np.int64)
        vals = np.array([s.val for s in self.steps], dtype=np.float32)
        phis = np.array([s.phi for s in self.steps], dtype=np.float32)

        advantages = np.zeros(K, dtype=np.float32)
        returns = np.zeros(K, dtype=np.float32)
        tau_ref = max(1.0, float(gamma_ticks))
        last_gae = 0.0

        for k in reversed(range(K)):
            next_tick = max(int(ticks[k]) + 1, int(end_tick)) if k == K - 1 else max(int(ticks[k]) + 1, int(ticks[k + 1]))
            dt = float(next_tick - int(ticks[k]))
            gamma_k = float(gamma ** (dt / tau_ref))

            raw_r = float(terminal_reward) if k == K - 1 else 0.0
            next_phi = 0.0 if k == K - 1 else float(phis[k + 1])  # Phi(s_terminal) == 0 strictly
            shaped_r = raw_r + float(pbrs_coef) * (gamma_k * next_phi - float(phis[k]))

            next_val = 0.0 if k == K - 1 else float(vals[k + 1])
            delta = shaped_r + gamma_k * next_val - float(vals[k])
            last_gae = delta + gamma_k * float(gae_lambda) * last_gae
            advantages[k] = last_gae
            returns[k] = last_gae + float(vals[k])

        for k, node in enumerate(self.steps):
            adv_k = float(advantages[k])
            ret_k = float(returns[k])

            step_data["g"].append(node.g)
            step_data["priv"].append(node.priv)
            step_data["ret"].append(ret_k)
            step_data["val"].append(node.val)
            step_data["bot_idx"].append(node.bot_idx)

            for it in node.items:
                prod_data["g"].append(node.g)
                prod_data["queue"].append(it.queue)
                prod_data["mask"].append(it.mask)
                prod_data["action"].append(it.action)
                prod_data["old_lp"].append(it.log_prob)
                prod_data["adv"].append(adv_k)
                prod_data["bot_idx"].append(node.bot_idx)

            for ar in node.armies:
                army_data["g"].append(node.g)
                army_data["cand"].append(ar.cand)
                army_data["mask"].append(ar.mask)
                army_data["action"].append(ar.action)
                army_data["old_lp"].append(ar.log_prob)
                army_data["adv"].append(adv_k)
                army_data["bot_idx"].append(node.bot_idx)

        return prod_data, army_data, step_data


def normalize_advantages(packed: dict[str, np.ndarray], per_bot: bool = True) -> None:
    """Normalize advantages globally or stratified within each opponent bot_idx."""
    adv = packed.get("adv")
    if adv is None or len(adv) <= 1:
        return
    adv = adv.astype(np.float32, copy=True)
    g_mean = float(adv.mean())
    g_std = float(adv.std()) + 1e-8

    if per_bot and "bot_idx" in packed and len(packed["bot_idx"]) == len(adv):
        bots = packed["bot_idx"]
        for b in np.unique(bots):
            idx = bots == b
            if int(idx.sum()) >= 2:
                sub = adv[idx]
                adv[idx] = (sub - float(sub.mean())) / (float(sub.std()) + 1e-8)
            else:
                adv[idx] = (adv[idx] - g_mean) / g_std
    else:
        adv = (adv - g_mean) / g_std

    packed["adv"] = adv


def masked_kl(
    logits: torch.Tensor,
    ref_logits: torch.Tensor,
    mask: torch.Tensor,
    clip_log_ratio: float = 12.0,
) -> torch.Tensor:
    """Numerically safe KL(pi_theta || pi_ref) over valid masked actions."""
    log_p = F.log_softmax(logits.masked_fill(~mask, NEG), dim=-1)
    log_q = F.log_softmax(ref_logits.masked_fill(~mask, NEG), dim=-1)
    p = log_p.exp().masked_fill(~mask, 0.0)
    diff = torch.clamp(log_p - log_q, min=-clip_log_ratio, max=clip_log_ratio).masked_fill(~mask, 0.0)
    return (p * diff).sum(dim=-1).mean()


def compute_behavior_stats(prod_np: dict[str, np.ndarray], army_np: dict[str, np.ndarray],
                           step_np: dict[str, np.ndarray]) -> dict[str, float]:
    """Extract live action-distribution & critic explained-variance diagnostics."""
    out: dict[str, float] = {}

    if len(step_np.get("ret", [])) > 1:
        ret = step_np["ret"]
        val = step_np["val"]
        var_ret = float(np.var(ret))
        ev = 1.0 - float(np.var(ret - val)) / (var_ret + 1e-8) if var_ret > 1e-8 else 0.0
        out["explained_var"] = round(float(np.clip(ev, -2.0, 1.0)), 4)
    else:
        out["explained_var"] = 0.0

    q_arr = prod_np.get("queue", np.zeros((0,), dtype=np.int64))
    a_arr = prod_np.get("action", np.zeros((0,), dtype=np.int64))

    bld_idx = q_arr == QUEUE_INDEX["Building"]
    out["beh_bld_wait_pct"] = round(100.0 * float((a_arr[bld_idx] == WAIT).mean()), 1) if bld_idx.any() else 0.0

    def_idx = q_arr == QUEUE_INDEX["Defense"]
    pbox_id = ITEM_INDEX["pbox"]
    out["beh_def_pbox_pct"] = round(100.0 * float((a_arr[def_idx] == pbox_id).mean()), 1) if def_idx.any() else 0.0

    veh_idx = q_arr == QUEUE_INDEX["Vehicle"]
    if veh_idx.any():
        veh_a = a_arr[veh_idx]
        out["beh_veh_harv_pct"] = round(100.0 * float((veh_a == ITEM_INDEX["harv"]).mean()), 1)
        out["beh_veh_2tnk_pct"] = round(100.0 * float((veh_a == ITEM_INDEX["2tnk"]).mean()), 1)
        out["beh_veh_arty_pct"] = round(100.0 * float((veh_a == ITEM_INDEX["arty"]).mean()), 1)
    else:
        out["beh_veh_harv_pct"] = 0.0
        out["beh_veh_2tnk_pct"] = 0.0
        out["beh_veh_arty_pct"] = 0.0

    army_a = army_np.get("action", np.zeros((0,), dtype=np.int64))
    if len(army_a) > 0:
        out["beh_army_atk_pct"] = round(100.0 * float((army_a < MAX_TARGETS).mean()), 1)
        out["beh_army_keep_pct"] = round(100.0 * float((army_a == KEEP).mean()), 1)
        out["beh_army_hold_pct"] = round(100.0 * float((army_a == HOLD).mean()), 1)
    else:
        out["beh_army_atk_pct"] = 0.0
        out["beh_army_keep_pct"] = 0.0
        out["beh_army_hold_pct"] = 0.0

    return out


def collect_game(
    bridge: Bridge,
    net: CommanderNet,
    match: MatchConfig,
    seed: int,
    gamma: float = 0.995,
    gae_lambda: float = 0.98,
    step_ticks: int = 500,
    gamma_ticks: float = 250.0,
    pbrs_coef: float = 0.15,
) -> tuple[dict, dict, dict, dict]:
    session = GameSession(bridge, match)
    feats = Features()
    traj = GameTrajectory()
    b_idx = opponent_id(match.bot)
    t0 = time.perf_counter()
    decisions = 0
    error = ""
    outcome = "error"

    try:
        step = session.reset(seed)
        while not (step.done or step.truncated):
            feats.observe(step.obs)
            answer: list[pb.Decision] = []
            if step.pending:
                priv = privileged_features(step.obs, match.bot)
                phi = state_potential(step.obs)
                g_tensor = torch.from_numpy(feats.g).unsqueeze(0)
                priv_tensor = torch.from_numpy(priv).unsqueeze(0)
                with torch.no_grad():
                    val = float(net.value(g_tensor, priv_tensor).item())

                node = traj.begin_step(
                    tick=step.obs.tick, g=feats.g, priv=priv, val=val, phi=phi, bot_idx=b_idx,
                )

                for req in step.pending:
                    if req.kind == pb.DECIDE_ARMY:
                        cand, mask = feats.candidates(req)
                        cand_t = torch.from_numpy(cand).unsqueeze(0)
                        mask_t = torch.from_numpy(mask).unsqueeze(0)
                        with torch.no_grad():
                            logits = net.army_logits(g_tensor, cand_t, mask_t)
                            dist = Categorical(logits=logits)
                            action = int(dist.sample().item())
                            lp = float(dist.log_prob(torch.tensor(action)).item())

                        d = feats.decision(req, action)
                        feats.record(req, d)
                        if d is not None:
                            answer.append(d)
                        node.add_army(cand, mask, action, lp)
                        decisions += 1
                    else:
                        q_idx = QUEUE_INDEX.get(req.queue, 0)
                        q_t = torch.tensor([q_idx])
                        mask = feats.item_mask(req)
                        mask_t = torch.from_numpy(mask).unsqueeze(0)
                        with torch.no_grad():
                            logits = net.item_logits(g_tensor, q_t, mask_t)
                            dist = Categorical(logits=logits)
                            action = int(dist.sample().item())
                            lp = float(dist.log_prob(torch.tensor(action)).item())

                        d = feats.decision(req, action)
                        feats.record(req, d)
                        if d is not None:
                            answer.append(d)
                        node.add_item(q_idx, mask, action, lp)
                        decisions += 1

            step = session.step(answer, step_ticks)
        outcome = step.result or "truncated"
    except (grpc.RpcError, RuntimeError) as e:
        error = repr(e)
    finally:
        session.close()

    if not error:
        term_reward = TERMINAL_REWARDS.get(outcome, -0.5)
        prod_data, army_data, step_data = traj.finish(
            term_reward,
            end_tick=session.tick,
            gamma=gamma,
            gae_lambda=gae_lambda,
            gamma_ticks=gamma_ticks,
            pbrs_coef=pbrs_coef,
        )
    else:
        prod_data = {k: [] for k in PROD_KEYS}
        army_data = {k: [] for k in ARMY_KEYS}
        step_data = {k: [] for k in STEP_KEYS}

    meta = {
        "seed": seed, "bot": match.bot, "outcome": outcome, "ticks": session.tick,
        "decisions": decisions, "steps_joint": len(traj.steps),
        "wall_s": round(time.perf_counter() - t0, 1), "error": error,
    }
    return meta, prod_data, army_data, step_data


def collect_rollouts(
    addresses: list[str],
    net: CommanderNet,
    games: int,
    seed_base: int,
    match: MatchConfig,
    gamma: float = 0.995,
    gae_lambda: float = 0.98,
    recycle_every: int = 60,
    step_ticks: int = 500,
    gamma_ticks: float = 250.0,
    pbrs_coef: float = 0.15,
) -> tuple[dict, dict, dict, dict]:
    net_cpu = copy.deepcopy(net).cpu().eval()
    seeds = [seed_base + i for i in range(games)]
    lock = threading.Lock()
    todo = list(seeds)

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
                    seed = todo.pop(0)

                if recycle_every and played >= recycle_every:
                    bridge.close()
                    daemons.restart(addr)
                    bridge = Bridge(addr)
                    played = 0

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

    normalize_advantages(packed_prod, per_bot=False)
    normalize_advantages(packed_army, per_bot=False)
    beh = compute_behavior_stats(packed_prod, packed_army, packed_step)

    valid_metas = [m for m in metas if not m["error"]]
    outcomes: dict[str, int] = {}
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
        **beh,
    }

    return stats, packed_prod, packed_army, packed_step


def to_torch_dict(d: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in d.items():
        if len(v) == 0:
            out[k] = torch.empty(0, device=device)
            continue
        t = torch.from_numpy(v)
        if k in ("g", "priv", "cand", "adv", "ret", "val", "old_lp"):
            t = t.float()
        elif k in ("queue", "action", "bot_idx"):
            t = t.long()
        elif k == "mask":
            t = t.bool()
        out[k] = t.to(device)
    return out


def ppo_update(
    net: CommanderNet,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    prod: dict[str, torch.Tensor],
    army: dict[str, torch.Tensor],
    steps: dict[str, torch.Tensor],
    ref_net: CommanderNet | None = None,
    critic_only: bool = False,
    epochs: int = 4,
    batch_size: int = 2048,
    clip_eps: float = 0.2,
    ent_coef: float = 0.005,
    kl_coef: float = 0.03,
    army_weight: float = 1.0,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.025,
) -> dict:
    net.train()
    if ref_net is not None:
        ref_net.eval()
    device = next(net.parameters()).device

    n_prod = len(prod.get("g", []))
    n_army = len(army.get("g", []))
    n_step = len(steps.get("g", []))

    metrics = {
        "pi_loss_prod": 0.0, "pi_loss_army": 0.0,
        "val_loss": 0.0,
        "ent_prod": 0.0, "ent_army": 0.0,
        "kl_prod": 0.0, "kl_army": 0.0,
        "kl_ref_prod": 0.0, "kl_ref_army": 0.0,
        "steps": 0, "critic_steps": 0, "epochs_ran": 0,
        "early_stopped": False, "critic_only": critic_only,
    }

    # 1. Update decoupled Asymmetric Critic on unique joint StepNodes (runs even in warmup).
    if n_step > 0:
        c_batch = min(batch_size, max(128, n_step // 4))
        c_steps_per_epoch = max(1, (n_step + c_batch - 1) // c_batch)
        c_epochs = max(epochs, 6) if critic_only else epochs
        for _ in range(c_epochs):
            perm_s = torch.randperm(n_step, device=device)
            for s in range(c_steps_per_epoch):
                si = perm_s[s * c_batch:min(n_step, (s + 1) * c_batch)]
                v_pred = net.value(steps["g"][si], steps["priv"][si])
                v_loss = F.smooth_l1_loss(v_pred, steps["ret"][si], beta=0.5)
                critic_opt.zero_grad(set_to_none=True)
                v_loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(list(net.critic_parameters()), max_grad_norm)
                critic_opt.step()
                metrics["val_loss"] += float(v_loss.item())
                metrics["critic_steps"] += 1

    c_div = max(1, metrics["critic_steps"])
    val_loss_avg = round(metrics["val_loss"] / c_div, 4)

    if critic_only or (n_prod == 0 and n_army == 0):
        return {
            "pi_loss_prod": 0.0, "pi_loss_army": 0.0,
            "val_loss": val_loss_avg,
            "val_loss_prod": val_loss_avg, "val_loss_army": val_loss_avg,
            "ent_prod": 0.0, "ent_army": 0.0,
            "kl_prod": 0.0, "kl_army": 0.0,
            "kl_ref_prod": 0.0, "kl_ref_army": 0.0,
            "epochs_ran": 0, "early_stopped": False, "critic_only": critic_only,
        }

    # 2. Update Actor heads + trunk with PPO-clip + masked KL anchor to BC teacher.
    actor_params = list(net.actor_parameters())
    steps_per_epoch = max(1, n_prod // batch_size) if n_prod > 0 else max(1, n_army // batch_size)
    army_batch = max(1, batch_size * n_army // max(1, n_prod)) if n_prod > 0 and n_army > 0 else batch_size

    for _ in range(epochs):
        metrics["epochs_ran"] += 1
        perm_prod = torch.randperm(n_prod, device=device) if n_prod > 0 else None

        for s in range(steps_per_epoch):
            total_loss = torch.tensor(0.0, device=device)

            if n_prod > 0 and perm_prod is not None:
                pi = perm_prod[s * batch_size:min(n_prod, (s + 1) * batch_size)]
                g_p, q_p, m_p = prod["g"][pi], prod["queue"][pi], prod["mask"][pi]
                logits = net.item_logits(g_p, q_p, m_p)
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(prod["action"][pi])
                log_ratio = torch.clamp(new_lp - prod["old_lp"][pi], -10.0, 10.0)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * prod["adv"][pi]
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * prod["adv"][pi]
                pi_loss_p = -torch.min(surr1, surr2).mean()

                ent_p = dist.entropy().mean()
                kl_p = ((ratio - 1.0) - log_ratio).mean()

                kl_ref_p = torch.tensor(0.0, device=device)
                if ref_net is not None and kl_coef > 0.0:
                    with torch.no_grad():
                        ref_logits_p = ref_net.item_logits(g_p, q_p, m_p)
                    kl_ref_p = masked_kl(logits, ref_logits_p, m_p)

                total_loss = total_loss + pi_loss_p + kl_coef * kl_ref_p - ent_coef * ent_p
                metrics["pi_loss_prod"] += float(pi_loss_p.item())
                metrics["ent_prod"] += float(ent_p.item())
                metrics["kl_prod"] += float(kl_p.item())
                metrics["kl_ref_prod"] += float(kl_ref_p.item())

            if n_army > 0:
                ai = torch.randint(0, n_army, (min(n_army, army_batch),), device=device)
                g_a, c_a, m_a = army["g"][ai], army["cand"][ai], army["mask"][ai]
                logits_a = net.army_logits(g_a, c_a, m_a)
                dist_a = Categorical(logits=logits_a)
                new_lp_a = dist_a.log_prob(army["action"][ai])
                log_ratio_a = torch.clamp(new_lp_a - army["old_lp"][ai], -10.0, 10.0)
                ratio_a = torch.exp(log_ratio_a)
                surr1_a = ratio_a * army["adv"][ai]
                surr2_a = torch.clamp(ratio_a, 1.0 - clip_eps, 1.0 + clip_eps) * army["adv"][ai]
                pi_loss_a = -torch.min(surr1_a, surr2_a).mean()

                ent_a = dist_a.entropy().mean()
                kl_a = ((ratio_a - 1.0) - log_ratio_a).mean()

                kl_ref_a = torch.tensor(0.0, device=device)
                if ref_net is not None and kl_coef > 0.0:
                    with torch.no_grad():
                        ref_logits_a = ref_net.army_logits(g_a, c_a, m_a)
                    kl_ref_a = masked_kl(logits_a, ref_logits_a, m_a)

                total_loss = total_loss + army_weight * (pi_loss_a + kl_coef * kl_ref_a) - ent_coef * ent_a
                metrics["pi_loss_army"] += float(pi_loss_a.item())
                metrics["ent_army"] += float(ent_a.item())
                metrics["kl_army"] += float(kl_a.item())
                metrics["kl_ref_army"] += float(kl_ref_a.item())

            actor_opt.zero_grad(set_to_none=True)
            total_loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(actor_params, max_grad_norm)
            actor_opt.step()
            metrics["steps"] += 1

        cur_steps = max(1, metrics["steps"])
        avg_kl_p = metrics["kl_prod"] / cur_steps if n_prod > 0 else 0.0
        avg_kl_a = metrics["kl_army"] / cur_steps if n_army > 0 else 0.0
        mean_kl = (avg_kl_p + avg_kl_a) / max(1, int(n_prod > 0) + int(n_army > 0))
        if mean_kl > target_kl:
            metrics["early_stopped"] = True
            break

    n = max(1, metrics["steps"])
    return {
        "pi_loss_prod": round(metrics["pi_loss_prod"] / n, 4),
        "pi_loss_army": round(metrics["pi_loss_army"] / n, 4),
        "val_loss": val_loss_avg,
        "val_loss_prod": val_loss_avg,
        "val_loss_army": val_loss_avg,
        "ent_prod": round(metrics["ent_prod"] / n, 4),
        "ent_army": round(metrics["ent_army"] / n, 4),
        "kl_prod": round(metrics["kl_prod"] / n, 4),
        "kl_army": round(metrics["kl_army"] / n, 4),
        "kl_ref_prod": round(metrics["kl_ref_prod"] / n, 4),
        "kl_ref_army": round(metrics["kl_ref_army"] / n, 4),
        "epochs_ran": metrics["epochs_ran"],
        "early_stopped": metrics["early_stopped"],
        "critic_only": critic_only,
    }


def policy_spec(ckpt_path: Path, eval_temp: float) -> str:
    if eval_temp <= 0.0:
        return f"net:{ckpt_path}"
    return f"net-t{eval_temp:g}:{ckpt_path}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="runs/phase2_5/bc_s3/model.pt", help="starting checkpoint (BC or previous PPO)")
    ap.add_argument("--ref", default="", help="frozen KL anchor checkpoint (defaults to --init)")
    ap.add_argument("--out", required=True, help="output directory for checkpoints and logs")
    ap.add_argument("--iterations", type=int, default=80, help="total PPO iterations")
    ap.add_argument("--games-per-iter", type=int, default=48, help="number of games collected per iteration")
    ap.add_argument("--bot", default="hard", help="opponent difficulty during training")
    ap.add_argument("--map", default=MatchConfig.map_name)
    ap.add_argument("--max-ticks", type=int, default=54000)
    ap.add_argument("--daemons", default=DEFAULT_DAEMONS)
    ap.add_argument("--seed-base", type=int, default=100000, help="starting seed for rollouts (never overlaps with eval)")
    ap.add_argument("--lr", type=float, default=5e-5, help="actor learning rate")
    ap.add_argument("--critic-lr", type=float, default=3e-4, help="decoupled critic learning rate")
    ap.add_argument("--critic-warmup-iters", type=int, default=6, help="train critic only for the first N iterations")
    ap.add_argument("--reset-critic", action="store_true", help="re-initialize critic weights on load")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995, help="discount factor per --gamma-ticks ticks")
    ap.add_argument("--gamma-ticks", type=float, default=250.0, help="reference tick interval for SMDP discounting")
    ap.add_argument("--gae-lambda", type=float, default=0.98, help="GAE lambda")
    ap.add_argument("--pbrs-coef", type=float, default=0.15, help="exact telescoping PBRS weight (0 disables)")
    ap.add_argument("--kl-coef", type=float, default=0.03, help="KL penalty toward --ref teacher (0 disables)")
    ap.add_argument("--ppo-epochs", type=int, default=4, help="optimization epochs per iteration")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--ent-decay", type=float, default=0.995)
    ap.add_argument("--ent-min", type=float, default=0.002)
    ap.add_argument("--army-weight", type=float, default=1.0)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.025)
    ap.add_argument("--eval-every", type=int, default=10, help="evaluate every N iterations")
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--eval-seeds-base", type=int, default=20000, help="selection seeds (20000-20199)")
    ap.add_argument("--eval-temp", type=float, default=0.0, help="evaluation softmax temperature (0.0 = argmax)")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--recycle-every", type=int, default=50)
    ap.add_argument("--tb", action=argparse.BooleanOptionalAction, default=True, help="log metrics to TensorBoard")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    daemons_list = [d.strip() for d in args.daemons.split(",") if d.strip()]
    writer = SummaryWriter(log_dir=str(out / "tb")) if (args.tb and HAS_TB) else None

    ref_path = args.ref or args.init
    print(json.dumps({
        "msg": "Starting Phase 3 v2 SMDP-PPO",
        "init": args.init, "ref": ref_path, "out": str(out),
        "device": device, "daemons": len(daemons_list), "games_per_iter": args.games_per_iter,
        "gamma": args.gamma, "gamma_ticks": args.gamma_ticks, "gae_lambda": args.gae_lambda,
        "pbrs_coef": args.pbrs_coef, "kl_coef": args.kl_coef, "critic_warmup": args.critic_warmup_iters,
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
    best_eval_wr = -1.0
    log = (out / "metrics.jsonl").open("a", encoding="utf-8")

    t_start = time.perf_counter()
    match = MatchConfig(map_name=args.map, bot=args.bot, max_ticks=args.max_ticks, brain="external")

    for it in range(1, args.iterations + 1):
        t_it = time.perf_counter()
        in_warmup = it <= args.critic_warmup_iters

        collect_stats, prod_np, army_np, step_np = collect_rollouts(
            addresses=daemons_list,
            net=net,
            games=args.games_per_iter,
            seed_base=cur_seed,
            match=match,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            recycle_every=args.recycle_every,
            gamma_ticks=args.gamma_ticks,
            pbrs_coef=args.pbrs_coef,
        )
        cur_seed += args.games_per_iter

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
            "it_s": round(time.perf_counter() - t_it, 1),
            "total_s": round(time.perf_counter() - t_start, 1),
            **collect_stats,
            **update_metrics,
        }

        if args.eval_every > 0 and it % args.eval_every == 0:
            print(f"[Iter {it}] Running evaluation ({args.eval_games} games vs {args.bot})...", flush=True)
            eval_match = MatchConfig(map_name=args.map, bot=args.bot, max_ticks=args.max_ticks)
            eval_summary = evaluate(
                daemons=daemons_list,
                policy=policy_spec(out / "checkpoint_latest.pt", args.eval_temp),
                match=eval_match,
                games=args.eval_games,
                seed_base=args.eval_seeds_base,
                step_ticks=500,
                log=lambda _: None,
                recycle_every=args.recycle_every,
                pre_recycle=True,
            )
            err_count = eval_summary["outcomes"].get("error", 0)
            it_record["eval"] = {
                "games": eval_summary["games"],
                "win_rate": eval_summary["win_rate"],
                "win_rate_ci95": eval_summary["win_rate_ci95"],
                "outcomes": eval_summary["outcomes"],
            }
            if eval_summary["win_rate"] > best_eval_wr and err_count <= max(2, args.eval_games // 10):
                best_eval_wr = eval_summary["win_rate"]
                net.save(str(out / "checkpoint_best.pt"), iteration=it, eval=eval_summary)
                it_record["new_best"] = True

        log.write(json.dumps(it_record) + "\n")
        log.flush()
        print(json.dumps(it_record), flush=True)

        if writer is not None:
            writer.add_scalar("rollout/win_rate", collect_stats["win_rate"], it)
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
                writer.add_scalar("eval/win_rate", eval_summary["win_rate"], it)
                writer.add_scalar("eval/mean_ticks", eval_summary["mean_ticks"], it)

            writer.flush()

    log.close()
    if writer is not None:
        writer.close()
    print(f"PPO complete in {time.perf_counter() - t_start:.0f}s. Best eval win rate: {best_eval_wr}", flush=True)


if __name__ == "__main__":
    main()
