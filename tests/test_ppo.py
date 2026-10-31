import numpy as np
import pytest
import torch

from ra_rl.features import CAND_DIM, GLOBAL_DIM, MAX_TARGETS, N_ITEM_ACTIONS, PRIV_DIM
from ra_rl.model import CommanderNet
from ra_rl.ppo import GameTrajectory, masked_kl, ppo_update


def test_smdp_joint_step_no_intratick_decay():
    traj = GameTrajectory()
    g = np.zeros(GLOBAL_DIM, dtype=np.float32)
    priv = np.zeros(PRIV_DIM, dtype=np.float32)
    mask_p = np.ones(N_ITEM_ACTIONS, dtype=bool)
    cand = np.zeros((MAX_TARGETS, CAND_DIM), dtype=np.float32)
    mask_a = np.ones(MAX_TARGETS + 2, dtype=bool)

    # Tick 0: 2 prod decisions + 1 army decision simultaneously
    n0 = traj.begin_step(tick=0, g=g, priv=priv, val=0.5, phi=0.0)
    n0.add_item(queue=0, mask=mask_p, action=1, log_prob=-0.2)
    n0.add_item(queue=2, mask=mask_p, action=3, log_prob=-0.4)
    n0.add_army(cand=cand, mask=mask_a, action=0, log_prob=-0.5)

    # Tick 250 (1 interval of gamma_ticks=250): 1 prod decision, ends at tick 500 with win (+1.0)
    n1 = traj.begin_step(tick=250, g=g, priv=priv, val=0.8, phi=0.0)
    n1.add_item(queue=3, mask=mask_p, action=2, log_prob=-0.3)

    prod, army, steps = traj.finish(
        terminal_reward=1.0, end_tick=500, gamma=0.9, gae_lambda=0.95, gamma_ticks=250.0, pbrs_coef=0.0,
    )

    assert len(steps["ret"]) == 2
    assert len(prod["adv"]) == 3 and len(army["adv"]) == 1

    # Step 1 (dt=250 -> gamma_1=0.9): delta_1 = 1.0 - 0.8 = 0.2 -> adv_1 = 0.2, ret_1 = 1.0
    assert pytest.approx(steps["ret"][1], rel=1e-5) == 1.0
    # Step 0 (dt=250 -> gamma_0=0.9): delta_0 = 0.9 * 0.8 - 0.5 = 0.22 -> adv_0 = 0.22 + 0.9*0.95*0.2 = 0.391
    assert pytest.approx(prod["adv"][0], rel=1e-5) == 0.391
    # Simultaneous decisions at tick 0 receive the EXACT same temporal advantage (zero intra-tick decay)
    assert prod["adv"][0] == prod["adv"][1] == army["adv"][0]


def test_pbrs_exact_telescoping_invariance():
    """Ng et al. (1999) invariance: under variable SMDP ticks, total return = disc_R_term - kappa * phi_0."""
    traj = GameTrajectory()
    g = np.zeros(GLOBAL_DIM, dtype=np.float32)
    priv = np.zeros(PRIV_DIM, dtype=np.float32)
    mask_p = np.ones(N_ITEM_ACTIONS, dtype=bool)

    ticks = [10, 110, 610, 1360]
    phis = [0.12, -0.45, 0.80, -0.30]
    for t, phi in zip(ticks, phis):
        node = traj.begin_step(tick=t, g=g, priv=priv, val=0.0, phi=phi)
        node.add_item(queue=0, mask=mask_p, action=0, log_prob=0.0)

    gamma = 0.99
    gamma_ticks = 250.0
    end_tick = 2000
    r_term = -0.5
    kappa = 0.25

    _, _, steps = traj.finish(
        terminal_reward=r_term,
        end_tick=end_tick,
        gamma=gamma,
        gae_lambda=1.0,  # exact Monte Carlo return
        gamma_ticks=gamma_ticks,
        pbrs_coef=kappa,
    )

    # Discount from tick 10 to last step tick 1360: gamma^((1360 - 10) / 250)
    expected_disc = gamma ** ((ticks[-1] - ticks[0]) / gamma_ticks)
    expected_ret0 = expected_disc * r_term - kappa * phis[0]
    assert pytest.approx(steps["ret"][0], abs=1e-6) == expected_ret0


def test_masked_kl_properties_and_extreme_logit_safety():
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    ref_logits = torch.tensor([[50.0, -50.0, 0.0, 0.0], [1.5, 99.0, -2.0, 0.0]])

    # Identical logits -> 0 KL and 0 gradient
    logits_same = ref_logits.clone().requires_grad_(True)
    kl_zero = masked_kl(logits_same, ref_logits, mask)
    assert pytest.approx(float(kl_zero.item()), abs=1e-6) == 0.0
    kl_zero.backward()
    assert torch.isfinite(logits_same.grad).all()
    assert torch.allclose(logits_same.grad, torch.zeros_like(logits_same.grad), atol=1e-6)

    # Shifted logits -> strictly positive finite KL and finite gradient (no inf/NaN despite -50/+50)
    logits_diff = torch.tensor([[0.0, 2.0, 0.0, 0.0], [-1.0, 0.0, 3.0, 0.0]], requires_grad=True)
    kl_pos = masked_kl(logits_diff, ref_logits, mask)
    assert float(kl_pos.item()) > 0.5 and torch.isfinite(kl_pos)
    kl_pos.backward()
    assert torch.isfinite(logits_diff.grad).all()
    assert logits_diff.grad[0, 2] == 0.0 and logits_diff.grad[1, 1] == 0.0  # masked slots untouched


def test_ppo_update_critic_warmup_and_full_step():
    net = CommanderNet(hidden=32, cand_hidden=16)
    ref_net = CommanderNet(hidden=32, cand_hidden=16)
    ref_net.load_state_dict(net.state_dict())
    ref_net.requires_grad_(False)

    actor_opt = torch.optim.AdamW(list(net.actor_parameters()), lr=1e-3)
    critic_opt = torch.optim.AdamW(list(net.critic_parameters()), lr=1e-3)

    prod = {
        "g": torch.randn(16, GLOBAL_DIM),
        "queue": torch.randint(0, 6, (16,)),
        "mask": torch.ones(16, N_ITEM_ACTIONS, dtype=torch.bool),
        "action": torch.randint(0, N_ITEM_ACTIONS, (16,)),
        "old_lp": torch.full((16,), -2.0),
        "adv": torch.randn(16),
    }
    army = {
        "g": torch.randn(8, GLOBAL_DIM),
        "cand": torch.randn(8, MAX_TARGETS, CAND_DIM),
        "mask": torch.ones(8, MAX_TARGETS + 2, dtype=torch.bool),
        "action": torch.randint(0, MAX_TARGETS + 2, (8,)),
        "old_lp": torch.full((8,), -1.5),
        "adv": torch.randn(8),
    }
    steps = {
        "g": torch.randn(20, GLOBAL_DIM),
        "priv": torch.randn(20, PRIV_DIM),
        "ret": torch.randn(20),
    }

    actor_before = [p.detach().clone() for p in net.actor_parameters()]
    critic_before = [p.detach().clone() for p in net.critic_parameters()]

    # 1. Warmup: critic changes, actor stays bit-identical
    m_warm = ppo_update(net, actor_opt, critic_opt, prod, army, steps, ref_net=ref_net, critic_only=True, batch_size=8)
    assert m_warm["critic_only"] and m_warm["epochs_ran"] == 0
    for p0, p1 in zip(actor_before, net.actor_parameters()):
        assert torch.equal(p0, p1)
    assert any(not torch.equal(c0, c1) for c0, c1 in zip(critic_before, net.critic_parameters()))

    # 2. Full PPO + KL anchor update: actor also updates cleanly
    m_full = ppo_update(net, actor_opt, critic_opt, prod, army, steps, ref_net=ref_net, critic_only=False, batch_size=8)
    assert m_full["epochs_ran"] >= 1
    assert any(not torch.equal(p0, p1) for p0, p1 in zip(actor_before, net.actor_parameters()))
