import pytest
import torch

from ra_rl.features import CAND_DIM, FEATURE_VERSION, GLOBAL_DIM, MAX_TARGETS, N_ITEM_ACTIONS, PRIV_DIM
from ra_rl.model import CommanderNet


def test_commandernet_decoupled_critic_and_logits():
    net = CommanderNet(hidden=64, cand_hidden=32)
    B = 4
    g = torch.randn(B, GLOBAL_DIM)
    priv = torch.randn(B, PRIV_DIM)

    # Value works both with priv and with default priv=None
    v_priv = net.value(g, priv)
    v_zero = net.value(g)
    assert v_priv.shape == (B,) and v_zero.shape == (B,)

    # Value backward MUST NOT touch any actor parameter
    net.zero_grad()
    v_priv.sum().backward()
    for p in net.actor_parameters():
        assert p.grad is None
    assert any(p.grad is not None for p in net.critic_parameters())

    # Item logits
    queue = torch.randint(0, 6, (B,))
    item_mask = torch.ones(B, N_ITEM_ACTIONS, dtype=torch.bool)
    item_mask[:, 5:] = False
    logits = net.item_logits(g, queue, item_mask)
    assert logits.shape == (B, N_ITEM_ACTIONS)
    assert (logits[:, 5:] < -1e8).all()

    # Army logits
    cand = torch.randn(B, MAX_TARGETS, CAND_DIM)
    army_mask = torch.ones(B, MAX_TARGETS + 2, dtype=torch.bool)
    army_mask[:, 3:MAX_TARGETS] = False
    a_logits = net.army_logits(g, cand, army_mask)
    assert a_logits.shape == (B, MAX_TARGETS + 2)
    assert (a_logits[:, 3:MAX_TARGETS] < -1e8).all()


def test_load_checkpoint_with_missing_critic_and_reject_wrong_feature_version(tmp_path):
    net = CommanderNet(hidden=32, cand_hidden=16)
    actor_only_state = {k: v for k, v in net.state_dict().items() if not k.startswith("critic.")}
    ckpt_path = tmp_path / "actor_only.pt"
    torch.save({"config": net.config, "state": actor_only_state}, ckpt_path)

    loaded = CommanderNet.load(str(ckpt_path))
    v = loaded.value(torch.randn(2, GLOBAL_DIM), torch.randn(2, PRIV_DIM))
    assert v.shape == (2,)

    bad_path = tmp_path / "old_v1.pt"
    torch.save({"config": {**net.config, "feature_version": FEATURE_VERSION - 1}, "state": actor_only_state}, bad_path)
    with pytest.raises(ValueError, match="features v"):
        CommanderNet.load(str(bad_path))
