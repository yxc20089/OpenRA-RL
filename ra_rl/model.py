"""Commander network: an actor MLP trunk + two action heads, and a decoupled asymmetric critic.

- item head (BUILD / TRAIN): logits over the item vocabulary + wait, conditioned on
  the queue, masked to what the queue offers right now.
- army head (ARMY): a pointer over the target candidates (each scored against the
  trunk) plus two fixed actions, hold and keep.
- critic (training only, CTDE): an independent MLP over [g, priv] with zero shared
  weights with the actor trunk, preventing value-gradient interference on the policy.
"""

from typing import Iterator

import torch
from torch import nn

from ra_rl.features import CAND_DIM, FEATURE_VERSION, GLOBAL_DIM, MAX_TARGETS, N_ITEM_ACTIONS, PRIV_DIM
from ra_rl.vocab import QUEUES

NEG = -1e9


class CommanderNet(nn.Module):
    def __init__(self, hidden: int = 256, cand_hidden: int = 128, priv_dim: int = PRIV_DIM):
        super().__init__()
        self.priv_dim = priv_dim
        self.config = {
            "hidden": hidden,
            "cand_hidden": cand_hidden,
            "priv_dim": priv_dim,
            "feature_version": FEATURE_VERSION,
        }
        self.trunk = nn.Sequential(
            nn.Linear(GLOBAL_DIM, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.queue_emb = nn.Embedding(len(QUEUES), hidden)
        self.item_head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, N_ITEM_ACTIONS))
        self.cand_enc = nn.Sequential(nn.Linear(CAND_DIM, cand_hidden), nn.ReLU())
        self.cand_score = nn.Sequential(
            nn.Linear(hidden + cand_hidden, cand_hidden), nn.ReLU(),
            nn.Linear(cand_hidden, 1),
        )
        self.army_fixed = nn.Linear(hidden, 2)  # hold, keep

        # Independent asymmetric critic trunk + scalar head (CTDE; unused at eval time).
        self.critic = nn.Sequential(
            nn.Linear(GLOBAL_DIM + priv_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def actor_parameters(self) -> Iterator[nn.Parameter]:
        for name, param in self.named_parameters():
            if not name.startswith("critic."):
                yield param

    def critic_parameters(self) -> Iterator[nn.Parameter]:
        return self.critic.parameters()

    def reset_critic(self) -> None:
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        last = self.critic[-1]
        if isinstance(last, nn.Linear):
            nn.init.orthogonal_(last.weight, gain=0.1)

    def value(self, g: torch.Tensor, priv: torch.Tensor | None = None) -> torch.Tensor:
        if priv is None:
            priv = torch.zeros(g.shape[0], self.priv_dim, device=g.device, dtype=g.dtype)
        return self.critic(torch.cat([g, priv], dim=-1)).squeeze(-1)

    def item_logits(self, g: torch.Tensor, queue: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.trunk(g) + self.queue_emb(queue)
        return self.item_head(h).masked_fill(~mask, NEG)

    def army_logits(self, g: torch.Tensor, cand: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.trunk(g)
        c = self.cand_enc(cand)                                            # B, T, C
        hc = torch.cat([h.unsqueeze(1).expand(-1, MAX_TARGETS, -1), c], dim=-1)
        scores = self.cand_score(hc).squeeze(-1)                           # B, T
        logits = torch.cat([scores, self.army_fixed(h)], dim=-1)           # B, T + 2
        return logits.masked_fill(~mask, NEG)

    def save(self, path: str, **extra) -> None:
        torch.save({"config": self.config, "state": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CommanderNet":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        if cfg.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"{path}: features v{cfg.get('feature_version')}, code is v{FEATURE_VERSION}")
        net = cls(
            hidden=cfg["hidden"],
            cand_hidden=cfg["cand_hidden"],
            priv_dim=cfg.get("priv_dim", PRIV_DIM),
        )
        missing, unexpected = net.load_state_dict(ckpt["state"], strict=False)
        non_critic_missing = [k for k in missing if not k.startswith("critic.")]
        unexpected = [k for k in unexpected if not k.startswith(("critic.", "value_head."))]
        if non_critic_missing or unexpected:
            raise RuntimeError(f"Error loading {path}: missing {non_critic_missing}, unexpected {unexpected}")
        return net.to(device).eval()
