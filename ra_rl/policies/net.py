"""Policies that act through the network's action space (features.py)."""

import random

import numpy as np

from ra_env import StepResult, pb
from ra_rl.features import Features
from ra_rl.model import NEG
from ra_rl.vocab import QUEUE_INDEX


class RandomPolicy:
    """Uniform over the valid actions of each request (the floor for every criterion)."""

    brain = "external"

    def __init__(self) -> None:
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.feats = Features()

    def decide(self, step: StepResult) -> list[pb.Decision]:
        self.feats.observe(step.obs)
        out = []
        for req in step.pending:
            mask = self.feats.candidates(req)[1] if req.kind == pb.DECIDE_ARMY else self.feats.item_mask(req)
            a = self.rng.choice(np.flatnonzero(mask).tolist())
            d = self.feats.decision(req, a)
            self.feats.record(req, d)
            if d is not None:
                out.append(d)
        return out


class NetPolicy:
    """The commander network, greedy (argmax) by default or temperature-sampled, on CPU."""

    brain = "external"

    def __init__(self, path, sample: bool = False, temperature: float | None = None):
        import torch

        from ra_rl.model import CommanderNet

        torch.set_num_threads(1)  # one game per thread; tiny batches
        self.torch = torch
        if isinstance(path, CommanderNet):
            self.net = path
        else:
            self.net = CommanderNet.load(str(path))
        if temperature is None:
            self.temperature = 1.0 if sample else 0.0
        else:
            self.temperature = max(0.0, float(temperature))
        self.sample = self.temperature > 0.0
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.feats = Features()
        self.gen = self.torch.Generator().manual_seed(seed)

    def actions(self, step: StepResult) -> list[int]:
        """Observe the step and pick an action for each pending request (not yet recorded)."""
        torch = self.torch
        self.feats.observe(step.obs)
        if not step.pending:
            return []
        g = torch.from_numpy(self.feats.g).unsqueeze(0)
        out = []
        with torch.no_grad():
            for req in step.pending:
                if req.kind == pb.DECIDE_ARMY:
                    cand, mask = self.feats.candidates(req)
                    mask_t = torch.from_numpy(mask).unsqueeze(0)
                    logits = self.net.army_logits(g, torch.from_numpy(cand).unsqueeze(0), mask_t)
                else:
                    queue = torch.tensor([QUEUE_INDEX.get(req.queue, 0)])
                    mask_t = torch.from_numpy(self.feats.item_mask(req)).unsqueeze(0)
                    logits = self.net.item_logits(g, queue, mask_t)
                if self.temperature > 0.0:
                    scaled = (logits[0] / self.temperature).masked_fill(~mask_t[0], NEG)
                    a = int(torch.multinomial(torch.softmax(scaled, -1), 1, generator=self.gen))
                else:
                    a = int(logits[0].argmax())
                out.append(a)
        return out

    def decide(self, step: StepResult) -> list[pb.Decision]:
        out = []
        for req, a in zip(step.pending, self.actions(step), strict=True):
            d = self.feats.decision(req, a)
            self.feats.record(req, d)
            if d is not None:
                out.append(d)
        return out
