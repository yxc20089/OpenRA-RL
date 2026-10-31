"""Policies for the v3 commander.

A policy picks the commander brain and, for brain "external", answers the
pending decisions of each step.
"""

from typing import Protocol

from ra_env import StepResult, pb


class Policy(Protocol):
    brain: str  # "external" | "builtin"

    def reset(self, seed: int) -> None: ...

    def decide(self, step: StepResult) -> list[pb.Decision]: ...


class BuiltinPolicy:
    """The normal AI's own heuristics drive the commander (executor parity / teacher)."""

    brain = "builtin"

    def reset(self, seed: int) -> None:
        pass

    def decide(self, step: StepResult) -> list[pb.Decision]:
        return []


def make_policy(name: str) -> Policy:
    """builtin | scripted | random | net:<model.pt> | net-sample:<model.pt> | net-t<temp>:<model.pt>"""
    if name == "builtin":
        return BuiltinPolicy()
    if name == "scripted":
        from ra_rl.policies.scripted import ScriptedPolicy
        return ScriptedPolicy()
    if name == "random":
        from ra_rl.policies.net import RandomPolicy
        return RandomPolicy()
    kind, _, path = name.partition(":")
    if kind in ("net", "net-sample") and path:
        from ra_rl.policies.net import NetPolicy
        return NetPolicy(path, sample=kind == "net-sample")
    if kind.startswith("net-t") and path:
        from ra_rl.policies.net import NetPolicy
        temp = float(kind[len("net-t"):])
        return NetPolicy(path, temperature=temp)
    raise ValueError(
        f"unknown policy {name!r}; expected builtin | scripted | random | net:<path> | net-sample:<path> | net-t<T>:<path>"
    )
