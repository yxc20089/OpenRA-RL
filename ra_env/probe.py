"""Minimal fixed policy used by the bench and determinism tools.

It only deploys the MCV and then idles, so the game length is decided by the
bot. Enough to load the engine like a real match without any learning code.
"""

import hashlib

from ra_env._proto import rl_bridge_pb2 as pb


def deploy_mcv_commands(obs: pb.GameObservation) -> list[pb.Command]:
    return [pb.Command(action=pb.DEPLOY, actor_id=u.actor_id) for u in obs.units if u.type == "mcv"]


def obs_digest(obs: pb.GameObservation, include_tick: bool = True) -> str:
    """Hash of the game state visible in an observation (agent side + spectator summary)."""
    h = hashlib.sha1()
    if include_tick:
        h.update(str(obs.tick).encode())
    for u in sorted(obs.units, key=lambda u: u.actor_id):
        h.update(f"u{u.actor_id}:{u.type}:{u.cell_x},{u.cell_y}:{u.hp_percent:.3f}".encode())
    for b in sorted(obs.buildings, key=lambda b: b.actor_id):
        h.update(f"b{b.actor_id}:{b.type}:{b.cell_x},{b.cell_y}:{b.hp_percent:.3f}".encode())
    for e in sorted(obs.visible_enemies, key=lambda e: e.actor_id):
        h.update(f"e{e.actor_id}:{e.type}:{e.cell_x},{e.cell_y}".encode())
    g = obs.global_summary
    for side in (g.own, g.enemy):
        h.update(f"s{side.cash}:{side.unit_value}:{side.building_value}:{side.n_buildings}:{side.earned}".encode())
    h.update(f"r{obs.done}:{obs.result}".encode())
    return h.hexdigest()
