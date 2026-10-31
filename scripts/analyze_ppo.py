"""Inspect PPO or League metrics.jsonl logs, including behavioral telemetry.

    python scripts/analyze_ppo.py runs/phase3/ppo_v2/metrics.jsonl
    python scripts/analyze_ppo.py runs/phase4/league_v2/metrics.jsonl
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default="runs/phase3/ppo_v2/metrics.jsonl")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Metrics file not found: {path}")

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(lines)} iterations from {path}")
    if not lines:
        return

    evals = [m for m in lines if "eval" in m]
    if evals:
        print("\n--- OFFICIAL EVALUATIONS ---")
        for e in evals:
            it = e["iteration"]
            ev = e["eval"]
            nb = "[BEST]" if e.get("new_best") else ""
            if "win_rate" in ev:
                ci = ev["win_rate_ci95"]
                oc = ev["outcomes"]
                print(
                    f"Iter {it:3d}: WR={ev['win_rate']*100:5.1f}% [{ci[0]*100:4.1f}%, {ci[1]*100:4.1f}%] "
                    f"| W:{oc.get('win',0):2d} L:{oc.get('lose',0):2d} T:{oc.get('truncated',0):2d} "
                    f"E:{oc.get('error',0):2d} {nb}"
                )
            else:
                parts = []
                for k, v in ev.items():
                    if isinstance(v, dict) and "win_rate" in v:
                        oc = v.get("outcomes", {})
                        parts.append(f"{k}={v['win_rate']*100:5.1f}%(E:{oc.get('error',0)})")
                harm = ev.get("harmonic_score", 0.0)
                print(f"Iter {it:3d}: {' | '.join(parts)} | Harm={harm:.3f} {nb}")

    print("\n--- ROLLOUT PROGRESSION (10-ITERATION BLOCKS) ---")
    for chunk_start in range(0, len(lines), 10):
        chunk = lines[chunk_start:chunk_start + 10]
        n = len(chunk)
        avg = lambda key, default=0.0: sum(float(m.get(key, default)) for m in chunk) / n
        print(
            f"Iters {chunk[0]['iteration']:3d}-{chunk[-1]['iteration']:3d} | "
            f"WR:{avg('win_rate')*100:5.1f}% | Ticks:{avg('mean_ticks'):5.0f} | "
            f"ExpVar:{avg('explained_var'):+.2f} | KLref(P/A):{avg('kl_ref_prod'):.3f}/{avg('kl_ref_army'):.3f} | "
            f"Veh(2tnk/harv):{avg('beh_veh_2tnk_pct'):4.1f}%/{avg('beh_veh_harv_pct'):4.1f}% | "
            f"BldWait:{avg('beh_bld_wait_pct'):4.1f}% | ArmyATK:{avg('beh_army_atk_pct'):4.1f}%"
        )

    best_ckpt = path.parent / "checkpoint_best.pt"
    if best_ckpt.exists():
        import torch
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        print("\n--- CHECKPOINT_BEST.PT ---")
        print(f"Iteration: {ckpt.get('iteration')}")
        print(f"Eval stats: {ckpt.get('eval')}")


if __name__ == "__main__":
    main()
