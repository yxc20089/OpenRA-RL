r"""End-to-end resumable runner for Phase 2.5 -> Phase 4 v2 recovery (Option A default).

    # Run the default recovery pipeline (Phase 2.5 -> Phase 4 v2 direct from bc_s3):
    .\.venv\Scripts\python.exe scripts/run_recovery_pipeline.py --stage all

    # Or run individual stages:
    .\.venv\Scripts\python.exe scripts/run_recovery_pipeline.py --stage phase2_5
    .\.venv\Scripts\python.exe scripts/run_recovery_pipeline.py --stage phase4
    .\.venv\Scripts\python.exe scripts/run_recovery_pipeline.py --stage phase3
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ra_rl.eval import DEFAULT_DAEMONS


def run_cmd(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")


def eval_gate(daemons: str, policy: str, bot: str, games: int, min_wr: float, out_jsonl: Path) -> float:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl.exists():
        out_jsonl.unlink()
    run_cmd([
        sys.executable, "-m", "ra_rl.eval",
        "--daemons", daemons,
        "--policy", policy,
        "--bot", bot,
        "--games", str(games),
        "--seed-base", "20000",
        "--max-ticks", "54000",
        "--pre-recycle",
        "--out", str(out_jsonl),
    ])
    lines = [json.loads(x) for x in out_jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
    summary = lines[-1]["summary"]
    wr = float(summary["win_rate"])
    errs = int(summary["outcomes"].get("error", 0))
    print(f"[GATE] policy={policy} vs {bot}: WR={wr*100:.1f}% (min {min_wr*100:.1f}%), errors={errs}", flush=True)
    if wr < min_wr:
        raise SystemExit(f"Gate failed for {policy} vs {bot}: {wr:.3f} < {min_wr:.3f}")
    return wr


def run_phase2_5(args) -> None:
    # 1. Record noisy multi-bot scripted demonstrations (seeds 1..1800)
    run_cmd([
        sys.executable, "-m", "ra_rl.record",
        "--teacher", "scripted",
        "--bots", "hard,brutal,turtle",
        "--eps", "0.15",
        "--eps-army", "0.10",
        "--games", str(args.bc_games),
        "--seed-base", "1",
        "--daemons", args.daemons,
        "--recycle-every", "50",
        "--out", "data/scripted_v2",
    ])

    # 2. Train initial multi-bot BC policy (bc_s0)
    run_cmd([
        sys.executable, "-m", "ra_rl.bc",
        "--data", "data/scripted_v2",
        "--out", "runs/phase2_5/bc_s0",
        "--epochs", "25",
    ])

    # 3. Record 1 round of multi-bot DAgger (seeds 30001..31200)
    run_cmd([
        sys.executable, "-m", "ra_rl.record",
        "--teacher", "scripted",
        "--executor", "net:runs/phase2_5/bc_s0/model.pt",
        "--bots", "hard,brutal,turtle",
        "--games", str(args.dagger_games),
        "--seed-base", "30001",
        "--daemons", args.daemons,
        "--recycle-every", "50",
        "--out", "data/dagger1_v2",
    ])

    # 4. Train final supervised anchor model (bc_s3)
    run_cmd([
        sys.executable, "-m", "ra_rl.bc",
        "--data", "data/scripted_v2,data/dagger1_v2",
        "--out", "runs/phase2_5/bc_s3",
        "--epochs", "25",
    ])

    # 5. Verify multi-bot gates on selection seeds
    gates_dir = Path("runs/phase2_5/gates")
    eval_gate(args.daemons, "net:runs/phase2_5/bc_s3/model.pt", "hard", args.gate_games, 0.55, gates_dir / "hard.jsonl")
    eval_gate(args.daemons, "net:runs/phase2_5/bc_s3/model.pt", "brutal", args.gate_games, 0.45, gates_dir / "brutal.jsonl")
    eval_gate(args.daemons, "net:runs/phase2_5/bc_s3/model.pt", "turtle", args.gate_games, 0.55, gates_dir / "turtle.jsonl")


def run_phase3(args) -> None:
    bc_ckpt = Path("runs/phase2_5/bc_s3/model.pt")
    if not bc_ckpt.exists():
        raise SystemExit(f"Missing Phase 2.5 anchor checkpoint: {bc_ckpt}")

    run_cmd([
        sys.executable, "-m", "ra_rl.ppo",
        "--init", str(bc_ckpt),
        "--ref", str(bc_ckpt),
        "--reset-critic",
        "--out", "runs/phase3/ppo_v2",
        "--iterations", str(args.p3_iters),
        "--games-per-iter", str(args.games_per_iter),
        "--daemons", args.daemons,
    ])


def run_phase4(args) -> None:
    bc_ckpt = Path("runs/phase2_5/bc_s3/model.pt")
    p3_best = Path("runs/phase3/ppo_v2/checkpoint_best.pt")
    use_p3 = args.use_p3_init and p3_best.exists()
    init_ckpt = p3_best if use_p3 else bc_ckpt
    if not init_ckpt.exists():
        raise SystemExit(f"Missing starting checkpoint: {init_ckpt}")

    cmd = [
        sys.executable, "-m", "ra_rl.league",
        "--init", str(init_ckpt),
        "--ref", str(bc_ckpt if bc_ckpt.exists() else init_ckpt),
        "--out", "runs/phase4/league_v2",
        "--iterations", str(args.p4_iters),
        "--games-per-iter", str(args.games_per_iter),
        "--critic-warmup-iters", str(args.critic_warmup_iters),
        "--kl-coef", str(args.kl_coef),
        "--ent-coef", str(args.ent_coef),
        "--ent-min", str(args.ent_min),
        "--daemons", args.daemons,
    ]
    if not use_p3:
        cmd.append("--reset-critic")
    run_cmd(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("all", "phase2_5", "phase3", "phase4"), default="all")
    ap.add_argument("--daemons", default=DEFAULT_DAEMONS)
    ap.add_argument("--bc-games", type=int, default=1800)
    ap.add_argument("--dagger-games", type=int, default=1200)
    ap.add_argument("--gate-games", type=int, default=60)
    ap.add_argument("--p3-iters", type=int, default=80)
    ap.add_argument("--p4-iters", type=int, default=80)
    ap.add_argument("--games-per-iter", type=int, default=48)
    ap.add_argument("--critic-warmup-iters", type=int, default=6)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--ent-min", type=float, default=0.001)
    ap.add_argument("--use-p3-init", action="store_true", help="initialize Phase 4 from Phase 3 best instead of bc_s3")
    args = ap.parse_args()

    if args.stage in ("all", "phase2_5"):
        run_phase2_5(args)
    if args.stage == "phase3":
        run_phase3(args)
    if args.stage in ("all", "phase4"):
        run_phase4(args)


if __name__ == "__main__":
    main()
