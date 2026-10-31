"""Behavior cloning: train the commander network on recorded games.

    python -m ra_rl.bc --data data/scripted_v2 --out runs/phase2_5/bc_s0
    python -m ra_rl.bc --data data/scripted_v2,data/dagger1_v2 --out runs/phase2_5/bc_s3

The validation split is by game (every 20th seed), never by row. The checkpoint
with the lowest validation loss is kept as <out>/model.pt; per-epoch metrics go
to <out>/metrics.jsonl.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ra_rl.features import FEATURE_VERSION, GLOBAL_DIM
from ra_rl.model import CommanderNet

ARRAYS = ("prod_g", "prod_queue", "prod_mask", "prod_y", "army_g", "army_cand", "army_mask", "army_y")


def load(dirs: list[str], val_every: int = 20) -> tuple[dict, dict, dict]:
    """Concatenate every game under `dirs` into a train and a validation set."""
    parts: dict[str, dict[str, list]] = {"train": {k: [] for k in ARRAYS}, "val": {k: [] for k in ARRAYS}}
    stats = {"games": 0, "outcomes": {}}
    for d in dirs:
        for f in sorted(Path(d.strip()).glob("g*.npz")):
            if f.name.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            if len(z["prod_g"]) and z["prod_g"].shape[1] != GLOBAL_DIM:
                raise ValueError(
                    f"{f}: prod_g dim {z['prod_g'].shape[1]} != GLOBAL_DIM {GLOBAL_DIM} "
                    f"(expected FEATURE_VERSION={FEATURE_VERSION})"
                )
            split = "val" if meta["seed"] % val_every == 0 else "train"
            for k in ARRAYS:
                if len(z[k]):
                    parts[split][k].append(z[k])
            stats["games"] += 1
            stats["outcomes"][meta["outcome"]] = stats["outcomes"].get(meta["outcome"], 0) + 1

    def cat(p: dict[str, list]) -> dict[str, np.ndarray]:
        return {k: np.concatenate(v) for k, v in p.items() if v}

    return cat(parts["train"]), cat(parts["val"]), stats


def to_device(arrays: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in arrays.items():
        t = torch.from_numpy(v)
        if k.endswith(("_g", "_cand")):
            t = t.float()
        elif k.endswith(("_y", "_queue")):
            t = t.long()
        out[k] = t.to(device)
    return out


def losses(net: CommanderNet, d: dict[str, torch.Tensor], pi: torch.Tensor, ai: torch.Tensor) -> dict:
    out = {}
    if len(pi):
        logits = net.item_logits(d["prod_g"][pi], d["prod_queue"][pi], d["prod_mask"][pi])
        y = d["prod_y"][pi]
        out["prod_loss"] = F.cross_entropy(logits, y)
        out["prod_acc"] = (logits.argmax(-1) == y).float().mean()
    if len(ai):
        logits = net.army_logits(d["army_g"][ai], d["army_cand"][ai], d["army_mask"][ai])
        y = d["army_y"][ai]
        out["army_loss"] = F.cross_entropy(logits, y)
        out["army_acc"] = (logits.argmax(-1) == y).float().mean()
    return out


@torch.no_grad()
def evaluate(net: CommanderNet, d: dict[str, torch.Tensor], batch: int = 8192) -> dict:
    net.eval()
    sums: dict[str, float] = {}
    for head, n in (("prod", len(d.get("prod_y", []))), ("army", len(d.get("army_y", [])))):
        for s in range(0, n, batch):
            idx = torch.arange(s, min(n, s + batch), device=d[f"{head}_y"].device)
            empty = idx[:0]
            m = losses(net, d, idx if head == "prod" else empty, idx if head == "army" else empty)
            for k, v in m.items():
                sums[k] = sums.get(k, 0.0) + float(v) * len(idx)
        for k in (f"{head}_loss", f"{head}_acc"):
            if k in sums:
                sums[k] /= n
    net.train()
    return sums


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="comma-separated directories of recorded games")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--army-weight", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=20, help="games whose seed is a multiple of this validate")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    train_np, val_np, stats = load(args.data.split(","), args.val_every)
    if not {"prod_y", "army_y"} <= train_np.keys() & val_np.keys():
        raise SystemExit("need BUILD/TRAIN and ARMY rows in both splits: record more games or lower --val-every")
    train, val = to_device(train_np, device), to_device(val_np, device)
    n_prod, n_army = len(train["prod_y"]), len(train["army_y"])
    print(json.dumps({"games": stats["games"], "outcomes": stats["outcomes"], "train_prod": n_prod,
                      "train_army": n_army, "val_prod": len(val["prod_y"]), "val_army": len(val["army_y"]),
                      "load_s": round(time.perf_counter() - t0, 1), "device": device}), flush=True)

    net = CommanderNet(hidden=args.hidden).to(device)
    net.reset_critic()
    actor_params = list(net.actor_parameters())
    opt = torch.optim.AdamW(actor_params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, n_prod // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
    army_batch = max(1, args.batch * n_army // max(1, n_prod))

    best = float("inf")
    log = (out / "metrics.jsonl").open("w", encoding="utf-8")
    for epoch in range(args.epochs):
        perm = torch.randperm(n_prod, device=device)
        for s in range(steps_per_epoch):
            pi = perm[s * args.batch:(s + 1) * args.batch]
            ai = torch.randint(0, n_army, (army_batch,), device=device)
            m = losses(net, train, pi, ai)
            loss = m["prod_loss"] + args.army_weight * m["army_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        v = evaluate(net, val)
        total = v["prod_loss"] + args.army_weight * v["army_loss"]
        row = {"epoch": epoch + 1, **{f"val_{k}": round(x, 4) for k, x in v.items()},
               "train_prod_loss": round(m["prod_loss"].item(), 4), "train_army_loss": round(m["army_loss"].item(), 4),
               "s": round(time.perf_counter() - t0, 1)}
        if total < best:
            best = total
            net.save(str(out / "model.pt"), epoch=epoch + 1, val=v, data=args.data)
            row["saved"] = True
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
    log.close()


if __name__ == "__main__":
    main()
