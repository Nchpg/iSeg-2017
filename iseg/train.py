"""Training of the 2.5D U-Net.

The split is by subject, never by slice: two neighbouring slices of the
same volume are nearly identical, and spreading them across train and
validation would inflate Dice without measuring anything. Two subjects
are held out, and the reported Dice is computed on their reassembled
volumes.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import model as models
from .augment import Augment
from .data import ISegSlices, TISSUE_NAMES, build_cache, load_cached, stack_context
from .losses import DiceCELoss, dice_volume

SUBJECTS = list(range(1, 11))
VAL_SUBJECTS = [1, 2]
TRAIN_SUBJECTS = [s for s in SUBJECTS if s not in VAL_SUBJECTS]


@torch.no_grad()
def predict_volume(net, vols, context, modalities, device, batch_size=16):
    """Segment every coronal slice and reassemble the volume."""
    net.eval()
    n = vols["t1"].shape[1]
    out = np.empty((vols["t1"].shape[0], n, vols["t1"].shape[2]), dtype=np.uint8)

    for start in range(0, n, batch_size):
        ys = range(start, min(start + batch_size, n))
        batch = np.stack([stack_context(vols, y, context, modalities) for y in ys])
        logits = net(torch.from_numpy(batch).float().to(device))
        out[:, list(ys), :] = logits.argmax(1).cpu().numpy().transpose(1, 0, 2)
    return out


def evaluate(net, cache_dir, subjects, context, modalities, device):
    """3D Dice per tissue, averaged over the validation subjects."""
    scores = []
    for s in subjects:
        vols = load_cached(cache_dir, s)
        pred = predict_volume(net, vols, context, modalities, device)
        scores.append(dice_volume(pred, vols["label"]))
    return np.array(scores).mean(axis=0)


def main():
    p = argparse.ArgumentParser(description="2.5D U-Net training on iSeg-2017")
    p.add_argument("--raw", default="iSeg-2017-Training", help="folder holding the .hdr/.img")
    p.add_argument("--cache", default="cache", help="folder holding the preprocessed volumes")
    p.add_argument("--out", default="runs", help="folder holding the weights")
    p.add_argument("--variant", default="separable", choices=list(models.VARIANTS))
    p.add_argument("--modalities", default="t1t2", choices=["t1t2", "t1"])
    p.add_argument("--context", type=int, default=5, help="number of 2.5D slices (odd)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.context % 2 == 0:
        p.error("--context must be odd (centre slice + symmetric neighbours)")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_cache(args.raw, args.cache, SUBJECTS, with_label=True)

    modalities = ["t1", "t2"] if args.modalities == "t1t2" else ["t1"]
    aug = Augment(context=args.context, n_modalities=len(modalities), seed=args.seed)
    train_set = ISegSlices(args.cache, TRAIN_SUBJECTS, args.context, modalities, augment=aug)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"),
                        drop_last=True)

    net = models.build(args.variant, train_set.in_channels).to(device)
    criterion = DiceCELoss()
    optim = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = out_dir / f"{args.variant}.pt"

    print(f"device {device} | {args.variant}: "
          f"{models.count_parameters(net):,} parameters")
    print(f"train {TRAIN_SUBJECTS} | validation {VAL_SUBJECTS}")
    print(f"{len(train_set)} slices, {args.epochs} epochs\n")

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0, total = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = criterion(net(x), y)
            loss.backward()
            optim.step()
            total += loss.item() * x.size(0)
        sched.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            dice = evaluate(net, args.cache, VAL_SUBJECTS, args.context, modalities, device)
            print(f"  ep {epoch:>3}  loss {total / len(train_set):.4f}  "
                  + "  ".join(f"{n} {d:.4f}" for n, d in zip(TISSUE_NAMES, dice))
                  + f"  mean {dice.mean():.4f}  ({time.time() - t0:.0f}s)")

            # Selected on validation Dice, not on the loss.
            if dice.mean() > best:
                best = dice.mean()
                torch.save({"state_dict": net.state_dict(), "variant": args.variant,
                            "in_channels": train_set.in_channels, "context": args.context,
                            "modalities": modalities,
                            "dice": dict(zip(TISSUE_NAMES, dice.tolist()))}, weights)

    print(f"\nbest Dice {best:.4f} -> {weights}")


if __name__ == "__main__":
    main()
