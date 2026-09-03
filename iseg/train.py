"""Entrainement et validation croisee.

Le decoupage se fait PAR SUJET, jamais par coupe. Melanger toutes les
coupes puis en tirer 20 % au hasard mettrait la coupe 128 d'un sujet en
entrainement et la 129 en validation : elles sont quasi identiques, le
Dice de validation grimperait a 0,97 et ne voudrait plus rien dire.

Avec seulement 10 sujets, un unique decoupage donne un chiffre trop
dependant du hasard : on entraine 5 blocs de 8 sujets / 2 sujets et on
reporte moyenne et ecart-type.
"""

import argparse
import json
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
N_FOLDS = 5


def folds(subjects=SUBJECTS, n_folds=N_FOLDS):
    """5 blocs de 2 sujets de validation."""
    chunks = np.array_split(np.array(subjects), n_folds)
    return [(sorted(set(subjects) - set(c.tolist())), sorted(c.tolist())) for c in chunks]


@torch.no_grad()
def predict_volume(net, vols, context, modalities, device, batch_size=16):
    """Segmente toutes les coupes coronales et reassemble le volume."""
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
    """Dice 3D par tissu, moyenne sur les sujets de validation."""
    scores = []
    for s in subjects:
        vols = load_cached(cache_dir, s)
        pred = predict_volume(net, vols, context, modalities, device)
        scores.append(dice_volume(pred, vols["label"]))
    return np.array(scores).mean(axis=0)


def train_fold(args, fold, device):
    train_ids, val_ids = folds()[fold]
    modalities = ["t1", "t2"] if args.modalities == "t1t2" else ["t1"]

    aug = Augment(context=args.context, n_modalities=len(modalities), seed=args.seed + fold)
    train_set = ISegSlices(args.cache, train_ids, args.context, modalities, augment=aug)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"),
                        drop_last=True)

    net = models.build(args.variant, train_set.in_channels).to(device)
    criterion = DiceCELoss()
    optim = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    n_params = models.count_parameters(net)
    print(f"\n=== bloc {fold} | entrainement {train_ids} | validation {val_ids}")
    print(f"    {args.variant} : {n_params:,} parametres "
          f"({models.model_size_mb(net):.2f} Mo float32)")
    print(f"    {len(train_set)} coupes d'entrainement, {args.epochs} epoques")

    history, best = [], {"dice_mean": -1.0}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        train_loss = total / len(train_set)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            dice = evaluate(net, args.cache, val_ids, args.context, modalities, device)
            entry = {"epoch": epoch, "loss": train_loss,
                     **{f"dice_{n}": float(d) for n, d in zip(TISSUE_NAMES, dice)},
                     "dice_mean": float(dice.mean()), "seconds": time.time() - t0}
            history.append(entry)
            print(f"    ep {epoch:>3}  loss {train_loss:.4f}  "
                  + "  ".join(f"{n} {d:.4f}" for n, d in zip(TISSUE_NAMES, dice))
                  + f"  moyenne {dice.mean():.4f}  ({entry['seconds']:.0f}s)")

            # Selection sur le Dice de validation, pas sur la perte.
            if entry["dice_mean"] > best["dice_mean"]:
                best = entry
                torch.save({"state_dict": net.state_dict(), "variant": args.variant,
                            "in_channels": train_set.in_channels, "context": args.context,
                            "modalities": modalities, "fold": fold, "metrics": entry},
                           out_dir / f"{args.variant}_fold{fold}.pt")

    result = {"fold": fold, "train": train_ids, "val": val_ids, "variant": args.variant,
              "modalities": modalities, "context": args.context, "params": n_params,
              "best": best, "history": history}
    (out_dir / f"{args.variant}_fold{fold}.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description="Entrainement U-Net 2.5D sur iSeg-2017")
    p.add_argument("--raw", default="iSeg-2017-Training", help="dossier des .hdr/.img")
    p.add_argument("--cache", default="cache", help="dossier des volumes pretraites")
    p.add_argument("--out", default="runs", help="dossier des poids et journaux")
    p.add_argument("--variant", default="standard", choices=list(models.VARIANTS))
    p.add_argument("--modalities", default="t1t2", choices=["t1t2", "t1"],
                   help="ablation : t1 seul pour mesurer l'apport reel du T2")
    p.add_argument("--context", type=int, default=5, help="nombre de coupes 2.5D (impair)")
    p.add_argument("--fold", type=int, default=-1, help="-1 pour les 5 blocs")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.context % 2 == 0:
        p.error("--context doit etre impair (coupe centrale + voisines symetriques)")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"peripherique : {device}")

    build_cache(args.raw, args.cache, SUBJECTS, with_label=True)

    targets = range(N_FOLDS) if args.fold < 0 else [args.fold]
    results = [train_fold(args, f, device) for f in targets]

    dices = np.array([[r["best"][f"dice_{n}"] for n in TISSUE_NAMES] for r in results])
    print(f"\n=== {args.variant} | {args.modalities} | contexte {args.context} "
          f"| {results[0]['params']:,} parametres")
    for i, n in enumerate(TISSUE_NAMES):
        print(f"    {n:<4} Dice {dices[:, i].mean():.4f} +/- {dices[:, i].std():.4f}")
    print(f"    moyenne  {dices.mean():.4f} +/- {dices.mean(axis=1).std():.4f}"
          f"   (sur {len(results)} bloc(s))")

    summary = {"variant": args.variant, "modalities": args.modalities,
               "context": args.context, "params": results[0]["params"],
               "per_tissue_mean": dict(zip(TISSUE_NAMES, dices.mean(axis=0).tolist())),
               "per_tissue_std": dict(zip(TISSUE_NAMES, dices.std(axis=0).tolist())),
               "dice_mean": float(dices.mean()), "folds": [r["fold"] for r in results]}
    tag = f"{args.variant}_{args.modalities}_ctx{args.context}"
    Path(args.out, f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
