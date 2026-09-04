"""Loss function and metrics.

Dice + cross-entropy in equal parts: cross-entropy gives stable
gradients at the start, when Dice is still near zero and signals little;
Dice optimises the evaluation metric directly and ignores class size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice(logits, target, n_classes=4, eps=1e-6):
    """Differentiable Dice, one value per tissue: (CSF, GM, WM).

    Background is left out: segmenting it is trivial, and including it
    would inflate the score without measuring anything useful.
    """
    probs = F.softmax(logits, dim=1)[:, 1:]
    onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()[:, 1:]

    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    return (2 * inter + eps) / (denom + eps)


class DiceCELoss(nn.Module):
    def __init__(self, n_classes=4, dice_weight=0.5):
        super().__init__()
        self.n_classes = n_classes
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        dice = soft_dice(logits, target, self.n_classes).mean()
        return self.dice_weight * (1 - dice) + (1 - self.dice_weight) * self.ce(logits, target)


@torch.no_grad()
def dice_volume(pred, target, n_classes=4):
    """Per-class Dice over a whole volume.

    Per volume rather than per slice: averaging slice-wise Dice gives a
    different number, not comparable to the challenge's.
    """
    out = []
    for c in range(1, n_classes):
        p, t = pred == c, target == c
        inter = (p & t).sum()
        denom = p.sum() + t.sum()
        out.append(1.0 if denom == 0 else 2.0 * inter / denom)
    return out
