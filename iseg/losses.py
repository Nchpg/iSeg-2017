"""Fonction de cout et metriques.

Dice + entropie croisee a parts egales : l'entropie croisee donne des
gradients stables au demarrage, quand le Dice est encore proche de zero
et ne signale pas grand-chose ; le Dice optimise directement la metrique
d'evaluation et ignore la taille des classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice(logits, target, n_classes=4, eps=1e-6):
    """Dice differentiable, une valeur par tissu : (LCR, GM, WM).

    Le fond est ecarte : le segmenter est trivial et l'inclure gonflerait
    le score sans rien mesurer d'utile.
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
    """Dice par classe sur un volume entier.

    Par volume et non par coupe : une moyenne des Dice coupe par coupe
    donne un chiffre different, incomparable a celui du challenge.
    """
    out = []
    for c in range(1, n_classes):
        p, t = pred == c, target == c
        inter = (p & t).sum()
        denom = p.sum() + t.sum()
        out.append(1.0 if denom == 0 else 2.0 * inter / denom)
    return out
