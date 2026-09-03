"""Fonction de cout et metriques.

Dice + entropie croisee, a parts egales. L'entropie croisee fournit des
gradients stables en debut d'entrainement, quand le Dice est encore
proche de zero et donne peu de signal ; le Dice optimise directement la
metrique d'evaluation et se moque de la taille des classes.

L'exploration a montre un desequilibre modere entre tissus a l'interieur
du cerveau (LCR 22 %, GM 47 %, WM 31 %, soit un facteur 2,2). Pas besoin
de ponderation agressive : le terme Dice suffit a mettre les trois tissus
sur un pied d'egalite.

Le fond est exclu du terme Dice : le segmenter est trivial et l'inclure
gonflerait artificiellement le score.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice(logits, target, n_classes=4, ignore_background=True, eps=1e-6):
    """Dice differentiable, moyenne sur les classes retenues."""
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()

    start = 1 if ignore_background else 0
    probs, onehot = probs[:, start:], onehot[:, start:]

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
    """Dice par classe sur un volume entier, en numpy.

    Point de methode : le Dice se calcule par volume, pas par coupe. Une
    moyenne des Dice coupe par coupe donnerait un chiffre different et
    non comparable a la litterature du challenge.
    """
    out = []
    for c in range(1, n_classes):
        p, t = pred == c, target == c
        inter = (p & t).sum()
        denom = p.sum() + t.sum()
        out.append(1.0 if denom == 0 else 2.0 * inter / denom)
    return out
