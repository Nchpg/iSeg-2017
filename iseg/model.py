"""U-Net 2.5D et ses variantes frugales.

Le reseau est un U-Net 2D ordinaire : ce sont les canaux d'entree qui
portent la troisieme dimension (`context` coupes adjacentes par
modalite). Consequence directe pour le deploiement : aucune convolution
3D, donc un modele que les runtimes mobiles (ONNX Runtime, TFLite,
Core ML) savent accelerer et quantifier.

Trois variantes, pour tracer la courbe Dice vs frugalite :

  standard   convolutions 3x3 pleines            1 943 636 params
  separable  depthwise 3x3 + pointwise 1x1         386 782 params
  tiny       separable, plus etroit et moins       26 974 params
             profond (base=8, depth=3)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Deux convolutions 3x3, chacune suivie de BatchNorm et ReLU."""

    def __init__(self, in_ch, out_ch, separable=False):
        super().__init__()
        conv = _separable_conv if separable else _standard_conv
        self.block = nn.Sequential(
            *conv(in_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            *conv(out_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def _standard_conv(in_ch, out_ch):
    return [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)]


def _separable_conv(in_ch, out_ch):
    """Convolution separable en profondeur.

    Une 3x3 par canal d'entree (depthwise), puis une 1x1 qui melange les
    canaux (pointwise). Cout : 9*C + C*C' au lieu de 9*C*C'.
    """
    return [
        nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
    ]


class UNet25D(nn.Module):
    def __init__(self, in_channels, n_classes=4, base=16, depth=4, separable=False):
        super().__init__()
        chans = [base * 2 ** i for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for c in chans:
            self.encoders.append(ConvBlock(prev, c, separable))
            prev = c

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(prev, prev * 2, separable)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = prev * 2
        for c in reversed(chans):
            self.ups.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            # concatenation avec la connexion de saut -> 2 * c canaux
            self.decoders.append(ConvBlock(2 * c, c, separable))
            prev = c

        self.head = nn.Conv2d(prev, n_classes, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            # Les connexions de saut sont indispensables ici : la frontiere
            # cortex / substance blanche se joue sur 2-3 mm, et le decodeur
            # ne peut pas retrouver ce detail perdu au sous-echantillonnage.
            x = dec(torch.cat([skip, x], dim=1))

        return self.head(x)


VARIANTS = {
    "standard":  dict(base=16, depth=4, separable=False),
    "separable": dict(base=16, depth=4, separable=True),
    "tiny":      dict(base=8,  depth=3, separable=True),
}


def build(variant, in_channels, n_classes=4):
    if variant not in VARIANTS:
        raise ValueError(f"variante inconnue : {variant} (choix : {list(VARIANTS)})")
    return UNet25D(in_channels, n_classes, **VARIANTS[variant])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    """Taille en float32, 4 octets par parametre."""
    return count_parameters(model) * 4 / 1024 ** 2
