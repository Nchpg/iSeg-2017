"""2.5D U-Net and its frugal variants.

An ordinary 2D U-Net: the input channels carry the third dimension
(`context` adjacent slices per modality). No 3D convolution, so mobile
runtimes can accelerate and quantize the model.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""

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
    """One 3x3 per input channel, then a 1x1 that mixes the channels:
    9*C + C*C' instead of 9*C*C'."""
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
            # concatenated with the skip connection -> 2 * c channels
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
            # The cortex / white matter boundary plays out over 2-3 mm:
            # without a skip connection the decoder cannot recover that
            # detail, lost during downsampling.
            x = dec(torch.cat([skip, x], dim=1))

        return self.head(x)


VARIANTS = {
    "standard":  dict(base=16, depth=4, separable=False),
    "separable": dict(base=16, depth=4, separable=True),
    "tiny":      dict(base=8,  depth=3, separable=True),
}


def build(variant, in_channels, n_classes=4):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant} (choices: {list(VARIANTS)})")
    return UNet25D(in_channels, n_classes, **VARIANTS[variant])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
