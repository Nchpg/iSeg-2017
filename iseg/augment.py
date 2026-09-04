"""Augmentation of 2.5D slices.

With 8 training subjects, this is where generalisation is won or lost.
Channels are ordered by modality: the first `context` ones are the T1
slices, the next ones the T2 slices. Intensity transforms are therefore
applied per group, so they stay consistent across neighbouring slices of
the same volume.
"""

import numpy as np
import torch
import torch.nn.functional as F


class Augment:
    def __init__(self, context=5, n_modalities=2, p_flip=0.5, max_rotation=10.0,
                 max_shift=0.10, max_scale=0.10, bias_strength=0.30,
                 gamma_range=(0.7, 1.5), noise_sigma=0.05, seed=None):
        self.context = context
        self.n_modalities = n_modalities
        self.p_flip = p_flip
        self.max_rotation = max_rotation
        self.max_shift = max_shift
        self.max_scale = max_scale
        self.bias_strength = bias_strength
        self.gamma_range = gamma_range
        self.noise_sigma = noise_sigma
        self.rng = np.random.default_rng(seed)

    def _affine(self, x, target):
        """Rotation + translation + zoom, in a single interpolation."""
        angle = np.deg2rad(self.rng.uniform(-self.max_rotation, self.max_rotation))
        scale = 1.0 + self.rng.uniform(-self.max_scale, self.max_scale)
        tx, ty = self.rng.uniform(-self.max_shift, self.max_shift, size=2)

        cos, sin = np.cos(angle) / scale, np.sin(angle) / scale
        theta = torch.tensor([[[cos, -sin, tx], [sin, cos, ty]]], dtype=torch.float32)

        img = torch.from_numpy(x).unsqueeze(0)
        lab = torch.from_numpy(target).to(torch.float32)[None, None]

        grid = F.affine_grid(theta, img.shape, align_corners=False)
        img = F.grid_sample(img, grid, mode="bilinear",
                            padding_mode="border", align_corners=False)
        # Nearest neighbour on the label: bilinear interpolation would
        # invent intermediate classes that do not exist.
        lab = F.grid_sample(lab, grid, mode="nearest",
                            padding_mode="zeros", align_corners=False)
        return img[0].numpy(), lab[0, 0].numpy().astype(np.int64)

    def _bias_field(self, x):
        """Smooth multiplicative inhomogeneity, drawn at low resolution.

        Simulates what N4 would have corrected: the network learns to
        ignore these variations rather than having them removed during
        preprocessing.
        """
        h, w = x.shape[-2:]
        for m in range(self.n_modalities):
            coarse = torch.from_numpy(
                self.rng.normal(0, self.bias_strength, size=(1, 1, 4, 4))
            ).to(torch.float32)
            field = F.interpolate(coarse, size=(h, w), mode="bicubic",
                                  align_corners=False)[0, 0].numpy()
            sl = slice(m * self.context, (m + 1) * self.context)
            x[sl] = x[sl] * np.exp(field)
        return x

    def _gamma(self, x):
        """Gamma correction, applied per modality over a [0, 1] range."""
        for m in range(self.n_modalities):
            sl = slice(m * self.context, (m + 1) * self.context)
            block = x[sl]
            lo, hi = block.min(), block.max()
            if hi - lo < 1e-6:
                continue
            g = self.rng.uniform(*self.gamma_range)
            x[sl] = ((block - lo) / (hi - lo)) ** g * (hi - lo) + lo
        return x

    def __call__(self, x, target):
        x = np.ascontiguousarray(x, dtype=np.float32)
        target = np.ascontiguousarray(target)

        if self.rng.random() < self.p_flip:
            # axis -2 = the x axis of the coronal plane, so left/right
            x = np.ascontiguousarray(x[:, ::-1, :])
            target = np.ascontiguousarray(target[::-1, :])

        x, target = self._affine(x, target)
        x = self._bias_field(x)
        x = self._gamma(x)
        if self.noise_sigma > 0:
            x = x + self.rng.normal(0, self.noise_sigma, size=x.shape).astype(np.float32)

        return x, target
