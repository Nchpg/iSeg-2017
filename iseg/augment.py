"""Augmentation pour les coupes 2.5D.

Avec 8 sujets d'entrainement, c'est ici que se joue la generalisation.

Deux familles, traitees differemment :

- geometriques (flip, rotation, translation, zoom) : appliquees a
  l'identique aux canaux d'entree ET a l'etiquette, en interpolation au
  plus proche voisin pour cette derniere. Une interpolation bilineaire
  sur les etiquettes creerait des classes intermediaires inexistantes.

- intensite (champ de biais, gamma, bruit) : appliquees a l'image seule.
  Le champ de biais est le point critique : il remplace la correction N4
  ecartee du pretraitement.

Les canaux sont ordonnes par modalite : les `context` premiers canaux
sont les coupes T1, les suivants les coupes T2. Les transformations
d'intensite sont appliquees par groupe de modalite, pour rester
coherentes entre coupes voisines d'un meme volume.
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

    # ------------------------------------------------------- geometrie
    def _affine(self, x, target):
        """Rotation + translation + zoom, en une seule interpolation."""
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
        lab = F.grid_sample(lab, grid, mode="nearest",
                            padding_mode="zeros", align_corners=False)
        return img[0].numpy(), lab[0, 0].numpy().astype(np.int64)

    # ------------------------------------------------------- intensite
    def _bias_field(self, x):
        """Inhomogeneite multiplicative lisse, tiree a basse resolution.

        Simule ce que N4 aurait corrige : le reseau apprend a ignorer ces
        variations plutot qu'on ne les retire au pretraitement.
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
        """Correction gamma, appliquee par modalite sur une plage [0, 1]."""
        for m in range(self.n_modalities):
            sl = slice(m * self.context, (m + 1) * self.context)
            block = x[sl]
            lo, hi = block.min(), block.max()
            if hi - lo < 1e-6:
                continue
            g = self.rng.uniform(*self.gamma_range)
            x[sl] = ((block - lo) / (hi - lo)) ** g * (hi - lo) + lo
        return x

    # ------------------------------------------------------------ appel
    def __call__(self, x, target):
        x = np.ascontiguousarray(x, dtype=np.float32)
        target = np.ascontiguousarray(target)

        if self.rng.random() < self.p_flip:
            # axe -2 = axe x du plan coronal, donc gauche/droite
            x = np.ascontiguousarray(x[:, ::-1, :])
            target = np.ascontiguousarray(target[::-1, :])

        x, target = self._affine(x, target)
        x = self._bias_field(x)
        x = self._gamma(x)
        if self.noise_sigma > 0:
            x = x + self.rng.normal(0, self.noise_sigma, size=x.shape).astype(np.float32)

        # Le float32 est garanti tout au long de la chaine ; la conversion
        # finale est faite une seule fois par ISegSlices.__getitem__.
        return x, target
