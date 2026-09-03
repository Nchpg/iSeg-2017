"""Chargement, pretraitement et dataset 2.5D pour iSeg-2017.

Choix de conception :

- Orientation coronale (axe 1 du volume). La symetrie gauche/droite du
  cerveau est visible dans le plan, ce qui donne un repere anatomique au
  reseau et rend le flip horizontal legitime comme augmentation.

- Recadrage fixe 144 x 144 dans le plan. L'exploration a montre que la
  boite englobante du cerveau, unie sur les 10 sujets d'entrainement,
  tient dans x[9:136] et z[85:199]. L'axe x (144 voxels) est conserve
  entier : il est deja divisible par 16 et la marge y est trop mince
  pour couper. L'axe z est ramene a [72:216], ce qui laisse 13 et 17
  voxels de marge de part et d'autre du cerveau le plus etendu -- assez
  pour absorber la variabilite des sujets de test, sur lesquels on n'a
  aucune etiquette pour verifier. Coupes 144 x 144 : divisibles par 16,
  aucun redimensionnement, aucun remplissage.

- Pas de correction de champ de biais N4. SimpleITK ne s'embarque pas
  dans un navigateur ; on compense par une augmentation qui simule des
  inhomogeneites (voir augment.py). Arbitrage assume au profit d'un
  pretraitement reproductible a l'inference.
"""

from pathlib import Path

import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

# Fenetre de recadrage dans le plan coronal : (x, z)
CROP_X = (0, 144)
CROP_Z = (72, 216)

# Valeurs d'etiquette du challenge -> indices de classe
LABEL_MAP = {0: 0, 10: 1, 150: 2, 250: 3}
CLASS_NAMES = ["fond", "LCR", "GM", "WM"]
TISSUE_NAMES = ["LCR", "GM", "WM"]

# Une coupe entre dans l'entrainement si au moins cette fraction de ses
# voxels est annotee (ecarte les coupes quasi vides des extremites).
MIN_OCCUPANCY = 0.02


def _read(path):
    """Les fichiers iSeg sont en 4D avec une derniere dimension singleton."""
    return np.squeeze(np.asarray(nib.load(path).dataobj))


def zscore(vol, mask):
    """Z-score calcule sur le cerveau seul.

    Inclure le fond tirerait la moyenne vers zero : les millions de voxels
    vides ecraseraient la statistique et la normalisation serait inoperante.
    """
    vals = vol[mask]
    return (vol - vals.mean()) / (vals.std() + 1e-8)


def preprocess_subject(root, subject, with_label=True):
    """Charge un sujet, normalise et recadre.

    Retourne t1, t2 en float32 et le label en uint8 (0-3), de forme
    (128, Y, 128). Le label vaut None pour les sujets de test.
    """
    root = Path(root)
    t1 = _read(root / f"subject-{subject}-T1.hdr").astype(np.float32)
    t2 = _read(root / f"subject-{subject}-T2.hdr").astype(np.float32)

    brain = t1 > 0
    t1 = zscore(t1, brain)
    t2 = zscore(t2, brain)

    xs, zs = slice(*CROP_X), slice(*CROP_Z)

    label = None
    if with_label:
        raw = _read(root / f"subject-{subject}-label.hdr").astype(np.int16)
        # Le recadrage ne doit couper aucun voxel annote.
        outside = raw.copy()
        outside[xs, :, zs] = 0
        if (outside > 0).any():
            raise ValueError(
                f"sujet {subject} : {(outside > 0).sum()} voxels annotes hors "
                f"de la fenetre de recadrage {CROP_X} x {CROP_Z}")
        lut = np.zeros(raw.max() + 1, dtype=np.uint8)
        for src, dst in LABEL_MAP.items():
            lut[src] = dst
        label = lut[raw][xs, :, zs]

    return t1[xs, :, zs], t2[xs, :, zs], label


def build_cache(raw_dir, cache_dir, subjects, with_label=True):
    """Pretraite une fois pour toutes et ecrit un .npz par sujet.

    Evite de relire et renormaliser les volumes a chaque epoque, ce qui
    domine largement le cout d'un entrainement 2D.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for s in subjects:
        out = cache_dir / f"subject-{s}.npz"
        if out.exists():
            written.append(out)
            continue
        t1, t2, label = preprocess_subject(raw_dir, s, with_label)
        payload = {"t1": t1.astype(np.float16), "t2": t2.astype(np.float16)}
        if label is not None:
            payload["label"] = label
            # Fraction de voxels annotes par coupe coronale.
            payload["occupancy"] = (
                (label > 0).sum(axis=(0, 2)) / (label.shape[0] * label.shape[2])
            ).astype(np.float32)
        np.savez_compressed(out, **payload)
        written.append(out)
    return written


def load_cached(cache_dir, subject):
    d = np.load(Path(cache_dir) / f"subject-{subject}.npz")
    vols = {"t1": d["t1"].astype(np.float32), "t2": d["t2"].astype(np.float32)}
    if "label" in d:
        vols["label"] = d["label"]
        vols["occupancy"] = d["occupancy"]
    return vols


def stack_context(vols, y, context, modalities):
    """Assemble l'entree 2.5D pour la coupe coronale d'indice y.

    Empile `context` coupes adjacentes (context // 2 de chaque cote) pour
    chaque modalite. Aux extremites du volume on replique la coupe de
    bord : remplir de zeros creerait un bord noir artificiel, information
    trompeuse pour le reseau.
    """
    half = context // 2
    n = vols["t1"].shape[1]
    idx = np.clip(np.arange(y - half, y + half + 1), 0, n - 1)
    return np.concatenate([vols[m][:, idx, :].transpose(1, 0, 2)
                           for m in modalities], axis=0)


class ISegSlices(Dataset):
    """Coupes coronales 2.5D.

    Entree  : (context * n_modalites, 128, 128)
    Cible   : (128, 128) entiers 0-3, pour la coupe centrale uniquement
    """

    def __init__(self, cache_dir, subjects, context=5, modalities=("t1", "t2"),
                 augment=None, min_occupancy=MIN_OCCUPANCY):
        self.context = context
        self.modalities = list(modalities)
        self.augment = augment
        self.volumes = {s: load_cached(cache_dir, s) for s in subjects}

        self.index = [(s, int(y))
                      for s in subjects
                      for y in np.where(self.volumes[s]["occupancy"] > min_occupancy)[0]]

    @property
    def in_channels(self):
        return self.context * len(self.modalities)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, y = self.index[i]
        vols = self.volumes[s]
        x = stack_context(vols, y, self.context, self.modalities)
        target = vols["label"][:, y, :].astype(np.int64)
        if self.augment is not None:
            x, target = self.augment(x, target)
        return x.astype(np.float32), target
