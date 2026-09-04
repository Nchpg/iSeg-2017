"""Chargement, pretraitement et dataset 2.5D pour iSeg-2017.

Les coupes sont coronales et recadrees a 144 x 144, une fenetre qui
couvre la boite englobante du cerveau des 10 sujets avec 13 a 17 voxels
de marge, et qui est divisible par 16 : ni redimensionnement, ni
remplissage. Pas de correction N4 : SimpleITK ne s'embarque pas dans un
navigateur, l'augmentation simule des champs de biais a la place.
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
TISSUE_NAMES = ["LCR", "GM", "WM"]

# Une coupe entre dans l'entrainement si au moins cette fraction de ses
# voxels est annotee (ecarte les coupes quasi vides des extremites).
MIN_OCCUPANCY = 0.02


def _read(path):
    """Les fichiers iSeg sont en 4D avec une derniere dimension singleton."""
    return np.squeeze(np.asarray(nib.load(path).dataobj))


def zscore(vol, mask):
    """Z-score calcule sur le cerveau seul : inclure le fond ecraserait
    la statistique sous des millions de voxels vides."""
    vals = vol[mask]
    return (vol - vals.mean()) / (vals.std() + 1e-8)


def preprocess_subject(root, subject, with_label=True):
    """Charge un sujet, normalise et recadre.

    Retourne t1, t2 en float32 et le label en uint8 (0-3), de forme
    (144, Y, 144). Le label vaut None pour les sujets de test, qui n'ont
    pas d'annotation publique.
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
    """Pretraite une fois pour toutes et ecrit un .npz par sujet : relire
    et renormaliser a chaque epoque dominerait le cout de l'entrainement."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for s in subjects:
        out = cache_dir / f"subject-{s}.npz"
        if out.exists():
            # Un cache complet dispense d'avoir les .hdr/.img sous la main,
            # ce qui allege le transfert vers Colab.
            continue
        t1, t2, label = preprocess_subject(raw_dir, s, with_label)
        payload = {"t1": t1.astype(np.float16), "t2": t2.astype(np.float16)}
        if label is not None:
            payload["label"] = label
            payload["occupancy"] = (
                (label > 0).sum(axis=(0, 2)) / (label.shape[0] * label.shape[2])
            ).astype(np.float32)
        np.savez_compressed(out, **payload)


def load_cached(cache_dir, subject):
    d = np.load(Path(cache_dir) / f"subject-{subject}.npz")
    vols = {"t1": d["t1"].astype(np.float32), "t2": d["t2"].astype(np.float32)}
    if "label" in d:
        vols["label"] = d["label"]
        vols["occupancy"] = d["occupancy"]
    return vols


def stack_context(vols, y, context, modalities):
    """Empile `context` coupes adjacentes par modalite autour de y.

    Aux extremites du volume la coupe de bord est repliquee : remplir de
    zeros creerait un bord noir artificiel, trompeur pour le reseau.
    """
    half = context // 2
    n = vols["t1"].shape[1]
    idx = np.clip(np.arange(y - half, y + half + 1), 0, n - 1)
    return np.concatenate([vols[m][:, idx, :].transpose(1, 0, 2)
                           for m in modalities], axis=0)


class ISegSlices(Dataset):
    """Coupes coronales 2.5D.

    Entree : (context * n_modalites, 144, 144)
    Cible  : (144, 144) entiers 0-3, pour la coupe centrale uniquement
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
