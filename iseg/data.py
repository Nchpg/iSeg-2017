"""Loading, preprocessing and 2.5D dataset for iSeg-2017.

Slices are coronal and cropped to 144 x 144, a window that covers the
brain bounding box of the 10 subjects with 13 to 17 voxels of margin and
is divisible by 16: no resize, no padding. No N4 correction: SimpleITK
does not ship in a browser, augmentation simulates bias fields instead.
"""

from pathlib import Path

import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

# Crop window in the coronal plane: (x, z)
CROP_X = (0, 144)
CROP_Z = (72, 216)

# Challenge label values -> class indices
LABEL_MAP = {0: 0, 10: 1, 150: 2, 250: 3}
TISSUE_NAMES = ["CSF", "GM", "WM"]

# A slice enters training if at least this fraction of its voxels is
# labelled (drops the nearly empty slices at both ends).
MIN_OCCUPANCY = 0.02


def _read(path):
    """iSeg files are 4D with a singleton last dimension."""
    return np.squeeze(np.asarray(nib.load(path).dataobj))


def zscore(vol, mask):
    """Z-score computed on the brain alone: including the background would
    crush the statistic under millions of empty voxels."""
    vals = vol[mask]
    return (vol - vals.mean()) / (vals.std() + 1e-8)


def preprocess_subject(root, subject, with_label=True):
    """Load one subject, normalise and crop.

    Returns t1, t2 as float32 and the label as uint8 (0-3), shaped
    (144, Y, 144). The label is None for test subjects, which have no
    public annotation.
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
        # The crop must not cut through any labelled voxel.
        outside = raw.copy()
        outside[xs, :, zs] = 0
        if (outside > 0).any():
            raise ValueError(
                f"subject {subject}: {(outside > 0).sum()} labelled voxels outside "
                f"the crop window {CROP_X} x {CROP_Z}")
        lut = np.zeros(raw.max() + 1, dtype=np.uint8)
        for src, dst in LABEL_MAP.items():
            lut[src] = dst
        label = lut[raw][xs, :, zs]

    return t1[xs, :, zs], t2[xs, :, zs], label


def build_cache(raw_dir, cache_dir, subjects, with_label=True):
    """Preprocess once and write one .npz per subject: re-reading and
    re-normalising every epoch would dominate the cost of training."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for s in subjects:
        out = cache_dir / f"subject-{s}.npz"
        if out.exists():
            # A complete cache removes the need for the .hdr/.img files,
            # which keeps the transfer to Colab small.
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
    """Stack `context` adjacent slices per modality around y.

    At the ends of the volume the edge slice is repeated: padding with
    zeros would create an artificial black border, misleading the network.
    """
    half = context // 2
    n = vols["t1"].shape[1]
    idx = np.clip(np.arange(y - half, y + half + 1), 0, n - 1)
    return np.concatenate([vols[m][:, idx, :].transpose(1, 0, 2)
                           for m in modalities], axis=0)


class ISegSlices(Dataset):
    """2.5D coronal slices.

    Input  : (context * n_modalities, 144, 144)
    Target : (144, 144) integers 0-3, for the centre slice only
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
