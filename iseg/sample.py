"""Build the demo sample loaded by the web page.

    python -m iseg.sample --subject 1 --out webdemo/sample.bin.gz

The original .hdr/.img weigh 28 MB per subject. Only what the app reads
is kept -- the crop window of data.py and the slices that contain brain
-- then compressed: about 2.5 MB.

Format produced, entirely little-endian:

    magic   4 bytes   "ISG1"
    x0 y0 z0          3 x int32   position of the block in the full volume
    nx ny nz          3 x int32   dimensions of the block
    fx fy fz          3 x int32   dimensions of the full volume
    T1                int16[nx*ny*nz]
    T2                int16[nx*ny*nz]

Voxels are laid out with x varying fastest, then y, then z, as in the
original Analyze files: the page can therefore drop them back into a
full-size volume as they are.
"""

import argparse
import gzip
import struct
from pathlib import Path

import numpy as np

from .data import CROP_X, CROP_Z, _read

MAGIC = b"ISG1"


def build(raw_dir, subject):
    t1 = _read(Path(raw_dir) / f"subject-{subject}-T1.hdr").astype(np.int16)
    t2 = _read(Path(raw_dir) / f"subject-{subject}-T2.hdr").astype(np.int16)
    if t1.shape != t2.shape:
        raise ValueError(f"T1 {t1.shape} and T2 {t2.shape} have different geometry")
    fx, fy, fz = t1.shape

    # bounds of the slices containing brain
    occupancy = (t1 > 0).sum(axis=(0, 2))
    useful = np.where(occupancy > 0)[0]
    y0, y1 = int(useful[0]), int(useful[-1]) + 1

    x0, x1 = CROP_X
    z0, z1 = CROP_Z
    block = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
    b1, b2 = t1[block], t2[block]
    nx, ny, nz = b1.shape

    header = MAGIC + struct.pack("<9i", x0, y0, z0, nx, ny, nz, fx, fy, fz)
    # Fortran order: x varies fastest, as in the original .img
    body = b1.tobytes(order="F") + b2.tobytes(order="F")
    return header + body, (nx, ny, nz), (fx, fy, fz)


def main():
    p = argparse.ArgumentParser(description="Demo sample for the web page")
    p.add_argument("--raw", default="iSeg-2017-Training")
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--out", default="webdemo/sample.bin.gz")
    args = p.parse_args()

    data, block, full = build(args.raw, args.subject)
    packed = gzip.compress(data, 9)
    Path(args.out).write_bytes(packed)

    print(f"subject {args.subject}: volume {full[0]}x{full[1]}x{full[2]}"
          f" -> block {block[0]}x{block[1]}x{block[2]}")
    print(f"  raw {len(data) / 1e6:.1f} MB -> compressed {len(packed) / 1e6:.2f} MB"
          f"  ({args.out})")


if __name__ == "__main__":
    main()
