"""Fabrique l'echantillon de demonstration charge par la page web.

    python -m iseg.sample --subject 1 --out webdemo/sample.bin.gz

Les .hdr/.img d'origine pesent 28 Mo par sujet. On ne garde que ce que
l'application lit -- la fenetre de recadrage de data.py et les coupes
qui contiennent du cerveau -- puis on compresse : environ 2,5 Mo.

Format produit, entierement little-endian :

    magic   4 octets  "ISG1"
    x0 y0 z0          3 x int32   position du bloc dans le volume complet
    nx ny nz          3 x int32   dimensions du bloc
    fx fy fz          3 x int32   dimensions du volume complet
    T1                int16[nx*ny*nz]
    T2                int16[nx*ny*nz]

Les voxels sont ranges avec x le plus rapide, puis y, puis z, comme dans
les fichiers Analyze d'origine : la page peut donc les replacer tels
quels dans un volume pleine taille.
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
        raise ValueError(f"T1 {t1.shape} et T2 {t2.shape} n'ont pas la meme geometrie")
    fx, fy, fz = t1.shape

    # bornes des coupes contenant du cerveau
    occupancy = (t1 > 0).sum(axis=(0, 2))
    utiles = np.where(occupancy > 0)[0]
    y0, y1 = int(utiles[0]), int(utiles[-1]) + 1

    x0, x1 = CROP_X
    z0, z1 = CROP_Z
    bloc = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
    b1, b2 = t1[bloc], t2[bloc]
    nx, ny, nz = b1.shape

    entete = MAGIC + struct.pack("<9i", x0, y0, z0, nx, ny, nz, fx, fy, fz)
    # ordre Fortran : x le plus rapide, comme dans le .img d'origine
    corps = b1.tobytes(order="F") + b2.tobytes(order="F")
    return entete + corps, (nx, ny, nz), (fx, fy, fz)


def main():
    p = argparse.ArgumentParser(description="Echantillon de demonstration pour la page web")
    p.add_argument("--raw", default="iSeg-2017-Training")
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--out", default="webdemo/sample.bin.gz")
    args = p.parse_args()

    donnees, bloc, complet = build(args.raw, args.subject)
    compresse = gzip.compress(donnees, 9)
    Path(args.out).write_bytes(compresse)

    print(f"sujet {args.subject} : volume {complet[0]}x{complet[1]}x{complet[2]}"
          f" -> bloc {bloc[0]}x{bloc[1]}x{bloc[2]}")
    print(f"  brut {len(donnees) / 1e6:.1f} Mo -> compresse {len(compresse) / 1e6:.2f} Mo"
          f"  ({args.out})")


if __name__ == "__main__":
    main()
