"""Export d'un checkpoint vers un .onnx quantifie, pret a embarquer.

    python -m iseg.export --checkpoint runs/separable.pt
    python -m iseg.export --checkpoint runs/separable.pt --deploy webdemo

Produit le .onnx float32 et sa version int8 (environ 3,5x plus petite).
Avec --deploy, le modele int8 est aussi copie sous le nom model.onnx
dans le dossier du site, ou la page va le chercher.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from . import model as models
from .data import ISegSlices

# Taille des coupes attendue par le reseau, fixee par le recadrage de data.py
SIZE = 144


def to_onnx(checkpoint, out_path):
    """Exporte le reseau, poids compris, dans un fichier .onnx unique."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = models.build(ckpt["variant"], ckpt["in_channels"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    dummy = torch.randn(1, ckpt["in_channels"], SIZE, SIZE)
    torch.onnx.export(
        net, dummy, str(out_path),
        input_names=["input"], output_names=["logits"],
        # Le lot reste dynamique : l'application segmente une coupe a la
        # fois, mais on peut vouloir en grouper plusieurs.
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # L'exportateur dynamo de torch >= 2.5 ecrit les poids dans un fichier
    # ".onnx.data" separe. On les reintegre, sinon le .onnx ne contient que
    # le graphe et le modele est inutilisable seul.
    import onnx
    m = onnx.load(str(out_path), load_external_data=True)
    onnx.save(m, str(out_path), save_as_external_data=False)
    for extra in Path(out_path).parent.glob(f"{Path(out_path).name}.data*"):
        extra.unlink(missing_ok=True)

    return ckpt


def quantize(onnx_path, out_path, cache_dir, subjects, context, modalities, n_samples=200):
    """Quantification statique int8, calibree sur de vraies coupes.

    Statique plutot que dynamique : sur un reseau convolutif, la
    quantification dynamique ne touche pas les convolutions et ne gagne
    presque rien. Quelques centaines de coupes suffisent a calibrer.
    """
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    ds = ISegSlices(cache_dir, subjects, context, modalities, augment=None)
    picks = np.random.default_rng(0).choice(len(ds), min(n_samples, len(ds)), replace=False)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(picks)

        def get_next(self):
            i = next(self.it, None)
            return None if i is None else {"input": ds[int(i)][0][None]}

    prepared = Path(onnx_path).with_suffix(".prep.onnx")
    quant_pre_process(str(onnx_path), str(prepared))
    quantize_static(str(prepared), str(out_path), Reader(),
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    prepared.unlink(missing_ok=True)


def deploy(onnx_path, site_dir):
    """Place le modele la ou la page ira le chercher."""
    cible = Path(site_dir) / "model.onnx"
    if not cible.parent.is_dir():
        raise ValueError(f"dossier du site introuvable : {site_dir}")
    shutil.copyfile(onnx_path, cible)
    return cible


def mo(path):
    return Path(path).stat().st_size / 1024 ** 2


def main():
    p = argparse.ArgumentParser(description="Export ONNX quantifie")
    p.add_argument("--checkpoint", required=True, help="fichier .pt produit par train.py")
    p.add_argument("--cache", default="cache", help="coupes servant a la calibration")
    p.add_argument("--calib-subjects", type=int, nargs="+", default=[1, 2])
    p.add_argument("--out", default="export")
    p.add_argument("--deploy", metavar="DOSSIER",
                   help="copier le modele int8 en <DOSSIER>/model.onnx")
    p.add_argument("--no-quant", action="store_true", help="float32 seulement")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).stem

    fp32 = out_dir / f"{stem}.onnx"
    ckpt = to_onnx(args.checkpoint, fp32)
    print(f"{ckpt['variant']:<12} {models.count_parameters(models.build(ckpt['variant'], ckpt['in_channels'])):>9,} parametres")
    print(f"  float32   {fp32.name:<32} {mo(fp32):>6.2f} Mo")

    final = fp32
    if not args.no_quant:
        final = out_dir / f"{stem}.int8.onnx"
        quantize(fp32, final, args.cache, args.calib_subjects,
                 ckpt["context"], ckpt["modalities"])
        print(f"  int8      {final.name:<32} {mo(final):>6.2f} Mo   "
              f"(/{mo(fp32) / mo(final):.1f})")

    if args.deploy:
        cible = deploy(final, args.deploy)
        print(f"  copie     {cible}")


if __name__ == "__main__":
    main()
