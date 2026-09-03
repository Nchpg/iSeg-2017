"""Export ONNX, quantification int8 et mesure de frugalite.

C'est ici que se fabrique le livrable du sujet : le tableau
Dice / parametres / Mo / MACs / latence pour chaque variante.

La latence est mesuree sur CPU MONO-THREAD. C'est un proxy honnete et
defendable pour un coeur ARM de telephone : on ne pretend pas mesurer un
appareil qu'on n'a pas, on borne le calcul a une ressource comparable.
Si un telephone Android est disponible, le meme .onnx se mesure sur
l'appareil reel avec onnxruntime_perf_test via adb (voir README).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import model as models
from .data import ISegSlices


# --------------------------------------------------------------- MACs
def count_macs(net, in_channels, size=144):
    """Multiplications-accumulations pour une inference, par couche conv.

    Compte a la main plutot que d'ajouter une dependance : seules les
    convolutions pesent, les BatchNorm et ReLU sont negligeables.
    """
    total = [0]
    hooks = []

    def hook(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            out_elems = output.numel()
            k = module.kernel_size[0] * module.kernel_size[1]
            total[0] += out_elems * k * (module.in_channels // module.groups)
        elif isinstance(module, nn.ConvTranspose2d):
            in_elems = inputs[0].numel()
            k = module.kernel_size[0] * module.kernel_size[1]
            total[0] += in_elems * k * module.out_channels

    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(hook))
    net.eval()
    with torch.no_grad():
        net(torch.zeros(1, in_channels, size, size))
    for h in hooks:
        h.remove()
    return total[0]


# -------------------------------------------------------------- export
def to_onnx(checkpoint, out_path, size=144, opset=17):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = models.build(ckpt["variant"], ckpt["in_channels"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    dummy = torch.randn(1, ckpt["in_channels"], size, size)
    torch.onnx.export(
        net, dummy, str(out_path),
        input_names=["input"], output_names=["logits"],
        # Le lot reste dynamique : l'application segmente une coupe a la
        # fois, mais le banc d'essai peut en grouper plusieurs.
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    return ckpt


def quantize(onnx_path, out_path, cache_dir, subjects, context, modalities, n_samples=200):
    """Quantification statique int8 avec calibration sur de vraies coupes.

    Statique plutot que dynamique : sur un reseau convolutif, la
    quantification dynamique ne touche pas les convolutions et n'apporte
    presque rien. La calibration a besoin de quelques centaines de coupes
    representatives, pas davantage.
    """
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    ds = ISegSlices(cache_dir, subjects, context, modalities, augment=None)
    rng = np.random.default_rng(0)
    picks = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(picks)

        def get_next(self):
            i = next(self.it, None)
            if i is None:
                return None
            return {"input": ds[int(i)][0][None]}

    prepared = Path(onnx_path).with_suffix(".prep.onnx")
    quant_pre_process(str(onnx_path), str(prepared))
    quantize_static(str(prepared), str(out_path), Reader(),
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    prepared.unlink(missing_ok=True)


# ------------------------------------------------------------ latence
def benchmark(onnx_path, in_channels, size=144, runs=50, warmup=5, threads=1):
    """Latence par coupe, CPU mono-thread par defaut."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    x = np.random.randn(1, in_channels, size, size).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"input": x})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {"ms_mean": float(times.mean()), "ms_std": float(times.std()),
            "ms_p50": float(np.percentile(times, 50)),
            "ms_p90": float(np.percentile(times, 90)), "threads": threads}


def main():
    p = argparse.ArgumentParser(description="Export ONNX + frugalite")
    p.add_argument("--checkpoint", required=True, help="fichier .pt produit par train.py")
    p.add_argument("--cache", default="cache")
    p.add_argument("--out", default="export")
    p.add_argument("--calib-subjects", type=int, nargs="+", default=[1, 2],
                   help="sujets de calibration (ceux de validation du bloc)")
    p.add_argument("--slices-per-volume", type=int, default=135,
                   help="coupes utiles par sujet, pour extrapoler au volume entier")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--no-quant", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).stem

    fp32 = out_dir / f"{stem}.onnx"
    ckpt = to_onnx(args.checkpoint, fp32)
    in_ch = ckpt["in_channels"]

    net = models.build(ckpt["variant"], in_ch)
    report = {
        "variant": ckpt["variant"], "context": ckpt["context"],
        "modalities": ckpt["modalities"], "in_channels": in_ch,
        "params": models.count_parameters(net),
        "macs_per_slice": count_macs(net, in_ch),
        "dice": {k: v for k, v in ckpt["metrics"].items() if k.startswith("dice")},
        "fp32": {"mb": fp32.stat().st_size / 1024 ** 2,
                 **benchmark(fp32, in_ch, runs=args.runs, threads=args.threads)},
    }

    if not args.no_quant:
        int8 = out_dir / f"{stem}.int8.onnx"
        quantize(fp32, int8, args.cache, args.calib_subjects,
                 ckpt["context"], ckpt["modalities"])
        report["int8"] = {"mb": int8.stat().st_size / 1024 ** 2,
                          **benchmark(int8, in_ch, runs=args.runs, threads=args.threads)}

    n = args.slices_per_volume
    print(f"\n=== {report['variant']} | {report['modalities']} | contexte {report['context']}")
    print(f"    parametres      {report['params']:,}")
    print(f"    MACs / coupe    {report['macs_per_slice'] / 1e6:.1f} M")
    print(f"    Dice moyen      {report['dice'].get('dice_mean', float('nan')):.4f}")
    for tag in ("fp32", "int8"):
        if tag in report:
            r = report[tag]
            print(f"    {tag:<5} {r['mb']:>6.2f} Mo   {r['ms_mean']:>7.2f} ms/coupe "
                  f"(+/- {r['ms_std']:.2f}, {r['threads']} thread)"
                  f"   volume ~{r['ms_mean'] * n / 1000:.1f} s")

    (out_dir / f"{stem}_frugalite.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
