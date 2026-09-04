"""Export a checkpoint to a quantized .onnx, ready to embed.

    python -m iseg.export --checkpoint runs/separable.pt
    python -m iseg.export --checkpoint runs/separable.pt --deploy webdemo

Produces the float32 .onnx and its int8 version, about 3.5x smaller.
With --deploy, the int8 model is also copied into the site folder.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from . import model as models
from .data import ISegSlices

# Slice size the network expects, set by the crop window in data.py
SIZE = 144


def to_onnx(checkpoint, out_path):
    """Export the network, weights included, into a single .onnx file.

    Returns the loaded checkpoint and the rebuilt network."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = models.build(ckpt["variant"], ckpt["in_channels"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    dummy = torch.randn(1, ckpt["in_channels"], SIZE, SIZE)
    torch.onnx.export(
        net, dummy, str(out_path),
        input_names=["input"], output_names=["logits"],
        # Dynamic batch: the app segments one slice at a time, but
        # grouping several may be useful.
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # The dynamo exporter of torch >= 2.5 writes the weights into a
    # separate ".onnx.data" file. Fold them back in, otherwise the .onnx
    # holds only the graph and the model is unusable on its own.
    import onnx
    m = onnx.load(str(out_path), load_external_data=True)
    onnx.save(m, str(out_path), save_as_external_data=False)
    for extra in Path(out_path).parent.glob(f"{Path(out_path).name}.data*"):
        extra.unlink(missing_ok=True)

    return ckpt, net


def quantize(onnx_path, out_path, cache_dir, subjects, context, modalities, n_samples=200):
    """Static int8 quantization, calibrated on real slices.

    Static rather than dynamic: on a convolutional network, dynamic
    quantization leaves the convolutions untouched and gains almost
    nothing.
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


def deploy(onnx_path, site_dir, variant):
    """Copy the model under the name the page expects: the name carries
    the variant, since the page offers several and loads them on demand."""
    target = Path(site_dir) / f"model-{variant}.onnx"
    if not target.parent.is_dir():
        raise ValueError(f"site folder not found: {site_dir}")
    shutil.copyfile(onnx_path, target)
    return target


def mb(path):
    return Path(path).stat().st_size / 1024 ** 2


def main():
    p = argparse.ArgumentParser(description="Quantized ONNX export")
    p.add_argument("--checkpoint", required=True, help=".pt file produced by train.py")
    p.add_argument("--cache", default="cache", help="slices used for calibration")
    p.add_argument("--calib-subjects", type=int, nargs="+", default=[1, 2])
    p.add_argument("--out", default="export")
    p.add_argument("--deploy", metavar="FOLDER",
                   help="copy the int8 model into the site folder")
    p.add_argument("--no-quant", action="store_true", help="float32 only")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).stem

    fp32 = out_dir / f"{stem}.onnx"
    ckpt, net = to_onnx(args.checkpoint, fp32)
    print(f"{ckpt['variant']:<12} {models.count_parameters(net):>9,} parameters")
    print(f"  float32   {fp32.name:<32} {mb(fp32):>6.2f} MB")

    final = fp32
    if not args.no_quant:
        final = out_dir / f"{stem}.int8.onnx"
        quantize(fp32, final, args.cache, args.calib_subjects,
                 ckpt["context"], ckpt["modalities"])
        print(f"  int8      {final.name:<32} {mb(final):>6.2f} MB   "
              f"(/{mb(fp32) / mb(final):.1f})")

    if args.deploy:
        target = deploy(final, args.deploy, ckpt["variant"])
        print(f"  copied    {target}")


if __name__ == "__main__":
    main()
