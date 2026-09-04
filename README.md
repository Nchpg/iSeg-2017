# iSeg-2017 — frugal infant brain segmentation

Three-tissue segmentation (CSF, grey matter, white matter) on T1 and T2 MRI of
6-month-old infants, sized for mobile and embedded deployment. A browser demo runs
the model on the device, in `webdemo/`.

## Protocol

The 13 subjects in `iSeg-2017-Testing/` **have no labels** — they were never released,
they served the challenge leaderboard. Every number below therefore comes from the 10
training subjects: 8 to train, subjects 1 and 2 held out to measure.

The split is **by subject, never by slice** (two neighbouring slices of the same volume
are nearly identical, mixing them inflates Dice for free), and Dice is computed **per
volume**, after reassembling all slices.

## What the data dictates

Measured on the 10 training subjects; the exploration script is in git history
(`git show 468185b^:explore.py`).

10 subjects, 144×192×256, 1 mm isotropic. T1, T2 and label share the same affine, so the
modalities are already registered. Class balance inside the brain is CSF 22 %, GM 47 %,
WM 31 % — a 2.2× imbalance, mild enough that aggressive weighting is not warranted.

Contrast is the real problem. Intensity histogram intersection (1 = indistinguishable,
0 = separable):

| pair | T1 | T2 |
|---|---|---|
| **GM / WM** | 0.702 | **0.901** |
| CSF / GM | 0.362 | 0.360 |
| CSF / WM | 0.268 | 0.347 |

The Fisher ratio for GM/WM is 0.269 on T1 alone, 0.001 on T2 alone, and 0.297 for the
best linear combination — a 10 % gain. So at 6 months **no combination of intensities
separates GM from WM**. This is the iso-intense phase, and it makes spatial context
mandatory rather than helpful. T2 is kept anyway: it costs almost nothing, and a convnet
exploits non-linear relations a Fisher ratio does not capture. Its real contribution was
never ablated — `--modalities t1` is there if the question comes up.

## Design choices

| Decision | Why |
|---|---|
| **Coronal** orientation | left/right symmetry is visible in-plane, which gives an anatomical cue and makes the horizontal flip a legitimate augmentation |
| Fixed **144×144** crop | covers the brain bounding box with 13–17 voxels of margin, divisible by 16, so no resize and no padding |
| **No N4 correction** | SimpleITK does not ship in a browser; compensated by an augmentation that simulates bias fields, and preprocessing stays reproducible at inference |
| Z-score **inside the brain mask** | including background would crush the statistic under millions of empty voxels |
| **2.5D** rather than 3D | no 3D convolution, so mobile runtimes can accelerate and quantize the model — NNAPI and Core ML support 3D convolutions poorly |
| **Dice + cross-entropy** loss | cross-entropy gives stable gradients early on, Dice optimises the metric directly and ignores class size |

## Models

A 2D U-Net whose input channels carry the third dimension: `context` adjacent coronal
slices per modality (5 by default, so 10 channels with T1+T2).

| variant | description | params | int8 | Dice |
|---|---|---|---|---|
| `standard` | full 3×3 convolutions, base 16, 4 levels | 1,943,636 | 1.89 MB | 0.8927 |
| **`separable`** | depthwise 3×3 + pointwise 1×1 | 386,782 | **0.43 MB** | **0.8624** |
| `tiny` | separable, base 8, 3 levels | 26,974 | 0.07 MB | 0.8281 |

`separable` is the one shipped: **97 % of `standard`'s Dice at 20 % of its size**. Going
smaller (`tiny`) costs another 3.4 Dice points for no practical gain, both already fit on
a phone comfortably. Per-tissue Dice for `separable`: CSF 0.8969, GM 0.8606, WM 0.8297 —
white matter is the hardest, as the 0.702 histogram overlap predicted.

These numbers come from a 5-fold cross-validation run once (best-to-worst fold spread:
0.0085, so performance does not hinge on the split). The code itself uses a single split,
which is enough to measure a model.

## Usage

```bash
uv venv .venv
uv pip install --python .venv/bin/python numpy nibabel torch onnx onnxruntime onnxscript
source env.sh   # NixOS only: puts libstdc++ and zlib where the manylinux wheels look
```

Training runs on a Colab T4 through `colab_iseg.ipynb`, which only orchestrates calls into
the `iseg/` package:

```bash
python -m iseg.train                       # separable, 60 epochs, ~10 min on a T4
python -m iseg.train --variant standard    # the high-end reference
python -m iseg.train --variant tiny        # the smallest
```

Dice on the two validation subjects prints every 5 epochs and the best weights land in
`runs/<variant>.pt`. Export then produces the float32 `.onnx` and its int8 version, about
3.5× smaller (quantization is static, calibrated on real slices — dynamic quantization
leaves convolutions untouched and buys almost nothing here):

```bash
python -m iseg.export --checkpoint runs/separable.pt --deploy webdemo
```

## Mobile demo

`webdemo/` is a static site that reads `.hdr/.img` files, lets you pick a slice and
segments it on the device, in WebAssembly. No image leaves the machine and there is no
inference server. Measured on a phone: **97 ms per slice**, about 13 s for a full volume.

It publishes as-is on GitHub Pages, Vercel or Netlify. Over HTTPS the service worker
caches everything on the first visit and the app becomes installable; it then works
offline, provided one segmentation ran online first, since that is what caches the ONNX
Runtime engine loaded from a CDN. Opening `index.html` over `file://` does not work —
browsers forbid a local page from loading `model.onnx` — so serve the folder instead:

```bash
cd webdemo && python3 -m http.server 8000
node webdemo/test.js                       # headless check of reader + preprocessing
```

The page ships the three variants in int8 and lets you switch between them, showing each
one's parameter count, size, Dice and the inference time measured on the device. Only
`separable` and the first sample MRI are preloaded, the rest is cached on first use.

The three bundled MRIs are for anyone without data at hand. Each holds only the crop
window and the slices containing brain, compressed: 2.4 MB instead of the 28 MB of the
original `.hdr/.img`, reconstructed voxel-exact. To regenerate them:

```bash
for n in 1 2 3; do python -m iseg.sample --subject $n --out webdemo/sample-$n.bin; done
```

*iSeg-2017 data is distributed under conditions — check that redistribution is allowed
before deploying.*

## Layout

```
iseg/data.py            preprocessing, cache, 2.5D dataset
iseg/augment.py         augmentation (geometry + intensity, including bias field)
iseg/model.py           2.5D U-Net and frugal variants
iseg/losses.py          Dice + cross-entropy, volume Dice
iseg/train.py           training, 8 subjects / 2 held out
iseg/export.py          ONNX, int8 quantization, copy to the site
iseg/sample.py          compact demo MRI for the web page
colab_iseg.ipynb        Colab orchestration
webdemo/index.html      demo page: structure
webdemo/app.css         styles
webdemo/app.js          Analyze reader, preprocessing, inference, rendering
webdemo/sw.js           service worker (offline cache), manifest.json, icon-*.png
webdemo/test.js         headless test
webdemo/model-*.onnx    the 3 variants in int8
webdemo/sample-*.bin    3 demo MRIs
```
