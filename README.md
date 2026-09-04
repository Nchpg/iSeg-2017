# iSeg-2017 — frugal infant brain segmentation

Three-tissue segmentation (CSF, grey matter, white matter) on T1 and T2 MRI of
6-month-old infants, sized for mobile and embedded deployment.

## Evaluation protocol

The 13 subjects in `iSeg-2017-Testing/` **have no labels** — they were never
released, they served the challenge leaderboard. Every number below therefore
comes from the 10 training subjects: **8 to train, subjects 1 and 2 held out to
measure**. The test folder only yields qualitative predictions and an inference
timing.

The split is **by subject, never by slice**: two neighbouring slices of the same
volume are nearly identical, and mixing them across train and validation inflates
Dice for free.

Dice is computed **per volume**, after reassembling all slices.

## What the data dictates

Measured on the 10 training subjects (the exploration script is in git history:
`git show HEAD~1:explore.py`).

**Geometry.** 10 subjects, 144×192×256, 1 mm isotropic. T1, T2 and label share the
same affine, so the modalities are already registered.

**Class balance** inside the brain: CSF 22 %, GM 47 %, WM 31 %. A 2.2× imbalance,
mild enough that aggressive weighting is not warranted.

**Low contrast, quantified.** Intensity histogram intersection (1 = indistinguishable,
0 = separable):

| pair | T1 | T2 |
|---|---|---|
| **GM / WM** | 0.702 | **0.901** |
| CSF / GM | 0.362 | 0.360 |
| CSF / WM | 0.268 | 0.347 |

Fisher ratio for GM/WM: **0.269 on T1 alone, 0.001 on T2 alone**, and 0.297 for the
best linear combination of the two — a 10 % gain.

So at 6 months **no combination of intensities separates GM from WM**. This is the
iso-intense phase, and it makes spatial context mandatory rather than helpful. T2 is
kept anyway: it costs almost nothing, and a convnet exploits non-linear relations
that a Fisher ratio does not capture. Its real contribution was never ablated —
`--modalities t1` is there if the question comes up.

## Design choices

| Decision | Why |
|---|---|
| **Coronal** orientation | left/right symmetry is visible in-plane, which gives an anatomical cue and makes the horizontal flip a legitimate augmentation |
| Fixed **144×144** crop | covers the brain bounding box with 13–17 voxels of margin, divisible by 16, so no resize and no padding |
| **No N4 correction** | SimpleITK does not ship in a browser; compensated by an augmentation that simulates bias fields. Traded for preprocessing that is reproducible at inference |
| Z-score **inside the brain mask** | including background would crush the statistic under millions of empty voxels |
| **2.5D** rather than 3D | no 3D convolution, so mobile runtimes can accelerate and quantize the model — NNAPI and Core ML support 3D convolutions poorly |
| **Dice + cross-entropy** loss | cross-entropy gives stable gradients early on, Dice optimises the metric directly and ignores class size |

## Models

A 2D U-Net whose input channels carry the third dimension: `context` adjacent
coronal slices per modality (5 by default, so 10 channels with T1+T2).

| variant | description | params | int8 | Dice |
|---|---|---|---|---|
| `standard` | full 3×3 convolutions, base 16, 4 levels | 1,943,636 | 1.89 MB | 0.8927 |
| **`separable`** | depthwise 3×3 + pointwise 1×1 | 386,782 | **0.43 MB** | **0.8624** |
| `tiny` | separable, base 8, 3 levels | 26,974 | 0.07 MB | 0.8281 |

`separable` is the one shipped: **97 % of `standard`'s Dice at 20 % of its size**.
Going smaller (`tiny`) costs another 3.4 Dice points for a size gain with no
practical effect — both already fit on a phone comfortably.

Per-tissue Dice for `separable`: CSF 0.8969, GM 0.8606, WM 0.8297. White matter is
the hardest, as the 0.702 histogram overlap predicted.

These numbers come from a 5-fold cross-validation run once (best-to-worst fold
spread: 0.0085, so performance does not hinge on the split). The code itself uses a
single split, which is enough to measure a model.

## Usage

### Local environment

```bash
uv venv .venv
uv pip install --python .venv/bin/python numpy nibabel torch onnx onnxruntime onnxscript
```

On NixOS the manylinux wheels need `libstdc++` and `zlib` on the library path;
`env.sh` puts them there:

```bash
source env.sh
```

### Training (Colab, T4 GPU)

Open `colab_iseg.ipynb`. The notebook only orchestrates calls into the `iseg/`
package, which stays versioned in git.

```bash
python -m iseg.train                       # separable, 60 epochs, ~10 min on a T4
python -m iseg.train --variant standard    # the high-end reference
python -m iseg.train --variant tiny        # the smallest
```

Dice on the two validation subjects prints every 5 epochs, and the best weights land
in `runs/<variant>.pt`.

### Export

```bash
python -m iseg.export --checkpoint runs/separable.pt
```

Produces the float32 `.onnx` and its int8 version, about 3.5× smaller (1.49 MB →
0.43 MB for `separable`). Quantization is static, calibrated on real slices —
dynamic quantization leaves convolutions untouched and buys almost nothing here.

### Mobile demo

`webdemo/` is a static site that reads `.hdr/.img` files, lets you pick a slice and
segments it **on the device**, in WebAssembly. No image leaves the machine and there
is no inference server. Measured on a phone: **97 ms per slice**, about 13 s for a
full volume.

```
webdemo/
  index.html      structure
  app.css         styles
  app.js          Analyze reader, preprocessing, inference, rendering
  model-*.onnx    the 3 variants in int8, selectable in the UI
  sample-*.bin    3 demo MRIs (2.0–2.4 MB each)
  sw.js           service worker (offline cache)
  manifest.json   PWA manifest
  icon-*.png      app icons
  test.js         headless test: node webdemo/test.js
```

The three bundled MRIs are for anyone without data at hand. Only the first is
preloaded by the service worker; the other two are cached on first use. To
regenerate them:

```bash
for n in 1 2 3; do python -m iseg.sample --subject $n --out webdemo/sample-$n.bin; done
```

Each file holds only the crop window and the slices containing brain, compressed:
2.4 MB instead of the 28 MB of the original `.hdr/.img`, reconstructed voxel-exact.
*iSeg-2017 data is distributed under conditions — check that redistribution is
allowed before deploying.*

The folder publishes as-is on GitHub Pages, Vercel or Netlify. Over HTTPS the service
worker caches everything on the first visit and the app becomes installable. It then
works without a network, provided **one segmentation ran online** first — that is what
caches the ONNX Runtime engine, which is loaded from a CDN.

Opening `index.html` over `file://` does not work: browsers forbid a local page from
loading `model.onnx`. To try it locally, serve the folder:

```bash
cd webdemo && python3 -m http.server 8000
```

To put another model in it:

```bash
for v in standard separable tiny; do
  python -m iseg.export --checkpoint runs/$v.pt --deploy webdemo
done
```

The page lets you switch variants and shows each one's parameter count, size, Dice
and the inference time measured on the device. Only `separable` is preloaded; the
other two are cached when selected.

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
webdemo/                static site: mobile demo (PWA)
```
