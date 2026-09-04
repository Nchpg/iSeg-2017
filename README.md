# iSeg-2017: frugal infant brain segmentation

Three-tissue segmentation (CSF, grey matter, white matter) on T1/T2 MRI of 6-month-old
infants, small enough to run on a phone. `webdemo/` segments in the browser, on the
device, with no server.

## Protocol

The 13 subjects in `iSeg-2017-Testing/` have no labels; they were never released. Every
number here comes from the 10 training subjects: 8 to train, subjects 1 and 2 held out.
The split is by subject, never by slice, and Dice is computed per volume after
reassembling all slices.

## Why it is hard

At 6 months, grey and white matter are iso-intense. Histogram intersection for GM/WM
(1 = indistinguishable) is 0.702 on T1 and 0.901 on T2. The Fisher ratio is 0.269 on T1,
0.001 on T2, and 0.297 for the best linear combination of the two. No combination of
intensities separates the tissues, so spatial context is mandatory rather than helpful.
Full measurements: `git show 468185b^:explore.py`.

The rest is comfortable: 10 subjects at 144×192×256, 1 mm isotropic, T1, T2 and label
already registered, and a mild class imbalance inside the brain (CSF 22 %, GM 47 %,
WM 31 %).

## Design choices

| Decision | Why |
|---|---|
| Coronal orientation | left/right symmetry is in-plane, which makes the horizontal flip a legitimate augmentation |
| Fixed 144×144 crop | covers the brain with 13-17 voxels of margin, divisible by 16, so no resize and no padding |
| No N4 correction | SimpleITK does not ship in a browser; an augmentation simulates bias fields instead |
| Z-score inside the brain mask | background would crush the statistic under millions of empty voxels |
| 2.5D rather than 3D | NNAPI and Core ML accelerate and quantize 3D convolutions poorly |
| Dice + cross-entropy | cross-entropy is stable early, Dice optimises the metric and ignores class size |

## Models

A 2D U-Net whose input channels carry the third dimension: 5 adjacent coronal slices per
modality, so 10 channels with T1+T2.

| variant | description | params | int8 | Dice |
|---|---|---|---|---|
| `standard` | full 3×3 convolutions, base 16, 4 levels | 1,943,636 | 1.89 MB | 0.8927 |
| **`separable`** | depthwise 3×3 + pointwise 1×1 | 386,782 | **0.43 MB** | **0.8624** |
| `tiny` | separable, base 8, 3 levels | 26,974 | 0.07 MB | 0.8281 |

`separable` ships: 97 % of `standard`'s Dice at 20 % of its size. Per tissue, CSF 0.8969,
GM 0.8606, WM 0.8297. White matter is the hardest, as the histogram overlap predicted.
A 5-fold cross-validation spread 0.0085 between best and worst fold, so the result does
not hinge on the split.

## Usage

```bash
uv venv .venv
uv pip install --python .venv/bin/python numpy nibabel torch onnx onnxruntime onnxscript
source env.sh   # NixOS only: puts libstdc++ and zlib where the manylinux wheels look
```

Training runs on a Colab T4 via `colab_iseg.ipynb`, which only calls into `iseg/`. Dice on
the validation subjects prints every 5 epochs; the best weights land in `runs/<variant>.pt`.

```bash
python -m iseg.train                                            # separable, 60 epochs, ~10 min
python -m iseg.train --variant standard                         # or standard, tiny
python -m iseg.export --checkpoint runs/separable.pt --deploy webdemo
```

Export produces the float32 `.onnx` and a static int8 version about 3.5× smaller,
calibrated on real slices.

## Demo

`webdemo/` is a static site: it reads `.hdr/.img`, segments the slice you pick in
WebAssembly, and ships three bundled MRIs for anyone without data at hand. 97 ms per
slice on a phone. Publish the folder as-is on GitHub Pages, Vercel or Netlify; over HTTPS
the service worker makes it installable and offline-capable once one segmentation has run.

`file://` does not work, since browsers forbid a local page from loading `model.onnx`:

```bash
cd webdemo && python3 -m http.server 8000
node webdemo/test.js                       # headless check of reader + preprocessing
for n in 1 2 3; do python -m iseg.sample --subject $n --out webdemo/sample-$n.bin; done
```

*iSeg-2017 data is distributed under conditions. Check that redistribution is allowed
before deploying.*

## Layout

```
iseg/data.py       preprocessing, cache, 2.5D dataset
iseg/augment.py    geometry + intensity augmentation, including bias field
iseg/model.py      2.5D U-Net and frugal variants
iseg/losses.py     Dice + cross-entropy, volume Dice
iseg/train.py      training, 8 subjects / 2 held out
iseg/export.py     ONNX, int8 quantization, copy to the site
iseg/sample.py     compact demo MRI for the web page
webdemo/app.js     Analyze reader, preprocessing, inference, rendering
webdemo/test.js    headless test
```
