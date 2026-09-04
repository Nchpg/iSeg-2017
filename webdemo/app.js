"use strict";

/* iSeg Viewer — 2.5D segmentation in the browser.

   Everything runs on the device: the ONNX model is loaded once, then
   each slice is segmented through WebAssembly. No image ever leaves the
   machine.

   Preprocessing mirrors iseg/data.py exactly — brain mask from T1 > 0,
   z-score within that mask applied to both modalities, fixed crop, 5
   context slices per modality. Any divergence here silently wrecks the
   Dice score. */

/* Must stay in sync with iseg/data.py and the exported checkpoint. */
const CROP_X = [0, 144];
const CROP_Z = [72, 216];
const CONTEXT = 5;
const SIZE = 144;
/* Training figures, hardcoded because they describe training rather
   than this run. Inference time is the one thing measured live. */
const MODELS = {
  tiny:      { params: 26974,   size: "0.07 MB", dice: "0.828" },
  separable: { params: 386782,  size: "0.43 MB", dice: "0.862" },
  standard:  { params: 1943636, size: "1.89 MB", dice: "0.893" },
};
const DEFAULT_MODEL = "separable";
const MODEL_URL = (v) => `model-${v}.onnx`;
let currentModel = DEFAULT_MODEL;

const COLOURS = {
  1: [0x4c, 0x9b, 0xe8],   // cerebrospinal fluid
  2: [0xe8, 0x93, 0x4c],   // grey matter
  3: [0x5f, 0xbf, 0x77],   // white matter
};

const $ = (id) => document.getElementById(id);
const buttons = (sel) => [...document.querySelectorAll(sel)];

/* Marks the active button of a segmented control. */
function markActive(sel, key, value) {
  for (const b of buttons(sel)) b.classList.toggle("on", b.dataset[key] === value);
}

let session = null;
let volT1 = null;          // {X, Y, Z, raw: Int16Array, norm: Float32Array, name}
let volT2 = null;
let displayedModality = "t1";    // modality DISPLAYED (the network always uses both)
let lastClasses = null;
let lastInput = null;
let occupancy = null;         // fraction of brain voxels per slice

function showError(msg) {
  $("errBanner").textContent = msg;
  $("errBanner").classList.add("on");
}
function clearError() { $("errBanner").classList.remove("on"); }

function setModelStatus(state, text) {
  $("modelChip").className = "chip " + state;
  $("modelChipText").textContent = text;
}

function showModelSpec(variant) {
  const m = MODELS[variant];
  $("mParams").textContent = m.params.toLocaleString("en-US");
  $("mSize").textContent = m.size;
  $("mDice").textContent = m.dice;
  markActive(".segmented .mdl", "variant", variant);
}

async function loadModel(variant = currentModel) {
  ort.env.wasm.numThreads = 1;   // avoids the COOP/COEP requirement of plain hosting
  const btns = buttons(".segmented .mdl");
  btns.forEach((b) => { b.disabled = true; });
  setModelStatus("busy", `loading ${variant}…`);
  try {
    session = await ort.InferenceSession.create(MODEL_URL(variant), { executionProviders: ["wasm"] });
    currentModel = variant;
    showModelSpec(variant);
    setModelStatus("ok", `${variant} ready`);
    $("mTime").textContent = "—";
    if (volT1 && volT2) requestSegmentation();
  } catch (e) {
    setModelStatus("err", "model unavailable");
    showError(
      location.protocol === "file:"
        ? "Opened over file://, this page cannot load model.onnx: browsers forbid a local " +
          "page from reading another local file. Serve the folder " +
          "(python3 -m http.server) or use the hosted version."
        : "Could not load the model: " + e.message
    );
  } finally {
    btns.forEach((b) => { b.disabled = false; });
  }
}

/* Analyze 7.5: 348-byte header, little-endian, raw data with no offset.
   Voxels stored with x varying fastest, then y, then z (checked against
   nibabel on the challenge files). */

function readHeader(buf) {
  const dv = new DataView(buf);
  const dim = [];
  for (let i = 0; i < 8; i++) dim.push(dv.getInt16(40 + i * 2, true));
  return { X: dim[1], Y: dim[2], Z: dim[3], datatype: dv.getInt16(70, true), bitpix: dv.getInt16(72, true) };
}

async function readVolume(hdrFile, imgFile) {
  const { X, Y, Z, datatype, bitpix } = readHeader(await hdrFile.arrayBuffer());
  if (datatype !== 4 || bitpix !== 16) {
    throw new Error(`unexpected data type (datatype=${datatype}, bitpix=${bitpix}): a 16-bit integer is expected`);
  }
  if (X < CROP_X[1] || Z < CROP_Z[1]) {
    throw new Error(`unexpected dimensions (${X}×${Y}×${Z}): the iSeg-2017 geometry (144×192×256) is expected`);
  }
  const buf = await imgFile.arrayBuffer();
  return { X, Y, Z, raw: new Int16Array(buf, 0, X * Y * Z),
           name: hdrFile.name.replace(/\.hdr$/i, "") };
}

function zscoreInMask(raw, mask) {
  let sum = 0, n = 0;
  for (let i = 0; i < raw.length; i++) if (mask[i]) { sum += raw[i]; n++; }
  const mean = sum / n;
  let sqSum = 0;
  for (let i = 0; i < raw.length; i++) if (mask[i]) { const d = raw[i] - mean; sqSum += d * d; }
  const sd = Math.sqrt(sqSum / n) || 1e-8;
  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = (raw[i] - mean) / sd;
  return out;
}

/* Extracts the coronal slice (x, z) at fixed Y, cropped, laid out as
   [x][z] to match the C-order flatten of stack_context() in Python. */
function extractSlice(vol, source, y) {
  const out = new Float32Array(SIZE * SIZE);
  const [x0] = CROP_X, [z0] = CROP_Z;
  for (let xi = 0; xi < SIZE; xi++) {
    const base = (xi + x0) + y * vol.X;
    for (let zi = 0; zi < SIZE; zi++) out[xi * SIZE + zi] = source[base + (zi + z0) * vol.X * vol.Y];
  }
  return out;
}

/* Stacks 5 neighbouring slices per modality -> 10 channels. At the ends
   of the volume the edge slice is repeated: padding with zeros would
   create an artificial black border, which misleads the network. */
function buildTensor(y) {
  const half = CONTEXT >> 1;
  const out = new Float32Array(2 * CONTEXT * SIZE * SIZE);
  let ptr = 0;
  for (const vol of [volT1, volT2]) {
    for (let k = -half; k <= half; k++) {
      const yy = Math.min(Math.max(y + k, 0), vol.Y - 1);
      out.set(extractSlice(vol, vol.norm, yy), ptr);
      ptr += SIZE * SIZE;
    }
  }
  return out;
}

function sliceToImage(vol, y) {
  const raw = extractSlice(vol, vol.raw, y);
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < raw.length; i++) { if (raw[i] < mn) mn = raw[i]; if (raw[i] > mx) mx = raw[i]; }
  const range = (mx - mn) || 1;
  const img = new Uint8ClampedArray(SIZE * SIZE * 4);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const g = ((raw[i] - mn) / range) * 255;
    img[i * 4] = img[i * 4 + 1] = img[i * 4 + 2] = g;
    img[i * 4 + 3] = 255;
  }
  return img;
}

function drawInput(y) {
  const img = sliceToImage(displayedModality === "t1" ? volT1 : volT2, y);
  $("canvasInput").getContext("2d").putImageData(new ImageData(img, SIZE, SIZE), 0, 0);
  $("emptyInput").style.display = "none";
  return img;
}

function drawSegmentation(base, classes, opacity) {
  const out = new Uint8ClampedArray(base);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const c = classes[i];
    if (c === 0) continue;
    const [r, g, b] = COLOURS[c];
    out[i * 4]     = out[i * 4]     * (1 - opacity) + r * opacity;
    out[i * 4 + 1] = out[i * 4 + 1] * (1 - opacity) + g * opacity;
    out[i * 4 + 2] = out[i * 4 + 2] * (1 - opacity) + b * opacity;
  }
  $("canvasSeg").getContext("2d").putImageData(new ImageData(out, SIZE, SIZE), 0, 0);
  $("emptySeg").style.display = "none";
}

/* Occupancy profile: share of brain per slice. The volume holds 192
   slices but the brain only spans about 130 of them; drawing it under
   the slider saves you from scrubbing blindly through empty space. */
function computeOccupancy(vol) {
  const p = new Float32Array(vol.Y);
  const [x0, x1] = CROP_X, [z0, z1] = CROP_Z;
  const area = (x1 - x0) * (z1 - z0);
  for (let y = 0; y < vol.Y; y++) {
    let n = 0;
    for (let z = z0; z < z1; z++) {
      const base = y * vol.X + z * vol.X * vol.Y;
      for (let x = x0; x < x1; x++) if (vol.raw[base + x] > 0) n++;
    }
    p[y] = n / area;
  }
  return p;
}

/* Slider thumb width in CSS pixels: must match the
   input[type=range]::-webkit-slider-thumb rule in app.css. The thumb
   centre only travels [THUMB/2, width - THUMB/2], so the profile has to
   use exactly the same range or the two drift apart at both ends. */
const THUMB = 18;

function profileGeometry() {
  const r = $("occupancy").getBoundingClientRect();
  return { width: r.width, left: r.left, usable: Math.max(1, r.width - THUMB) };
}

function drawProfile() {
  const c = $("occupancy");
  const { width, usable } = profileGeometry();
  if (width < 2) return;

  // Resize the buffer to the size actually displayed, accounting for
  // screen density: otherwise the marker is blurry and its coordinates
  // no longer line up with the slider.
  const dpr = window.devicePixelRatio || 1;
  const h = 26;
  if (c.width !== Math.round(width * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(width * dpr);
    c.height = Math.round(h * dpr);
  }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, h);
  if (!occupancy) return;

  const max = Math.max(...occupancy) || 1;
  const x0 = THUMB / 2;

  ctx.fillStyle = "#1a2029";
  ctx.fillRect(0, 0, width, h);

  ctx.fillStyle = "#313c4b";
  for (let i = 0; i < usable; i++) {
    const v = occupancy[Math.min(occupancy.length - 1, Math.floor((i / usable) * occupancy.length))] / max;
    const hb = Math.max(1, v * (h - 4));
    ctx.fillRect(x0 + i, h - hb - 2, 1, hb);
  }

  // current slice marker, on the same range as the thumb
  const frac = +$("sliceSlider").value / (+$("sliceSlider").max || 1);
  ctx.fillStyle = "#38c6b0";
  ctx.fillRect(x0 + frac * usable - 1, 0, 2, h);
}

function updateStats(classes) {
  const n = [0, 0, 0, 0];
  for (let i = 0; i < classes.length; i++) n[classes[i]]++;
  const brain = n[1] + n[2] + n[3];
  const pct = brain ? [n[1], n[2], n[3]].map((c) => (100 * c) / brain) : [0, 0, 0];
  const bars = $("statBar").children;
  ["pctCSF", "pctGM", "pctWM"].forEach((id, i) => {
    bars[i].style.width = pct[i].toFixed(1) + "%";
    $(id).textContent = pct[i].toFixed(1) + " %";
  });
}

/* "Latest request wins" queue: a new inference starts as soon as the
   previous one finishes, on the most recent slice requested. A plain
   debounce would freeze the view until the gesture stopped. */
let inferenceRunning = false;
let requestedSlice = null;

function requestSegmentation(y) {
  requestedSlice = y !== undefined ? y : +$("sliceSlider").value;
  if (!inferenceRunning) segmentationLoop();
}

async function segmentationLoop() {
  inferenceRunning = true;
  try {
    while (requestedSlice !== null) {
      const y = requestedSlice;
      requestedSlice = null;
      await segmentSlice(y);
    }
  } finally {
    // A thrown error must not leave the flag set, or every later request
    // wedges on "computing…" for good.
    inferenceRunning = false;
  }
}

/* Redraws the background image only: a few milliseconds, so the
   greyscale tracks your finger without waiting for inference. */
function refreshBackground(y) {
  if (!volT1 || !volT2) return;
  lastInput = drawInput(y);
  if (lastClasses) {
    drawSegmentation(lastInput, lastClasses, +$("opacitySlider").value / 100);
  }
}

async function segmentSlice(y) {
  if (!session || !volT1 || !volT2) return;
  clearError();
  $("segStateLabel").textContent = "computing…";

  try {
    const data = buildTensor(y);
    const t1 = performance.now();
    const output = await session.run({
      input: new ort.Tensor("float32", data, [1, 2 * CONTEXT, SIZE, SIZE]),
    });
    const msInf = performance.now() - t1;

    const logits = output.logits.data;   // [1, 4, 144, 144]
    const classes = new Uint8Array(SIZE * SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      let best = 0, val = -Infinity;
      for (let c = 0; c < 4; c++) {
        const v = logits[c * SIZE * SIZE + i];
        if (v > val) { val = v; best = c; }
      }
      classes[i] = best;
    }

    lastClasses = classes;
    // The displayed slice may have changed while computing: redraw the
    // matching background so two different slices never get overlaid.
    lastInput = drawInput(y);
    drawSegmentation(lastInput, classes, +$("opacitySlider").value / 100);
    updateStats(classes);

    const timing = $("mTime");
    timing.textContent = `${msInf.toFixed(0)} ms`;
    timing.classList.add("live");
    $("segStateLabel").textContent = "slice " + y;
    $("panelStats").classList.remove("off");
  } catch (e) {
    $("segStateLabel").textContent = "error";
    showError("Segmentation failed: " + e.message);
  }
}

/* Bundled sample: the crop window plus the slices holding brain, gzipped
   to about 2.4 MB instead of the 28 MB of the original .hdr/.img pair,
   unpacked back to full size so the rest of the code sees no difference.
   Layout is described in iseg/sample.py. */

const SAMPLE_URL = (n) => `sample-${n}.bin`;

function unpackSample(buffer) {
  const dv = new DataView(buffer);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== "ISG1") throw new Error("unrecognised sample file");

  const n = [];
  for (let i = 0; i < 9; i++) n.push(dv.getInt32(4 + i * 4, true));
  const [x0, y0, z0, nx, ny, nz, fx, fy, fz] = n;

  const voxels = nx * ny * nz;
  const build = (offset) => {
    const block = new Int16Array(buffer, offset, voxels);
    const full = new Int16Array(fx * fy * fz);   // untouched voxels stay 0, like the background
    let ptr = 0;
    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        const base = (y0 + iy) * fx + (z0 + iz) * fx * fy;
        for (let ix = 0; ix < nx; ix++) full[base + x0 + ix] = block[ptr++];
      }
    }
    return full;
  };

  const header = 4 + 9 * 4;
  return {
    t1: { X: fx, Y: fy, Z: fz, raw: build(header), name: "sample T1" },
    t2: { X: fx, Y: fy, Z: fz, raw: build(header + voxels * 2), name: "sample T2" },
  };
}

async function loadSample(subject) {
  const btns = buttons(".segmented .ex");
  btns.forEach((b) => { b.disabled = true; });
  clearError();
  try {
    const response = await fetch(SAMPLE_URL(subject));
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    // Some hosts gunzip the response themselves, so check the magic
    // number rather than assume.
    let buffer = await response.arrayBuffer();
    const head = new Uint8Array(buffer, 0, 2);
    if (head[0] === 0x1f && head[1] === 0x8b) {
      buffer = await new Response(
        new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"))
      ).arrayBuffer();
    }

    const { t1, t2 } = unpackSample(buffer);
    t1.name = `example ${subject} T1`;
    t2.name = `example ${subject} T2`;
    volT1 = t1;
    volT2 = t2;
    showRow("rowT1", t1);
    showRow("rowT2", t2);
    normaliseVolumes();
    $("dropZone").classList.add("filled");
    $("shell").classList.add("loaded");
    $("btnBrowse").textContent = "Change volumes";
    markActive(".segmented .ex", "subject", subject);
    enableWhenReady();
  } catch (e) {
    showError("Could not load the example: " + e.message);
  } finally {
    btns.forEach((b) => { b.disabled = false; });
  }
}

function showRow(id, vol) {
  const row = $(id);
  row.classList.add("ok");
  row.querySelector(".name").textContent = `${vol.name} — ${vol.X}x${vol.Y}x${vol.Z}`;
}

/* The brain mask comes from T1 (data.py: brain = t1 > 0) and normalises
   T2 as well: both modalities share the same statistics, as in training. */
function normaliseVolumes() {
  if (!volT1) return;
  const mask = new Uint8Array(volT1.raw.length);
  for (let i = 0; i < mask.length; i++) mask[i] = volT1.raw[i] > 0 ? 1 : 0;
  volT1.norm = zscoreInMask(volT1.raw, mask);
  if (volT2) volT2.norm = zscoreInMask(volT2.raw, mask);
  occupancy = computeOccupancy(volT1);
}

async function acceptFiles(fileList) {
  clearError();
  const files = Array.from(fileList);
  const pick = (mod, ext) =>
    files.find((f) => new RegExp(`t${mod}\\b|t${mod}[._-]`, "i").test(f.name) &&
                         new RegExp(`\\.${ext}$`, "i").test(f.name));

  try {
    for (const [mod, target] of [["1", "t1"], ["2", "t2"]]) {
      const hdr = pick(mod, "hdr"), img = pick(mod, "img");
      if (!hdr || !img) continue;
      const vol = await readVolume(hdr, img);
      if (target === "t1") volT1 = vol; else volT2 = vol;
      showRow(target === "t1" ? "rowT1" : "rowT2", vol);
    }

    if (!volT1 && !volT2) {
      throw new Error("no file recognised: names must contain T1 or T2, and each volume needs its .hdr and .img pair");
    }
    if (volT1 && volT2 && volT1.raw.length !== volT2.raw.length) {
      throw new Error("the T1 and T2 volumes have different dimensions");
    }

    normaliseVolumes();

    const complete = !!(volT1 && volT2);
    $("dropZone").classList.toggle("filled", complete);
    markActive(".segmented .ex", "subject", null);
    // the image moves to the top, the drop area moves down
    $("shell").classList.toggle("loaded", complete);
    if (complete) $("btnBrowse").textContent = "Change volumes";
    enableWhenReady();
  } catch (e) {
    showError(e.message);
  }
}

function enableWhenReady() {
  if (!(volT1 && volT2)) return;
  const maxY = Math.min(volT1.Y, volT2.Y) - 1;
  const s = $("sliceSlider");
  s.max = maxY;
  s.value = Math.floor(maxY / 2);
  s.disabled = false;
  $("btnPrev").disabled = $("btnNext").disabled = false;
  $("panelSlice").classList.remove("off");
  $("panelDisplay").classList.remove("off");
  $("dimsLabel").textContent = `${volT1.X}×${volT1.Y}×${volT1.Z}`;
  updateSliceReadout();
  showTouchHint();
  // Greyscale first, so the image appears without waiting for inference.
  refreshBackground(+s.value);
  requestSegmentation();
}

function updateSliceReadout() {
  const s = $("sliceSlider");
  $("sliceNum").textContent = s.value;
  $("sliceMax").textContent = "/ " + s.max;
  drawProfile();
}

/* "Swipe to change slice" hint, shown once on touch devices. */
function showTouchHint() {
  if (!window.matchMedia("(pointer: coarse)").matches) return;
  const h = $("swipeHint");
  h.classList.add("on");
  setTimeout(() => h.classList.remove("on"), 2600);
}

/* Every way of changing slice — slider, buttons, keys, scrub bar, swipe —
   goes through here: readout, immediate greyscale, then inference. */
function setSlice(y) {
  const s = $("sliceSlider");
  if (s.disabled) return;
  s.value = Math.min(+s.max, Math.max(0, y));
  updateSliceReadout();
  refreshBackground(+s.value);
  requestSegmentation();
}

function step(delta) {
  setSlice(+$("sliceSlider").value + delta);
}

$("btnBrowse").addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", (e) => acceptFiles(e.target.files));
for (const b of document.querySelectorAll(".segmented .ex")) {
  b.addEventListener("click", () => loadSample(b.dataset.subject));
}

for (const b of document.querySelectorAll(".segmented .mdl")) {
  b.addEventListener("click", () => {
    if (b.dataset.variant !== currentModel) loadModel(b.dataset.variant);
  });
}

/* A missing element must never halt the script: a page and a script from
   two different deploys would otherwise freeze every button. */
window.addEventListener("error", (e) => {
  console.error("iSeg Viewer:", e.message);
});

const dropArea = $("dropZone");
["dragenter", "dragover"].forEach((ev) =>
  dropArea.addEventListener(ev, (e) => { e.preventDefault(); dropArea.classList.add("hover"); }));
["dragleave", "drop"].forEach((ev) =>
  dropArea.addEventListener(ev, (e) => { e.preventDefault(); dropArea.classList.remove("hover"); }));
dropArea.addEventListener("drop", (e) => acceptFiles(e.dataTransfer.files));

$("sliceSlider").addEventListener("input", () => setSlice(+$("sliceSlider").value));
$("btnPrev").addEventListener("click", () => step(-1));
$("btnNext").addEventListener("click", () => step(+1));

document.addEventListener("keydown", (e) => {
  // Only defer to fields where arrows already mean something natively.
  // On a button they mean nothing, so a click must not eat the keyboard.
  const tag = ((e.target && e.target.tagName) || "").toUpperCase();
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  if (e.key === "ArrowLeft" || e.key === "ArrowDown") { e.preventDefault(); step(-1); }
  if (e.key === "ArrowRight" || e.key === "ArrowUp") { e.preventDefault(); step(+1); }
});

/* The occupancy profile doubles as a scrub bar: clicking or dragging on
   it jumps straight to the slice under the pointer. */
function sliceFromProfile(clientX) {
  const { left, usable } = profileGeometry();
  const frac = Math.min(1, Math.max(0, (clientX - left - THUMB / 2) / usable));
  return Math.round(frac * +$("sliceSlider").max);
}

function scrubTo(clientX) {
  const target = sliceFromProfile(clientX);
  if (target !== +$("sliceSlider").value) setSlice(target);
}

const profileCanvas = $("occupancy");
profileCanvas.addEventListener("pointerdown", (e) => {
  if ($("sliceSlider").disabled) return;
  profileCanvas.setPointerCapture(e.pointerId);
  scrubTo(e.clientX);
});
profileCanvas.addEventListener("pointermove", (e) => {
  if (profileCanvas.hasPointerCapture(e.pointerId)) scrubTo(e.clientX);
});
profileCanvas.addEventListener("pointerup", (e) => {
  if (profileCanvas.hasPointerCapture(e.pointerId)) profileCanvas.releasePointerCapture(e.pointerId);
});

/* On touch devices, dragging horizontally across the image scrolls
   through slices — more direct than aiming for the slider. */
let dragStartX = null, dragStartSlice = 0;
const segWrap = $("wrapSeg");
segWrap.addEventListener("touchstart", (e) => {
  if ($("sliceSlider").disabled || e.touches.length !== 1) return;
  dragStartX = e.touches[0].clientX;
  dragStartSlice = +$("sliceSlider").value;
}, { passive: true });
segWrap.addEventListener("touchmove", (e) => {
  if (dragStartX === null) return;
  const delta = Math.round((e.touches[0].clientX - dragStartX) / 14);
  const target = Math.min(+$("sliceSlider").max, Math.max(0, dragStartSlice + delta));
  if (target !== +$("sliceSlider").value) setSlice(target);
}, { passive: true });
segWrap.addEventListener("touchend", () => { dragStartX = null; }, { passive: true });

$("opacitySlider").addEventListener("input", (e) => {
  $("opacityVal").textContent = e.target.value + " %";
  if (lastClasses && lastInput) {
    drawSegmentation(lastInput, lastClasses, +e.target.value / 100);
  }
});

for (const [id, mod] of [["btnT1", "t1"], ["btnT2", "t2"]]) {
  $(id).addEventListener("click", () => {
    displayedModality = mod;
    $("btnT1").classList.toggle("on", mod === "t1");
    $("btnT2").classList.toggle("on", mod === "t2");
    $("modLabel").textContent = mod.toUpperCase();
    refreshBackground(+$("sliceSlider").value);
  });
}

/* Over HTTPS the service worker caches the app, so the page keeps working
   without a network and can be installed to the home screen. */

if (location.protocol !== "file:" && "serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}

// The install button only exists when the browser offers the prompt.
let installPrompt = null;
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  installPrompt = e;
  $("installBar").hidden = false;
});
$("btnInstall").addEventListener("click", async () => {
  if (!installPrompt) return;
  installPrompt.prompt();
  await installPrompt.userChoice;
  installPrompt = null;
  $("installBar").hidden = true;
});

window.addEventListener("resize", () => { if (occupancy) drawProfile(); });

showModelSpec(DEFAULT_MODEL);
loadModel(DEFAULT_MODEL);
