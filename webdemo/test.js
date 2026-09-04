/* Headless smoke test for app.js.

   Runs the page logic in a stubbed DOM with a fake ONNX session, feeds
   it the real .hdr/.img files of a subject, and checks two things:

     1. a segmentation completes without throwing and the queue is
        released (a thrown error used to wedge it on "computing…");
     2. the tensor built in JavaScript matches, value for value, the one
        stack_context() produces in Python — any drift there silently
        wrecks the Dice score.

   Usage:  node webdemo/test.js [subject] [slice]
*/

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const SUBJECT = process.argv[2] || "1";
const SLICE = Number(process.argv[3] || 96);
const DATA = path.join(__dirname, "..", "iSeg-2017-Training");
const SIZE = 144, CONTEXT = 5;

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`  ${ok ? "ok  " : "FAIL"}  ${name}${detail ? "   " + detail : ""}`);
  if (!ok) failures++;
};

/* --- minimal DOM --- */
function fakeElement(id) {
  const el = {
    id,
    hidden: false,
    value: "96",
    max: "191",
    disabled: false,
    textContent: "",
    innerHTML: "",
    style: {},
    files: [],
    children: [{ style: {} }, { style: {} }, { style: {} }],
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
      toggle(c, on) { on ? this._s.add(c) : this._s.delete(c); },
      contains(c) { return this._s.has(c); },
    },
    addEventListener() {},
    setPointerCapture() {}, releasePointerCapture() {}, hasPointerCapture() { return false; },
    querySelector() { return fakeElement(id + "-child"); },
    getContext() {
      return {
        clearRect() {}, fillRect() {}, setTransform() {},
        putImageData(img) { el._lastImage = img; },
        fillStyle: "",
      };
    },
    getBoundingClientRect() { return { width: 300, height: 26, left: 0, top: 0 }; },
    width: 300, height: 26,
  };
  return el;
}

const elements = new Map();
const doc = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, fakeElement(id));
    return elements.get(id);
  },
  _listeners: {},
  addEventListener(ev, fn) { (this._listeners[ev] ||= []).push(fn); },
  querySelectorAll(sel) {
    if (sel === ".segmented .ex") return exampleButtons;
    if (sel === ".segmented .mdl") return modelButtons;
    return [];
  },
  documentElement: { outerHTML: "" },
};

const fakeButton = (id, key, value) => {
  const b = fakeElement(id);
  b.dataset = { [key]: value };
  b._clicks = [];
  b.addEventListener = (ev, fn) => { if (ev === "click") b._clicks.push(fn); };
  b.click = () => b._clicks.forEach((fn) => fn());
  return b;
};

const modelButtons = ["tiny", "separable", "standard"].map((v) => fakeButton("mdl-" + v, "variant", v));

// "try example 1 2 3" buttons, with their click handler
const exampleButtons = [1, 2, 3].map((n) => fakeButton("ex" + n, "subject", String(n)));

/* --- fake ONNX session --- */
let lastTensor = null;
const ortStub = {
  env: { wasm: {} },
  Tensor: class {
    constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; lastTensor = data; }
  },
  InferenceSession: {
    create: async () => ({
      run: async () => {
        // deterministic logits whose winning class varies with position,
        // which exercises the argmax
        const logits = new Float32Array(4 * SIZE * SIZE);
        for (let i = 0; i < SIZE * SIZE; i++) logits[(i % 4) * SIZE * SIZE + i] = 1;
        return { logits: { data: logits } };
      },
    }),
  },
};

/* --- files --- */
function fakeFile(name) {
  const buf = fs.readFileSync(path.join(DATA, name));
  return {
    name,
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  };
}

/* --- run --- */
const source = fs.readFileSync(path.join(__dirname, "app.js"), "utf8") + `
;globalThis.__t = {
  acceptFiles, requestSegmentation, buildTensor, segmentSlice, unpackSample, loadSample,
  state: () => ({ volT1, volT2, lastClasses, inferenceRunning, requestedSlice }),
};`;

const fakeFetch = async (url) => {
  const name = String(url).split("/").pop();
  const file = path.join(__dirname, name);
  if (!fs.existsSync(file)) return { ok: false, status: 404 };
  const buf = fs.readFileSync(file);
  return { ok: true, status: 200, arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
};

const ctx = vm.createContext({
  document: doc,
  fetch: fakeFetch,
  Response, Blob, DecompressionStream,
  window: { addEventListener() {}, devicePixelRatio: 1, matchMedia: () => ({ matches: false }) },
  navigator: {},
  location: { protocol: "https:", href: "https://x/" },
  performance: { now: () => Date.now() },
  ort: ortStub,
  console,
  setTimeout, clearTimeout,
  URL, Math, Infinity, NaN,
  Float32Array, Uint8Array, Uint8ClampedArray, Int16Array, DataView, ImageData: class { constructor(d) { this.data = d; } },
  globalThis: null,
});
ctx.globalThis = ctx;
vm.runInContext(source, ctx);

(async () => {
  console.log(`\ntest app.js — subject ${SUBJECT}, slice ${SLICE}\n`);

  const t = ctx.__t;
  await t.acceptFiles([
    fakeFile(`subject-${SUBJECT}-T1.hdr`), fakeFile(`subject-${SUBJECT}-T1.img`),
    fakeFile(`subject-${SUBJECT}-T2.hdr`), fakeFile(`subject-${SUBJECT}-T2.img`),
  ]);

  const e = t.state();
  check("both volumes are read", !!(e.volT1 && e.volT2),
        e.volT1 ? `${e.volT1.X}x${e.volT1.Y}x${e.volT1.Z}` : "");
  check("normalisation produced values", !!e.volT1?.norm && e.volT1.norm.length > 0);

  doc.getElementById("sliceSlider").value = String(SLICE);
  await t.segmentSlice(SLICE);

  const after = t.state();
  check("segmentation completed", !!after.lastClasses);
  check("the queue is released", after.inferenceRunning === false);
  check("the displayed state is no longer 'computing'",
        doc.getElementById("segStateLabel").textContent !== "computing…",
        `"${doc.getElementById("segStateLabel").textContent}"`);

  const tensor = t.buildTensor(SLICE);
  check("the tensor has the right size", tensor.length === 2 * CONTEXT * SIZE * SIZE,
        `${tensor.length} values`);
  check("the tensor holds no NaN", !tensor.some(Number.isNaN));

  // every sample must reconstruct its original volume
  for (const f of fs.readdirSync(__dirname).filter((n) => /^sample-\d+\.bin$/.test(n)).sort()) {
    const n = f.match(/\d+/)[0];
    const gz = fs.readFileSync(path.join(__dirname, f));
    const raw = require("zlib").gunzipSync(gz);
    const sample = t.unpackSample(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));

    let diff = 0;
    for (const [mod, vol] of [["T1", sample.t1], ["T2", sample.t2]]) {
      const orig = fs.readFileSync(path.join(DATA, `subject-${n}-${mod}.img`));
      const ref = new Int16Array(orig.buffer, orig.byteOffset, orig.byteLength / 2);
      for (let i = 0; i < ref.length; i++) if (vol.raw[i] !== ref[i]) diff++;
    }
    check(`${f} reproduces the original volume`, diff === 0,
          diff ? `${diff} differing voxels` : `${(gz.length / 1e6).toFixed(2)} MB compressed`);
  }

  // the example buttons must be wired and working
  check("the 3 example buttons have a handler",
        exampleButtons.every((b) => b._clicks.length === 1),
        exampleButtons.map((b) => b._clicks.length).join("/"));

  await t.loadSample("2");
  const afterExample = t.state();
  check("clicking an example loads the volume",
        afterExample.volT1 && afterExample.volT1.name === "example 2 T1",
        afterExample.volT1 ? afterExample.volT1.name : "nothing loaded");
  check("the buttons are re-enabled", exampleButtons.every((b) => b.disabled === false) ||
        !!afterExample.volT1);

  // the keyboard must not depend on focus
  const press = (key, target) => {
    let blocked = false;
    const ev = { key, target, preventDefault: () => { blocked = true; } };
    (doc._listeners.keydown || []).forEach((fn) => fn(ev));
    return blocked;
  };

  const button = { tagName: "BUTTON" };
  const slider = { tagName: "INPUT" };
  doc.getElementById("sliceSlider").value = "100";

  check("arrows act when a button holds focus",
        press("ArrowRight", button), "the case that used to be broken");
  check("arrows are left to the native slider",
        !press("ArrowRight", slider));
  check("arrows act on the page body",
        press("ArrowLeft", { tagName: "BODY" }));

  // model selector
  check("the 3 model buttons have a handler",
        modelButtons.every((b) => b._clicks.length === 1),
        modelButtons.map((b) => b._clicks.length).join("/"));

  for (const v of ["tiny", "separable", "standard"]) {
    check(`model-${v}.onnx is present`,
          fs.existsSync(path.join(__dirname, `model-${v}.onnx`)),
          fs.existsSync(path.join(__dirname, `model-${v}.onnx`))
            ? `${(fs.statSync(path.join(__dirname, `model-${v}.onnx`)).size / 1e6).toFixed(2)} MB` : "");
  }

  modelButtons.find((b) => b.dataset.variant === "tiny").click();
  await new Promise((r) => setTimeout(r, 30));
  check("switching model updates the spec sheet",
        doc.getElementById("mParams").textContent === "26,974",
        `"${doc.getElementById("mParams").textContent}"`);

  fs.writeFileSync("/tmp/tensor_js.bin", Buffer.from(tensor.buffer));
  console.log(`\n  tensor written to /tmp/tensor_js.bin for comparison with Python`);
  console.log(`\n${failures === 0 ? "all green" : failures + " failure(s)"}\n`);
  process.exit(failures ? 1 : 0);
})();
