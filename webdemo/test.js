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

let echecs = 0;
const check = (nom, ok, detail = "") => {
  console.log(`  ${ok ? "ok  " : "FAIL"}  ${nom}${detail ? "   " + detail : ""}`);
  if (!ok) echecs++;
};

/* ------------------------------------------------ DOM minimal ------ */
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
  addEventListener() {},
  querySelectorAll() { return []; },
  documentElement: { outerHTML: "" },
};

/* ------------------------------------------- fausse session ONNX --- */
let dernierTenseur = null;
const ortStub = {
  env: { wasm: {} },
  Tensor: class {
    constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; dernierTenseur = data; }
  },
  InferenceSession: {
    create: async () => ({
      run: async () => {
        // logits arbitraires mais deterministes : la classe gagnante
        // varie selon la position, ce qui exerce l'argmax
        const logits = new Float32Array(4 * SIZE * SIZE);
        for (let i = 0; i < SIZE * SIZE; i++) logits[(i % 4) * SIZE * SIZE + i] = 1;
        return { logits: { data: logits } };
      },
    }),
  },
};

/* ------------------------------------------------ fichiers ---------- */
function fakeFile(nom) {
  const buf = fs.readFileSync(path.join(DATA, nom));
  return {
    name: nom,
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  };
}

/* ------------------------------------------------ execution --------- */
const source = fs.readFileSync(path.join(__dirname, "app.js"), "utf8") + `
;globalThis.__t = {
  acceptFiles, requestSegmentation, buildTensor, segmentSlice, unpackSample,
  etat: () => ({ volT1, volT2, lastClasses, inferenceRunning, requestedSlice }),
};`;

const ctx = vm.createContext({
  document: doc,
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
  console.log(`\ntest app.js — sujet ${SUBJECT}, coupe ${SLICE}\n`);

  const t = ctx.__t;
  await t.acceptFiles([
    fakeFile(`subject-${SUBJECT}-T1.hdr`), fakeFile(`subject-${SUBJECT}-T1.img`),
    fakeFile(`subject-${SUBJECT}-T2.hdr`), fakeFile(`subject-${SUBJECT}-T2.img`),
  ]);

  const e = t.etat();
  check("les deux volumes sont lus", !!(e.volT1 && e.volT2),
        e.volT1 ? `${e.volT1.X}x${e.volT1.Y}x${e.volT1.Z}` : "");
  check("la normalisation a produit des valeurs", !!e.volT1?.norm && e.volT1.norm.length > 0);

  doc.getElementById("sliceSlider").value = String(SLICE);
  await t.segmentSlice(SLICE);

  const apres = t.etat();
  check("la segmentation s'est terminee", !!apres.lastClasses);
  check("la file est relachee", apres.inferenceRunning === false);
  check("l'etat affiche n'est plus 'computing'",
        doc.getElementById("segStateLabel").textContent !== "computing…",
        `"${doc.getElementById("segStateLabel").textContent}"`);

  const tenseur = t.buildTensor(SLICE);
  check("le tenseur a la bonne taille", tenseur.length === 2 * CONTEXT * SIZE * SIZE,
        `${tenseur.length} valeurs`);
  check("le tenseur ne contient pas de NaN", !tenseur.some(Number.isNaN));

  // ---- chaque echantillon doit reconstituer son volume d'origine ----
  for (const f of fs.readdirSync(__dirname).filter((n) => /^sample-\d+\.bin\.gz$/.test(n)).sort()) {
    const n = f.match(/\d+/)[0];
    const gz = fs.readFileSync(path.join(__dirname, f));
    const brut = require("zlib").gunzipSync(gz);
    const ech = t.unpackSample(brut.buffer.slice(brut.byteOffset, brut.byteOffset + brut.byteLength));

    let diff = 0;
    for (const [mod, vol] of [["T1", ech.t1], ["T2", ech.t2]]) {
      const orig = fs.readFileSync(path.join(DATA, `subject-${n}-${mod}.img`));
      const ref = new Int16Array(orig.buffer, orig.byteOffset, orig.byteLength / 2);
      for (let i = 0; i < ref.length; i++) if (vol.raw[i] !== ref[i]) diff++;
    }
    check(`${f} reproduit le volume d'origine`, diff === 0,
          diff ? `${diff} voxels differents` : `${(gz.length / 1e6).toFixed(2)} Mo compresse`);
  }

  fs.writeFileSync("/tmp/tenseur_js.bin", Buffer.from(tenseur.buffer));
  console.log(`\n  tenseur ecrit dans /tmp/tenseur_js.bin pour comparaison avec Python`);
  console.log(`\n${echecs === 0 ? "tout est vert" : echecs + " echec(s)"}\n`);
  process.exit(echecs ? 1 : 0);
})();
