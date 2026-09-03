"use strict";

/* iSeg Viewer — segmentation 2.5D dans le navigateur.

   Tout le calcul se fait sur l'appareil : le modele ONNX est charge une
   fois, puis chaque coupe est segmentee en WebAssembly. Aucune image ne
   quitte la machine.

   Le pretraitement reproduit exactement iseg/data.py cote Python :
   masque cerebral T1 > 0, z-score dans ce masque applique aux deux
   modalites, recadrage fixe, empilement de 5 coupes de contexte par
   modalite. Toute divergence ici ferait chuter le Dice sans prevenir. */

/* ============================ constantes ============================
   Doivent rester en phase avec iseg/data.py (CROP_X, CROP_Z) et le
   checkpoint exporte : variante separable, contexte 5, T1+T2 = 10
   canaux, entree 144x144. */
const CROP_X = [0, 144];
const CROP_Z = [72, 216];
const CONTEXT = 5;
const SIZE = 144;
const MODELE = "model.onnx";

const COULEURS = {
  1: [0x4c, 0x9b, 0xe8],   // LCR
  2: [0xe8, 0x93, 0x4c],   // substance grise
  3: [0x5f, 0xbf, 0x77],   // substance blanche
};

const $ = (id) => document.getElementById(id);

let session = null;
let volT1 = null;          // {X, Y, Z, raw: Int16Array, norm: Float32Array, name}
let volT2 = null;
let vueModalite = "t1";    // modalite AFFICHEE (le reseau utilise toujours les deux)
let dernieresClasses = null;
let derniereEntree = null;

/* ============================== erreurs ============================== */

function erreur(msg) {
  $("errBanner").textContent = msg;
  $("errBanner").classList.add("on");
}
function effacerErreur() { $("errBanner").classList.remove("on"); }

function etatModele(classe, texte) {
  $("modelChip").className = "chip " + classe;
  $("modelChipText").textContent = texte;
}

/* =========================== chargement modele ======================= */

async function chargerModele() {
  ort.env.wasm.numThreads = 1;   // evite l'exigence COOP/COEP d'un hebergement simple
  try {
    const t0 = performance.now();
    session = await ort.InferenceSession.create(MODELE, { executionProviders: ["wasm"] });
    etatModele("ok", `modele pret (${(performance.now() - t0).toFixed(0)} ms)`);
  } catch (e) {
    etatModele("err", "modele indisponible");
    erreur(
      location.protocol === "file:"
        ? "Ouverte en file://, la page ne peut pas charger model.onnx : les navigateurs bloquent " +
          "l'acces d'un fichier local a un autre. Sers le dossier (python3 -m http.server) ou " +
          "utilise la version deployee."
        : "Impossible de charger le modele : " + e.message
    );
  }
}

/* ======================== lecture Analyze 7.5 ========================
   En-tete de 348 octets, little-endian, donnees brutes sans decalage.
   L'ordre de stockage des voxels est x le plus rapide, puis y, puis z
   (verifie contre nibabel sur les fichiers du challenge). */

function lireEntete(buf) {
  const dv = new DataView(buf);
  const dim = [];
  for (let i = 0; i < 8; i++) dim.push(dv.getInt16(40 + i * 2, true));
  return { X: dim[1], Y: dim[2], Z: dim[3], datatype: dv.getInt16(70, true), bitpix: dv.getInt16(72, true) };
}

async function lireVolume(fichierHdr, fichierImg) {
  const { X, Y, Z, datatype, bitpix } = lireEntete(await fichierHdr.arrayBuffer());
  if (datatype !== 4 || bitpix !== 16) {
    throw new Error(`type de donnees inattendu (datatype=${datatype}, bitpix=${bitpix}), entier 16 bits attendu`);
  }
  if (X < CROP_X[1] || Z < CROP_Z[1]) {
    throw new Error(`dimensions inattendues (${X}x${Y}x${Z}), geometrie iSeg-2017 (144x192x256) attendue`);
  }
  const buf = await fichierImg.arrayBuffer();
  // Ordre d'octets little-endian : vrai sur tous les navigateurs deployes.
  return { X, Y, Z, raw: new Int16Array(buf, 0, X * Y * Z),
           name: fichierHdr.name.replace(/\.hdr$/i, "") };
}

/* ============================ pretraitement ========================== */

function zscoreDansMasque(raw, masque) {
  let somme = 0, n = 0;
  for (let i = 0; i < raw.length; i++) if (masque[i]) { somme += raw[i]; n++; }
  const moy = somme / n;
  let carres = 0;
  for (let i = 0; i < raw.length; i++) if (masque[i]) { const d = raw[i] - moy; carres += d * d; }
  const ecart = Math.sqrt(carres / n) || 1e-8;
  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = (raw[i] - moy) / ecart;
  return out;
}

/* Extrait la coupe coronale (x, z) a Y fixe, recadree, ordonnee [x][z]
   pour correspondre au flatten C de stack_context() cote Python. */
function extraireCoupe(vol, source, y) {
  const out = new Float32Array(SIZE * SIZE);
  const [x0] = CROP_X, [z0] = CROP_Z;
  for (let xi = 0; xi < SIZE; xi++) {
    const base = (xi + x0) + y * vol.X;
    for (let zi = 0; zi < SIZE; zi++) out[xi * SIZE + zi] = source[base + (zi + z0) * vol.X * vol.Y];
  }
  return out;
}

/* Empile 5 coupes voisines par modalite -> 10 canaux. Aux extremites du
   volume on replique la coupe de bord : remplir de zeros creerait un
   bord noir artificiel, information trompeuse pour le reseau. */
function construireTenseur(y) {
  const demi = CONTEXT >> 1;
  const out = new Float32Array(2 * CONTEXT * SIZE * SIZE);
  let ptr = 0;
  for (const vol of [volT1, volT2]) {
    for (let k = -demi; k <= demi; k++) {
      const yy = Math.min(Math.max(y + k, 0), vol.Y - 1);
      out.set(extraireCoupe(vol, vol.norm, yy), ptr);
      ptr += SIZE * SIZE;
    }
  }
  return out;
}

/* ================================ rendu ============================== */

function coupeEnImage(vol, y) {
  const brut = extraireCoupe(vol, vol.raw, y);
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < brut.length; i++) { if (brut[i] < mn) mn = brut[i]; if (brut[i] > mx) mx = brut[i]; }
  const etendue = (mx - mn) || 1;
  const img = new Uint8ClampedArray(SIZE * SIZE * 4);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const g = ((brut[i] - mn) / etendue) * 255;
    img[i * 4] = img[i * 4 + 1] = img[i * 4 + 2] = g;
    img[i * 4 + 3] = 255;
  }
  return img;
}

function dessinerEntree(y) {
  const img = coupeEnImage(vueModalite === "t1" ? volT1 : volT2, y);
  $("canvasT1").getContext("2d").putImageData(new ImageData(img, SIZE, SIZE), 0, 0);
  $("emptyT1").style.display = "none";
  return img;
}

function dessinerSegmentation(base, classes, opacite) {
  const out = new Uint8ClampedArray(base);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const c = classes[i];
    if (c === 0) continue;
    const [r, g, b] = COULEURS[c];
    out[i * 4]     = out[i * 4]     * (1 - opacite) + r * opacite;
    out[i * 4 + 1] = out[i * 4 + 1] * (1 - opacite) + g * opacite;
    out[i * 4 + 2] = out[i * 4 + 2] * (1 - opacite) + b * opacite;
  }
  $("canvasSeg").getContext("2d").putImageData(new ImageData(out, SIZE, SIZE), 0, 0);
  $("emptySeg").style.display = "none";
}

function majStatistiques(classes) {
  const n = [0, 0, 0, 0];
  for (let i = 0; i < classes.length; i++) n[classes[i]]++;
  const cerveau = n[1] + n[2] + n[3];
  const pct = cerveau ? [n[1], n[2], n[3]].map((c) => (100 * c) / cerveau) : [0, 0, 0];
  const barres = $("statBar").children;
  ["pctLCR", "pctGM", "pctWM"].forEach((id, i) => {
    barres[i].style.width = pct[i].toFixed(1) + "%";
    $(id).textContent = pct[i].toFixed(1) + " %";
  });
}

/* ============================== inference ============================ */

async function segmenter() {
  if (!session || !volT1 || !volT2) return;
  effacerErreur();
  const y = +$("sliceSlider").value;
  $("segStateLabel").textContent = "calcul...";

  const t0 = performance.now();
  derniereEntree = dessinerEntree(y);
  const donnees = construireTenseur(y);
  const msPrep = performance.now() - t0;

  try {
    const t1 = performance.now();
    const sortie = await session.run({
      input: new ort.Tensor("float32", donnees, [1, 2 * CONTEXT, SIZE, SIZE]),
    });
    const msInf = performance.now() - t1;

    const logits = sortie.logits.data;   // [1, 4, 144, 144]
    const classes = new Uint8Array(SIZE * SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      let best = 0, val = -Infinity;
      for (let c = 0; c < 4; c++) {
        const v = logits[c * SIZE * SIZE + i];
        if (v > val) { val = v; best = c; }
      }
      classes[i] = best;
    }

    dernieresClasses = classes;
    dessinerSegmentation(derniereEntree, classes, +$("opacitySlider").value / 100);
    majStatistiques(classes);

    $("tPrep").innerHTML = msPrep.toFixed(1) + "<small>ms</small>";
    $("tInfer").innerHTML = msInf.toFixed(1) + "<small>ms</small>";
    $("segStateLabel").textContent = "coupe " + y;
    $("panelStats").classList.remove("off");
  } catch (e) {
    $("segStateLabel").textContent = "erreur";
    erreur("Echec de l'inference : " + e.message);
  }
}

/* =============================== fichiers ============================
   Les 4 fichiers d'un sujet se deposent d'un coup : T1 et T2 sont
   reconnus par leur nom, puis apparies .hdr avec .img. */

async function accepterFichiers(liste) {
  effacerErreur();
  const fichiers = Array.from(liste);
  const trouve = (mod, ext) =>
    fichiers.find((f) => new RegExp(`t${mod}\\b|t${mod}[._-]`, "i").test(f.name) &&
                         new RegExp(`\\.${ext}$`, "i").test(f.name));

  try {
    for (const [mod, cible] of [["1", "t1"], ["2", "t2"]]) {
      const hdr = trouve(mod, "hdr"), img = trouve(mod, "img");
      if (!hdr || !img) continue;
      const vol = await lireVolume(hdr, img);
      if (cible === "t1") volT1 = vol; else volT2 = vol;
      const ligne = $(cible === "t1" ? "rowT1" : "rowT2");
      ligne.classList.add("ok");
      ligne.querySelector(".nom").textContent = `${vol.name} — ${vol.X}x${vol.Y}x${vol.Z}`;
    }

    if (!volT1 && !volT2) {
      throw new Error("aucun fichier reconnu : les noms doivent contenir T1 ou T2, avec les paires .hdr et .img");
    }
    if (volT1 && volT2 && volT1.raw.length !== volT2.raw.length) {
      throw new Error("les volumes T1 et T2 n'ont pas les memes dimensions");
    }

    // Le masque cerebral vient du T1 (data.py : brain = t1 > 0) et sert
    // aussi a normaliser le T2 : les deux modalites partagent la meme
    // statistique, comme a l'entrainement.
    if (volT1) {
      const masque = new Uint8Array(volT1.raw.length);
      for (let i = 0; i < masque.length; i++) masque[i] = volT1.raw[i] > 0 ? 1 : 0;
      volT1.norm = zscoreDansMasque(volT1.raw, masque);
      if (volT2) volT2.norm = zscoreDansMasque(volT2.raw, masque);
    }

    $("dropZone").classList.toggle("rempli", !!(volT1 && volT2));
    activerSiPret();
  } catch (e) {
    erreur(e.message);
  }
}

function activerSiPret() {
  if (!(volT1 && volT2)) return;
  const maxY = Math.min(volT1.Y, volT2.Y) - 1;
  const s = $("sliceSlider");
  s.max = maxY;
  s.value = Math.floor(maxY / 2);
  s.disabled = false;
  $("btnPrev").disabled = $("btnNext").disabled = false;
  $("panelCoupe").classList.remove("off");
  $("dimsLabel").textContent = `${volT1.X}×${volT1.Y}×${volT1.Z}`;
  majAffichageCoupe();
  segmenter();
}

function majAffichageCoupe() {
  const s = $("sliceSlider");
  $("sliceNum").textContent = s.value;
  $("sliceMax").textContent = "/ " + s.max;
}

function allerA(delta) {
  const s = $("sliceSlider");
  if (s.disabled) return;
  s.value = Math.min(+s.max, Math.max(0, +s.value + delta));
  majAffichageCoupe();
  segmenter();
}

/* ============================== interface ============================ */

$("btnParcourir").addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", (e) => accepterFichiers(e.target.files));

const zone = $("dropZone");
["dragenter", "dragover"].forEach((ev) =>
  zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("survol"); }));
["dragleave", "drop"].forEach((ev) =>
  zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("survol"); }));
zone.addEventListener("drop", (e) => accepterFichiers(e.dataTransfer.files));

let minuteur = null;
$("sliceSlider").addEventListener("input", () => {
  majAffichageCoupe();
  clearTimeout(minuteur);
  minuteur = setTimeout(segmenter, 110);
});
$("btnPrev").addEventListener("click", () => allerA(-1));
$("btnNext").addEventListener("click", () => allerA(+1));

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, button")) return;
  if (e.key === "ArrowLeft" || e.key === "ArrowDown") { e.preventDefault(); allerA(-1); }
  if (e.key === "ArrowRight" || e.key === "ArrowUp") { e.preventDefault(); allerA(+1); }
});

$("opacitySlider").addEventListener("input", (e) => {
  $("opacityVal").textContent = e.target.value + " %";
  if (dernieresClasses && derniereEntree) {
    dessinerSegmentation(derniereEntree, dernieresClasses, +e.target.value / 100);
  }
});

for (const [id, mod] of [["btnT1", "t1"], ["btnT2", "t2"]]) {
  $(id).addEventListener("click", () => {
    vueModalite = mod;
    $("btnT1").classList.toggle("on", mod === "t1");
    $("btnT2").classList.toggle("on", mod === "t2");
    if (volT1 && volT2) {
      derniereEntree = dessinerEntree(+$("sliceSlider").value);
      if (dernieresClasses) {
        dessinerSegmentation(derniereEntree, dernieresClasses, +$("opacitySlider").value / 100);
      }
    }
  });
}

/* ============================= hors ligne ============================
   Deployee en HTTPS, la page s'installe et fonctionne sans reseau une
   fois le moteur ONNX Runtime mis en cache — ce qui suppose une
   premiere segmentation en ligne. */

function annonce(couleur, texte) {
  $("etatPoint").style.color = couleur;
  $("etatTexte").textContent = texte;
}

if (location.protocol === "file:") {
  annonce("var(--danger)", "ouverte en file:// — sers le dossier ou utilise la version deployee");
} else if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").then((reg) => {
    const pret = reg.active && navigator.serviceWorker.controller;
    annonce(pret ? "var(--wm)" : "var(--accent)",
            pret ? "hors ligne : pret, la page fonctionne sans reseau"
                 : "hors ligne : cache en cours, lance une segmentation pour finir");
  }).catch(() => annonce("var(--danger)", "hors ligne indisponible (service worker refuse)"));
} else {
  annonce("var(--text-faint)", "hors ligne indisponible sur ce navigateur");
}

let invite = null;
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  invite = e;
  $("btnInstaller").hidden = false;
});
$("btnInstaller").addEventListener("click", async () => {
  if (!invite) return;
  invite.prompt();
  await invite.userChoice;
  invite = null;
  $("btnInstaller").hidden = true;
});

chargerModele();
