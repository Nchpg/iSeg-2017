"""Exploration des donnees iSeg-2017.

Verifie les hypotheses du pipeline avant de coder l'entrainement :
geometrie et alignement T1/T2, proportions de classes, chevauchement
des intensites GM/WM (le "faible contraste" du sujet), profil de
remplissage des coupes.

Usage : LD_LIBRARY_PATH=... .venv/bin/python explore.py
"""

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent
TRAIN = ROOT / "iSeg-2017-Training"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# iSeg encode les tissus 10 / 150 / 250 dans le fichier label, le fond a 0
TISSUES = ["LCR", "GM", "WM"]
SUBJECTS = list(range(1, 11))

# Bornes communes pour tous les histogrammes z-scores
HIST_BINS = np.linspace(-3, 5, 200)
HIST_CENTERS = 0.5 * (HIST_BINS[:-1] + HIST_BINS[1:])


def load(subject, kind):
    return nib.load(TRAIN / f"subject-{subject}-{kind}.hdr")


def load_data(subject, kind, dtype=np.float32):
    """Les fichiers iSeg sont stockes en 4D avec une derniere dimension
    singleton ; on la supprime pour travailler en 3D."""
    return np.squeeze(load(subject, kind).get_fdata()).astype(dtype)


def zscore_in_mask(vol, mask):
    """Normalisation z-score calculee sur le cerveau uniquement.

    Inclure le fond tirerait la moyenne vers zero et rendrait la
    normalisation inoperante.
    """
    vals = vol[mask]
    return (vol - vals.mean()) / vals.std()


def bbox(mask):
    """Boite englobante du cerveau, par axe."""
    out = []
    for axis in range(3):
        proj = mask.any(axis=tuple(i for i in range(3) if i != axis))
        idx = np.where(proj)[0]
        out.append((int(idx[0]), int(idx[-1])))
    return out


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------- geometrie
section("1. GEOMETRIE ET ALIGNEMENT T1/T2")

geom_ok = True
for s in SUBJECTS:
    t1, t2, lab = load(s, "T1"), load(s, "T2"), load(s, "label")
    same_shape = t1.shape == t2.shape == lab.shape
    same_affine = np.allclose(t1.affine, t2.affine) and np.allclose(t1.affine, lab.affine)
    geom_ok &= same_shape and same_affine
    zooms = t1.header.get_zooms()[:3]
    print(f"  sujet {s:<3} shape={t1.shape[:3]}  spacing={tuple(round(z, 3) for z in zooms)}"
          f"  dtype={t1.get_data_dtype()}"
          f"  {'aligne' if same_shape and same_affine else 'DESALIGNE'}")

print(f"\n  -> tous les volumes partagent shape et affine : {geom_ok}")
print("     (condition necessaire pour empiler T1 et T2 en canaux)")

# ------------------------------------------------------- accumulation stats
section("2. PROPORTIONS DE CLASSES ET BOITES ENGLOBANTES")

counts = np.zeros(4, dtype=np.int64)          # fond, LCR, GM, WM
hists = {m: {t: np.zeros(len(HIST_CENTERS)) for t in TISSUES} for m in ("T1", "T2")}
bboxes = []
occupancy = []                                 # remplissage par coupe coronale
joint = {t: [] for t in TISSUES}               # echantillon (T1, T2) pour le nuage

rng = np.random.default_rng(0)

print(f"  {'sujet':<7}{'fond':>8}{'LCR':>8}{'GM':>8}{'WM':>8}   bbox (x, y, z)")
for s in SUBJECTS:
    t1 = load_data(s, "T1")
    t2 = load_data(s, "T2")
    lab = load_data(s, "label", np.int16)

    brain = t1 > 0
    t1n = zscore_in_mask(t1, brain)
    t2n = zscore_in_mask(t2, brain)

    masks = {"LCR": lab == 10, "GM": lab == 150, "WM": lab == 250}
    c = np.array([(lab == v).sum() for v in (0, 10, 150, 250)])
    counts += c
    frac = c / c.sum()

    bb = bbox(brain)
    bboxes.append(bb)
    print(f"  {s:<7}" + "".join(f"{f * 100:>7.2f}%" for f in frac)
          + f"   {bb[0]} {bb[1]} {bb[2]}")

    for t, m in masks.items():
        hists["T1"][t] += np.histogram(t1n[m], bins=HIST_BINS)[0]
        hists["T2"][t] += np.histogram(t2n[m], bins=HIST_BINS)[0]
        idx = rng.choice(m.sum(), size=min(4000, m.sum()), replace=False)
        joint[t].append(np.stack([t1n[m][idx], t2n[m][idx]], axis=1))

    # axe 1 = coronal (shape 144 x 192 x 256 -> y de taille 192)
    occupancy.append((lab > 0).sum(axis=(0, 2)) / (lab.shape[0] * lab.shape[2]))

frac_tot = counts / counts.sum()
print(f"\n  global :  fond {frac_tot[0]*100:.2f}%   LCR {frac_tot[1]*100:.2f}%"
      f"   GM {frac_tot[2]*100:.2f}%   WM {frac_tot[3]*100:.2f}%")
brain_frac = frac_tot[1:] / frac_tot[1:].sum()
print(f"  dans le cerveau :  LCR {brain_frac[0]*100:.1f}%   GM {brain_frac[1]*100:.1f}%"
      f"   WM {brain_frac[2]*100:.1f}%")
print(f"  desequilibre max entre tissus : facteur {brain_frac.max()/brain_frac.min():.1f}")

bb = np.array(bboxes)
print(f"\n  bbox cerveau (union sur les 10 sujets) :"
      f" x[{bb[:,0,0].min()}:{bb[:,0,1].max()}]"
      f" y[{bb[:,1,0].min()}:{bb[:,1,1].max()}]"
      f" z[{bb[:,2,0].min()}:{bb[:,2,1].max()}]")
vol_full = np.prod(np.squeeze(load(1, "T1").get_fdata()).shape)
ext = (bb[:,0,1].max()-bb[:,0,0].min(), bb[:,1,1].max()-bb[:,1,0].min(),
       bb[:,2,1].max()-bb[:,2,0].min())
print(f"  -> recadrage : {vol_full/1e6:.1f} M voxels vers {np.prod(ext)/1e6:.1f} M"
      f"  ({100*np.prod(ext)/vol_full:.0f}% du volume)")

# ------------------------------------------------------------ chevauchement
section("3. CHEVAUCHEMENT DES INTENSITES (le defi de faible contraste)")


def overlap(a, b):
    """Intersection d'histogrammes normalises : 1 = confondus, 0 = disjoints."""
    a, b = a / a.sum(), b / b.sum()
    return float(np.minimum(a, b).sum())


print("  intersection d'histogrammes (1 = indistinguable, 0 = separable)\n")
print(f"  {'paire':<12}{'T1':>8}{'T2':>8}")
for pair in [("GM", "WM"), ("LCR", "GM"), ("LCR", "WM")]:
    o1 = overlap(hists["T1"][pair[0]], hists["T1"][pair[1]])
    o2 = overlap(hists["T2"][pair[0]], hists["T2"][pair[1]])
    print(f"  {pair[0]}/{pair[1]:<8}{o1:>8.3f}{o2:>8.3f}")

# separabilite 2D : LDA a une dimension sur le couple (T1, T2) vs meilleur axe seul
gm = np.concatenate(joint["GM"])
wm = np.concatenate(joint["WM"])


def fisher(a, b):
    """Ratio de Fisher : plus il est grand, plus les classes se separent."""
    return (a.mean() - b.mean()) ** 2 / (a.var() + b.var())


f_t1 = fisher(gm[:, 0], wm[:, 0])
f_t2 = fisher(gm[:, 1], wm[:, 1])
# direction optimale (Fisher 2D)
d = gm.mean(0) - wm.mean(0)
S = np.cov(gm.T) + np.cov(wm.T)
w = np.linalg.solve(S, d)
p_gm, p_wm = gm @ w, wm @ w
f_2d = fisher(p_gm, p_wm)

print(f"\n  separabilite GM/WM (ratio de Fisher, plus haut = mieux)")
print(f"    T1 seul        : {f_t1:.3f}")
print(f"    T2 seul        : {f_t2:.3f}")
print(f"    T1 + T2 (2D)   : {f_2d:.3f}"
      f"   -> gain x{f_2d / max(f_t1, f_t2):.2f} sur la meilleure modalite seule")
print("  (justifie l'entree a 2 modalites plutot qu'une seule)")

# ----------------------------------------------------------- coupes utiles
section("4. PROFIL DE REMPLISSAGE DES COUPES (axe coronal)")

occ = np.stack(occupancy)
for thr in (0.0, 0.01, 0.02, 0.05):
    n = (occ > thr).sum(axis=1)
    print(f"  seuil {thr:>5.0%} de voxels annotes : {n.mean():>6.1f} coupes/sujet"
          f"  (min {n.min()}, max {n.max()})  -> {int(n.sum())} exemples sur 10 sujets")
print("\n  -> le seuil retenu filtre les coupes quasi vides des extremites")

# -------------------------------------------------------------- figures
section("5. FIGURES")

# 5a. histogrammes
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
colors = {"LCR": "#4C9BE8", "GM": "#E8934C", "WM": "#5FBF77"}
for ax, mod in zip(axes, ("T1", "T2")):
    for t in TISSUES:
        h = hists[mod][t] / hists[mod][t].sum()
        ax.plot(HIST_CENTERS, h, label=t, color=colors[t], lw=1.8)
        ax.fill_between(HIST_CENTERS, h, alpha=0.18, color=colors[t])
    ax.set_title(f"{mod} — distribution des intensites par tissu")
    ax.set_xlabel("intensite normalisee (z-score dans le cerveau)")
    ax.set_ylabel("densite")
    ax.legend()
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "histogrammes_intensite.png", dpi=130)
plt.close(fig)

# 5b. nuage conjoint T1 x T2
fig, ax = plt.subplots(figsize=(6, 5.5))
for t in TISSUES:
    j = np.concatenate(joint[t])
    sub = j[rng.choice(len(j), 6000, replace=False)]
    ax.scatter(sub[:, 0], sub[:, 1], s=1.5, alpha=0.15, color=colors[t], label=t)
ax.set_xlabel("T1 normalise")
ax.set_ylabel("T2 normalise")
ax.set_title("Espace conjoint T1 x T2\n(les nuages se separent mieux qu'en projection 1D)")
leg = ax.legend(markerscale=8)
for lh in leg.legend_handles:
    lh.set_alpha(1)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "nuage_T1_T2.png", dpi=130)
plt.close(fig)

# 5c. coupes exemples dans les 3 orientations
s = 1
t1 = load_data(s, "T1")
t2 = load_data(s, "T2")
lab = load_data(s, "label", np.int16)
cmap = matplotlib.colors.ListedColormap(["black", colors["LCR"], colors["GM"], colors["WM"]])
lut = np.zeros(251, dtype=np.uint8)
lut[10], lut[150], lut[250] = 1, 2, 3
labc = lut[lab]

views = {"sagittal": (0, t1.shape[0] // 2), "coronal": (1, t1.shape[1] // 2),
         "axial": (2, t1.shape[2] // 2)}
fig, axes = plt.subplots(3, 3, figsize=(10, 11))
for r, (name, (axis, i)) in enumerate(views.items()):
    sl = [slice(None)] * 3
    sl[axis] = i
    for c, (img, title, kw) in enumerate([
            (t1[tuple(sl)].T, "T1", dict(cmap="gray")),
            (t2[tuple(sl)].T, "T2", dict(cmap="gray")),
            (labc[tuple(sl)].T, "label", dict(cmap=cmap, vmin=0, vmax=3))]):
        axes[r, c].imshow(np.flipud(img), **kw)
        axes[r, c].set_title(f"{name} — {title}", fontsize=10)
        axes[r, c].axis("off")
fig.suptitle(f"sujet {s} — coupes centrales", fontsize=13)
fig.tight_layout()
fig.savefig(FIG / "coupes_exemples.png", dpi=130)
plt.close(fig)

# 5d. profil de remplissage
fig, ax = plt.subplots(figsize=(8, 4))
for i, o in enumerate(occ):
    ax.plot(o * 100, lw=1, alpha=0.7, label=f"sujet {i+1}" if i < 3 else None)
ax.axhline(2, color="crimson", ls="--", lw=1.2, label="seuil 2%")
ax.set_xlabel("indice de coupe coronale")
ax.set_ylabel("% de voxels annotes")
ax.set_title("Remplissage des coupes — les extremites sont quasi vides")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "remplissage_coupes.png", dpi=130)
plt.close(fig)

for f in sorted(FIG.glob("*.png")):
    print(f"  ecrit : figures/{f.name}")
print()
