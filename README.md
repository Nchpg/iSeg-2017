# iSeg-2017 — segmentation frugale du cerveau du nourrisson

Segmentation en trois tissus (LCR, substance grise, substance blanche) sur IRM T1 et
T2 de bébés de 6 mois, avec un objectif de déploiement mobile ou embarqué.

## Protocole d'évaluation

Les 13 sujets du dossier `iSeg-2017-Testing/` **n'ont pas d'étiquettes** : elles n'ont
jamais été publiées, elles servaient au classement du challenge. Toute évaluation
chiffrée se fait donc en **validation croisée 5 blocs sur les 10 sujets
d'entraînement** (8 sujets d'entraînement, 2 de validation par bloc). Le dossier de
test ne sert qu'à produire des prédictions qualitatives et à mesurer un temps
d'inférence.

Le découpage est fait **par sujet**, jamais par coupe : deux coupes voisines d'un
même volume sont quasi identiques, les mélanger entre entraînement et validation
donnerait un Dice artificiellement élevé.

Le Dice se calcule **par volume**, après réassemblage de toutes les coupes.

## Ce que l'exploration a établi

`python explore.py` produit quatre figures dans `figures/` et le rapport chiffré
ci-dessous.

**Géométrie.** 10 sujets, 144×192×256, 1 mm isotrope. T1, T2 et label partagent la
même matrice affine : les modalités sont déjà recalées, aucun réalignement à faire.

**Répartition des classes**, à l'intérieur du cerveau : LCR 22 %, GM 47 %, WM 31 %.
Déséquilibre modéré (facteur 2,2), qui ne justifie pas de pondération agressive.

**Le faible contraste, quantifié.** Intersection des histogrammes d'intensité
(1 = distributions confondues, 0 = séparables) :

| paire | T1 | T2 |
|---|---|---|
| **GM / WM** | 0,702 | **0,901** |
| LCR / GM | 0,362 | 0,360 |
| LCR / WM | 0,268 | 0,347 |

Ratio de Fisher pour GM/WM : **0,269 en T1 seul, 0,001 en T2 seul**, et 0,297 pour la
meilleure combinaison linéaire des deux — soit 10 % de gain seulement.

Conséquence structurante : à 6 mois, **aucune combinaison d'intensités ne sépare GM et
WM**. C'est la phase iso-intense, et elle rend le contexte spatial indispensable
plutôt qu'accessoire. Le T2 est conservé (son coût est dérisoire, et un réseau
convolutif exploite des relations non linéaires que le critère de Fisher ne capte
pas), mais son apport réel doit être **mesuré par ablation**, pas supposé.

Attendu : le Dice de la WM sera le plus bas des trois.

## Choix de conception

| Décision | Justification |
|---|---|
| Orientation **coronale** | symétrie gauche/droite visible dans le plan : repère anatomique, et le flip horizontal devient une augmentation légitime |
| Recadrage fixe **144×144** | couvre la boîte englobante du cerveau avec 13-17 voxels de marge ; divisible par 16, donc ni redimensionnement ni remplissage |
| **Pas de correction N4** | SimpleITK ne s'embarque pas dans un navigateur ; compensé par une augmentation qui simule des champs de biais. Arbitrage assumé au profit d'un prétraitement reproductible à l'inférence |
| Normalisation z-score **dans le masque cérébral** | inclure le fond écraserait la statistique sous des millions de voxels vides |
| **2.5D** plutôt que 3D | aucune convolution 3D, donc un modèle que les runtimes mobiles savent accélérer et quantifier ; les convolutions 3D sont mal supportées par NNAPI et Core ML |
| Perte **Dice + entropie croisée** | l'entropie croisée donne des gradients stables au démarrage, le Dice optimise directement la métrique et ignore la taille des classes |

## Modèles

Un U-Net 2D dont les canaux d'entrée portent la troisième dimension : `context`
coupes coronales adjacentes par modalité (5 par défaut, soit 10 canaux avec T1+T2).

| variante | description | paramètres | MACs/coupe |
|---|---|---|---|
| `standard` | convolutions 3×3 pleines, base 16, 4 niveaux | 1 943 636 | 981 M |
| `separable` | depthwise 3×3 + pointwise 1×1 | 386 782 | 174 M |
| `tiny` | separable, base 8, 3 niveaux | 26 974 | 43 M |

Mesuré pour 10 canaux d'entrée en 144×144. Le rapport entre les extrêmes est de
**72× sur les paramètres et 23× sur le calcul** — de quoi tracer une courbe de
compromis lisible.

## Utilisation

### Environnement local (exploration seulement)

Sous NixOS, les wheels manylinux ont besoin de `libstdc++` et `zlib` dans le chemin
des bibliothèques ; `env.sh` s'en charge.

```bash
uv venv .venv
uv pip install --python .venv/bin/python numpy nibabel matplotlib
source env.sh
.venv/bin/python explore.py
```

### Entraînement (Colab, GPU T4)

Ouvrir `colab_iseg.ipynb`. Le notebook n'orchestre que des appels au paquet `iseg/`,
qui reste versionné dans git.

```bash
python -m iseg.train --variant standard --modalities t1t2 --context 5 --epochs 60
python -m iseg.train --variant separable --epochs 60
python -m iseg.train --variant tiny --epochs 60

# ablations
python -m iseg.train --variant standard --modalities t1   # le T2 sert-il ?
python -m iseg.train --variant standard --context 1        # 2D pur
python -m iseg.train --variant standard --context 7        # contexte élargi
```

### Export et mesure de frugalité

```bash
python -m iseg.export --checkpoint runs/standard_fold0.pt --cache cache \
    --calib-subjects 1 2 --runs 50 --threads 1
```

Produit le `.onnx` float32, sa version quantifiée int8 (statique, calibrée sur de
vraies coupes) et un rapport JSON : paramètres, MACs, taille en Mo, latence par coupe
et extrapolation au volume entier.

La latence est mesurée **CPU mono-thread**, proxy honnête pour un cœur ARM. Sur un
modèle très petit, la quantification int8 peut se révéler *plus lourde et plus lente*
que le float32 : les nœuds de quantification et déquantification ajoutés dominent
alors le calcul utile. C'est un résultat à reporter tel quel, pas à masquer.

### Mesure sur téléphone réel (optionnel)

Aucune application n'est nécessaire pour obtenir le chiffre qui compte :

```bash
adb push onnxruntime_perf_test /data/local/tmp/
adb push export/standard_fold0.int8.onnx /data/local/tmp/
adb shell "cd /data/local/tmp && ./onnxruntime_perf_test -e cpu -r 50 standard_fold0.int8.onnx"
```

## Structure

```
explore.py              exploration et figures
iseg/data.py            prétraitement, cache, dataset 2.5D
iseg/augment.py         augmentation (géométrie + intensité, dont champ de biais)
iseg/model.py           U-Net 2.5D et variantes frugales
iseg/losses.py          Dice + entropie croisée, Dice volumique
iseg/train.py           validation croisée 5 blocs
iseg/export.py          ONNX, quantification int8, MACs, latence
colab_iseg.ipynb        orchestration Colab
```
