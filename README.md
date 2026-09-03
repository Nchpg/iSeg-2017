# iSeg-2017 — segmentation frugale du cerveau du nourrisson

Segmentation en trois tissus (LCR, substance grise, substance blanche) sur IRM T1 et
T2 de bébés de 6 mois, avec un objectif de déploiement mobile ou embarqué.

## Protocole d'évaluation

Les 13 sujets du dossier `iSeg-2017-Testing/` **n'ont pas d'étiquettes** : elles n'ont
jamais été publiées, elles servaient au classement du challenge. Toute évaluation
chiffrée se fait donc sur les 10 sujets d'entraînement : **8 pour apprendre, les
sujets 1 et 2 mis de côté pour mesurer**. Le dossier de test ne sert qu'à produire
des prédictions qualitatives et à mesurer un temps d'inférence.

Le découpage est fait **par sujet**, jamais par coupe : deux coupes voisines d'un
même volume sont quasi identiques, les mélanger entre entraînement et validation
donnerait un Dice artificiellement élevé.

Le Dice se calcule **par volume**, après réassemblage de toutes les coupes.

## Ce que les données imposent

Constats mesurés sur les 10 sujets (le script d'analyse qui les a produits est dans
l'historique git : `git show HEAD~1:explore.py`, figures dans `figures/`).

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
pas). Son apport réel n'a pas été mesuré par ablation — `--modalities t1` permet
de le faire si la question se pose.

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

| variante | description | paramètres | int8 | Dice |
|---|---|---|---|---|
| `standard` | convolutions 3×3 pleines, base 16, 4 niveaux | 1 943 636 | 1,89 Mo | 0,8927 |
| **`separable`** | depthwise 3×3 + pointwise 1×1 | 386 782 | **0,43 Mo** | **0,8624** |
| `tiny` | separable, base 8, 3 niveaux | 26 974 | 0,07 Mo | 0,8281 |

`separable` est le modèle retenu : **97 % du Dice de `standard` pour 20 % de sa
taille**. Descendre plus bas (`tiny`) coûte 3,4 points de Dice supplémentaires pour
un gain de taille sans effet pratique — les deux tiennent déjà largement sur un
téléphone.

Détail par tissu pour `separable` : LCR 0,8969, GM 0,8606, WM 0,8297 — la substance
blanche est bien la plus difficile, comme le chevauchement d'histogrammes de 0,702
le laissait prévoir.

Ces chiffres viennent d'une validation croisée 5 blocs faite une fois (écart entre
le meilleur et le pire bloc : 0,0085, donc la performance ne dépend pas du
découpage). Le code actuel se contente d'un découpage unique, suffisant pour
mesurer un modèle.

## Utilisation

### Environnement local

Sous NixOS, les wheels manylinux ont besoin de `libstdc++` et `zlib` dans le chemin
des bibliothèques ; `env.sh` s'en charge.

```bash
uv venv .venv
uv pip install --python .venv/bin/python numpy nibabel torch onnx onnxruntime onnxscript
source env.sh
```

### Entraînement (Colab, GPU T4)

Ouvrir `colab_iseg.ipynb`. Le notebook n'orchestre que des appels au paquet `iseg/`,
qui reste versionné dans git.

```bash
python -m iseg.train                       # separable, 60 epoques, ~10 min sur T4
python -m iseg.train --variant standard    # la reference haute
python -m iseg.train --variant tiny        # la plus petite
```

Le Dice sur les deux sujets de validation s'affiche toutes les 5 époques, et les
poids du meilleur passage sont écrits dans `runs/<variante>.pt`.

### Export du modèle

```bash
python -m iseg.export --checkpoint runs/separable.pt
```

Produit le `.onnx` float32 et sa version quantifiée int8, environ 3,5× plus petite
(1,49 Mo → 0,43 Mo pour `separable`). La quantification est statique, calibrée sur
de vraies coupes — sur un réseau convolutif, la quantification dynamique ne touche
pas les convolutions et n'apporte presque rien.

### Démonstration mobile

`webdemo/index.html` est une page autonome : elle lit les fichiers `.hdr/.img`,
laisse choisir une coupe et segmente sur l'appareil, en WebAssembly. Le modèle y est
embarqué en base64, ce qui évite le `fetch()` d'un fichier local que les navigateurs
bloquent en `file://`.

Pour y placer un autre modèle :

```bash
python -m iseg.export --checkpoint runs/separable.pt --embed webdemo/index.html
```

Il suffit ensuite d'ouvrir le `.html` — aucun serveur nécessaire. Une connexion reste
requise au premier chargement pour récupérer le moteur ONNX Runtime depuis son CDN.

Mesuré sur téléphone : **97 ms par coupe**, soit environ 13 s pour un volume complet.

## Structure

```
iseg/data.py            prétraitement, cache, dataset 2.5D
iseg/augment.py         augmentation (géométrie + intensité, dont champ de biais)
iseg/model.py           U-Net 2.5D et variantes frugales
iseg/losses.py          Dice + entropie croisée, Dice volumique
iseg/train.py           entraînement, 8 sujets / 2 en validation
iseg/export.py          ONNX, quantification int8, injection dans la page web
colab_iseg.ipynb        orchestration Colab
webdemo/index.html      démonstration mobile autonome
```
