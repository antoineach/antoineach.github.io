---
layout: post
title: "Point Transformer v2: Architecture and Implementation Details"
date: 2025-10-26
description: Detailed analysis of the Point Transformer v2 architecture for point-cloud segmentation and classification
tags: deep-learning point-cloud transformer architecture
categories: computer-vision
---
# Point Transformer v2: Architecture et Améliorations

## Introduction

**Point Transformer v2** améliore significativement son prédécesseur en termes d'efficacité computationnelle et de performances. Les innovations clés incluent :

- **Grid Pooling** au lieu de Furthest Point Sampling (3-5× plus rapide)
- **Map Unpooling** qui réutilise l'information du downsampling
- **GroupedLinear** pour réduire drastiquement le nombre de paramètres
- **Attention vectorielle enrichie** avec encodage de position sur les values
- **Masking des voisins invalides** pour gérer les nuages de tailles variables

Avant de plonger dans l'architecture globale, commençons par comprendre deux innovations fondamentales : GroupedLinear et GroupedVectorAttention.

---

## Architecture Globale

{% include figure.liquid path="assets/img/poinTransformerV2/architecture.svg" class="img-fluid rounded z-depth-1" %}

PTv2 suit une architecture U-Net avec :

**Encodeur (Downsampling):**
```
Input (N points, in_channels)
    ↓ GVAPatchEmbed
N points, 48 channels
    ↓ Encoder 1 (GridPool)
N1 points, 96 channels
    ↓ Encoder 2 (GridPool)
N2 points, 192 channels
    ↓ Encoder 3 (GridPool)
N3 points, 384 channels
    ↓ Encoder 4 (GridPool)
N4 points, 512 channels [BOTTLENECK]
```

**Décodeur (Upsampling):**
```
N4 points, 512 channels
    ↓ Decoder 4 (Unpool + skip)
N3 points, 384 channels
    ↓ Decoder 3 (Unpool + skip)
N2 points, 192 channels
    ↓ Decoder 2 (Unpool + skip)
N1 points, 96 channels
    ↓ Decoder 1 (Unpool + skip)
N points, 48 channels
    ↓ Segmentation Head
N points, num_classes
```

**Points clés:**
- Chaque **Encoder** réduit le nombre de points via **GridPool** (voxelisation)
- Chaque **Decoder** remonte en résolution via **Map Unpooling** + skip connection
- Les **clusters** stockent le mapping de voxelisation pour l'unpooling
- **Pas de Furthest Point Sampling** → beaucoup plus rapide !

---

## GroupedLinear : Réduction Paramétrique Intelligente

### Le problème avec Linear classique

Dans un réseau profond, générer des poids d'attention via des couches Linear standard accumule rapidement des paramètres :

```python
# Linear classique pour générer 8 poids d'attention depuis 64 features
Linear(in_features=64, out_features=8)
# Paramètres: 64 × 8 = 512 poids + 8 biais = 520 paramètres
```

### L'innovation GroupedLinear

{% include figure.liquid path="assets/img/poinTransformerV2/groupedLinear.svg" class="img-fluid rounded z-depth-1" %}

**GroupedLinear** remplace la matrice de poids par un **vecteur de poids partagé** :

```python
# GroupedLinear
weight: (1, 64)  # UN SEUL vecteur au lieu d'une matrice
# Paramètres: 64 (pas de biais)
```

### Fonctionnement étape par étape

```python
def forward(self, input):
    # input: (N, in_features) = (N, 64)
    # weight: (1, in_features) = (1, 64)
    
    # Étape 1: Multiplication élément par élément
    temp = input * weight  # (N, in_features)
    
    # Étape 2: Reshape en groupes
    temp = temp.reshape(N, groups, in_features/groups)
    # temp: (N, groups, in_features/groups)
    
    # Étape 3: Somme par groupe
    output = temp.sum(dim=-1)  # (N, groups) = (N, out_features)
    
    return output
```

### Exemple numérique concret

Prenons **N=1, in_features=8, groups=out_features=4** pour simplifier :

```python
# Input
x = [2, 3, 1, 4, 5, 2, 3, 1]  # (8,)

# Weight (vecteur partagé)
w = [0.5, 1.0, 0.2, 0.8, 0.3, 0.9, 0.4, 0.7]  # (8,)

# Étape 1: Multiplication élément par élément
temp = [2×0.5, 3×1.0, 1×0.2, 4×0.8, 5×0.3, 2×0.9, 3×0.4, 1×0.7]
     = [1.0, 3.0, 0.2, 3.2, 1.5, 1.8, 1.2, 0.7]

# Étape 2: Reshape en 4 groupes de 2 dimensions
temp_grouped = [
    [1.0, 3.0],     # Groupe 0
    [0.2, 3.2],     # Groupe 1
    [1.5, 1.8],     # Groupe 2
    [1.2, 0.7]      # Groupe 3
]

# Étape 3: Somme par groupe
output = [
    1.0 + 3.0 = 4.0,    # Groupe 0
    0.2 + 3.2 = 3.4,    # Groupe 1
    1.5 + 1.8 = 3.3,    # Groupe 2
    1.2 + 0.7 = 1.9     # Groupe 3
]
# Résultat: [4.0, 3.4, 3.3, 1.9]
```

### Comparaison des paramètres

| Configuration | Linear classique | GroupedLinear | Réduction |
|---------------|------------------|---------------|-----------|
| 64 → 8 | 64×8 = **512** | **64** | 8× |
| 128 → 16 | 128×16 = **2048** | **128** | 16× |
| 256 → 32 | 256×32 = **8192** | **256** | 32× |

GroupedLinear force le modèle à utiliser les mêmes poids pour tous les groupes, mais appliqués sur des portions différentes de l'input. 
---

## GroupedVectorAttention : Attention Locale Enrichie

### Vue d'ensemble

`GroupedVectorAttention` est le cœur de PTv2, avec plusieurs améliorations par rapport à PTv1.

{% include figure.liquid path="assets/img/poinTransformerV2/groupedVectorAttention.svg" class="img-fluid rounded z-depth-1" %}

### Comparaison détaillée avec PTv1

| Aspect | PTv1 (PointTransformerLayer) | PTv2 (GroupedVectorAttention) |
|--------|------------------------------|-------------------------------|
| **Projections Q, K, V** | Simple Linear | Linear + **BatchNorm1d + ReLU** |
| **Position Encoding** | Additif uniquement | Additif (+ option multiplicatif) |
| **Position Encoding sur values** | ❌ Non | ✅ **Oui** |
| **Masking voisins invalides** | ❌ Non (assume tous valides) | ✅ **Oui** |
| **Weight generation** | MLP standard (C×C/G params) | **GroupedLinear** (C params seulement) |
| **Normalisation** | Après weight encoding | **Avant et après** attention |

### Innovation 1 : Normalisation des Projections Q, K, V

**PTv1 :**
```python
# Projections simples sans normalisation
self.linear_q = nn.Linear(in_planes, mid_planes)
self.linear_k = nn.Linear(in_planes, mid_planes)
self.linear_v = nn.Linear(in_planes, out_planes)

# Usage
query = self.linear_q(feat)  # (N, C)
key = self.linear_k(feat)    # (N, C)
value = self.linear_v(feat)  # (N, C)
```

**PTv2 :**
```python
# Projections avec normalisation et activation
self.linear_q = nn.Sequential(
    nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
    PointBatchNorm(embed_channels),  # Normalisation !
    nn.ReLU(inplace=True)            # Activation !
)
# Idem pour linear_k

# Usage
query = self.linear_q(feat)  # (N, C) - normalisé et activé
```

**Pourquoi c'est important ?**

La normalisation des Q, K stabilise l'entraînement en évitant des valeurs extrêmes dans la relation Q-K :

**Impact :** Convergence plus rapide et training plus stable.

---

### Innovation 2 : Position Encoding sur les Values

**PTv1 :** L'encodage de position n'est ajouté qu'à la relation Q-K

```python
# Code PTv1 (simplifié)
relative_positions = neighbor_positions - query_position  # (N, K, 3)
encoded_positions = MLP(relative_positions)               # (N, K, out_dim)

# Application UNIQUEMENT sur relation Q-K
relation_qk = (key - query) + encoded_positions
# Les values ne sont PAS modifiées par la géométrie

```

**PTv2 :** L'encodage est ajouté à la relation Q-K **ET aux values**

```python
# Code PTv2
pe_bias = MLP(relative_positions)  # (N, K, C)

# Sur la relation Q-K (comme PTv1)
relation_qk = (key - query) + pe_bias

# NOUVEAU: aussi sur les values !
value = value + pe_bias

# (values contiennent maintenant l'info géométrique)
```


### Innovation 3 : Masking des Voisins Invalides

**Contexte : Différence Fondamentale entre PTv1 et PTv2**

#### PTv1 : K-NN garantit toujours K voisins

Dans PTv1, les voisins sont trouvés via **K-Nearest Neighbors (K-NN)** :

```python
# PTv1 - Dans chaque PointTransformerLayer
x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)
# Retourne TOUJOURS exactement K voisins (via K-NN search)
```

**PTv1 n'a donc pas besoin de masking** : tous les K voisins sont valides (même si certains peuvent être très éloignés).

---

#### PTv2 : Grid Pooling peut avoir < K voisins

Dans PTv2, les voisins sont déterminés par **Grid Pooling** (voxelisation), qui peut créer des régions avec moins de K points.

**Rappel : Qu'est-ce que le Grid Pooling ?**

Le **Grid Pooling** partitionne l'espace en **voxels** (cubes 3D de taille `grid_size`) et agrège tous les points d'un même voxel :

```
Avant Grid Pooling (N=16 points):

        grid_size = 0.5m
      
            ┌─────┬─────┬─────┬─────┐
            │ ●●  │     │  ●  │     │
            │ ●   │     │     │     │
            ├─────┼─────┼─────┼─────┤
            │  ●  │ ●●  │     │     │
            │     │  ●  │     │     │
            ├─────┼─────┼─────┼─────┤
            │     │  ●  │ ●   │  ●  │
            │  ●  │     │   ● │     │
            ├─────┼─────┼─────┼─────┤
            │  ●  │     │     │     │
            │     │  ●  │  ●  │     │
            └─────┴─────┴─────┴─────┘

Après Grid Pooling (M=10 voxels):
    
    ┌─────┬─────┬─────┬─────┐
    │  ◉ │     │  ◉  │     │  ← 1 point par voxel occupé
    │     │     │     │     │
    ├─────┼─────┼─────┼─────┤
    │  ◉  │  ◉  │     │     │
    │     │     │     │     │
    ├─────┼─────┼─────┼─────┤
    │  ◉  │  ◉  │  ◉  │  ◉  │
    │     │     │     │     │
    ├─────┼─────┼─────┼─────┤
    │  ◉  │  ◉  │     │     │
    │     │     │     │     │
    └─────┴─────┴─────┴─────┘
```

**Conséquence :** Après Grid Pooling, certaines zones peuvent être **peu denses** :

```
Configuration: K=8 voisins demandés

Zone dense:                   Zone peu dense (bord du nuage):
    ◉  ◉  ◉                       ◉
    ◉  ●  ◉                          
    ◉  ◉  ◉                               ◉
    
Point ● a 8 voisins ✓         Point ◉ a seulement 2 voisins ✗
```

**Comment PTv2 gère-t-il le manque de voisins ?**

Lors du K-NN sur les voxels poolés, si un voxel a moins de K voisins disponibles, les indices manquants sont marqués par **-1** :

```python
# PTv2 - K-NN sur les voxels après Grid Pooling
reference_index = knn_query(K=8, coord_pooled, offset)
# reference_index: (M, K)

# Exemple pour un voxel isolé
reference_index[voxel_42] = [15, 23, -1, -1, -1, -1, -1, -1]
#                            ↑───↑   ↑──────────────────────↑
#                            2 voisins    6 indices invalides (-1)
```

**Pourquoi -1 et pas juste moins d'indices ?**

Pour garder une **shape uniforme** `(M, K)` compatible avec les opérations matricielles :
- Tous les tensors ont la même forme
- Permet le batching efficace sur GPU
- Le padding avec -1 permet le masking explicite

---

**Solution PTv2 : Masking Explicite**

**Étape 1 : Création du masque**

```python
# reference_index contient -1 pour les voisins invalides
mask = torch.sign(reference_index + 1)  # (M, K)

# Comportement de sign(x+1):
# Si reference_index[i] = -1  → sign(-1+1) = sign(0) = 0  ← invalide
# Si reference_index[i] ≥ 0   → sign(≥1) = 1             ← valide
```


**Étape 2 : Application sur les poids d'attention**

```python
# Dans GroupedVectorAttention, après softmax
attention_weights = softmax(attention_scores)  # (M, K, groups)

# Application du masque
attention_weights = attention_weights * mask.unsqueeze(-1)
# Shape: (M, K, groups) × (M, K, 1) → (M, K, groups)
```

**Visualisation :**

```python
# Avant masking (après softmax sur K voisins)
attention_weights[voxel_42] = [
    [0.20, 0.15, 0.10, ...],  # Voisin 15 (valide)
    [0.18, 0.12, 0.09, ...],  # Voisin 23 (valide)
    [0.12, 0.14, 0.11, ...],  # Padding -1 (invalide mais a des poids !)
    [0.11, 0.13, 0.10, ...],  # Padding -1 (invalide)
    [0.10, 0.12, 0.12, ...],  # Padding -1 (invalide)
    [0.09, 0.11, 0.13, ...],  # Padding -1 (invalide)
    [0.10, 0.12, 0.18, ...],  # Padding -1 (invalide)
    [0.10, 0.11, 0.17, ...],  # Padding -1 (invalide)
]

# Après masking
mask = [1, 1, 0, 0, 0, 0, 0, 0]

attention_weights[voxel_42] = [
    [0.20, 0.15, 0.10, ...],  # Voisin 15 ✓
    [0.18, 0.12, 0.09, ...],  # Voisin 23 ✓
    [0, 0, 0, ...],            # Annulé ✓
    [0, 0, 0, ...],            # Annulé ✓
    [0, 0, 0, ...],            # Annulé ✓
    [0, 0, 0, ...],            # Annulé ✓
    [0, 0, 0, ...],            # Annulé ✓
    [0, 0, 0, ...],            # Annulé ✓
]
```

**Étape 3 : Agrégation**

```python
# Agrégation finale (somme pondérée)
output = (value_grouped * attention_weights.unsqueeze(-1)).sum(dim=1)
# Les voisins invalides (poids=0) ne contribuent pas ✓
```

---

**Pourquoi c'est Crucial ?**

**Sans masking**, les voisins padding contribueraient avec des **features aléatoires** :

---

### Innovation 4 : GroupedLinear pour les Poids d'Attention

Au lieu d'un MLP standard `Linear(C, groups)` avec C×groups paramètres, PTv2 utilise `GroupedLinear(C, groups)` avec seulement C paramètres.

```python
# PTv1: MLP standard
self.linear_w = nn.Sequential(
    nn.Linear(mid_planes, mid_planes // share_planes),  # C × C/G paramètres
    ...
)

# PTv2: avec GroupedLinear
self.weight_encoding = nn.Sequential(
    GroupedLinear(embed_channels, groups, groups),  # Seulement C paramètres !
    ...
)
```

**Gain :**  moins de paramètres pour générer les poids d'attention, sans perte de performance.

### Innovation 5 : Architecture de Normalisation

**PTv1 :** Normalisation minimale

```python
# PTv1 - Pas de normalisation sur les projections Q, K, V
query = Linear(x)  # Pas normalisé
key = Linear(x)
value = Linear(x)

# Normalisation seulement dans le MLP des poids
attention_scores = MLP_with_BatchNorm(relation_qk)
```

**PTv2 :** Normalisation extensive

```python
# PTv2 - Normalisation partout
query = Linear(x) → BatchNorm → ReLU  # Normalisé
key = Linear(x) → BatchNorm → ReLU
value = Linear(x)  # Pas d'activation (reste linéaire)

# Position encoding aussi normalisé
pe_bias = Linear(pos) → BatchNorm → ReLU → Linear

# Weight encoding aussi normalisé
attention_scores = GroupedLinear → BatchNorm → ReLU → Linear
```

**Impact :** Training plus stable, convergence plus rapide, moins sensible aux hyperparamètres.

---

# Block et BlockSequence : Architecture Résiduelle

## Block : Residual Block avec DropPath

Le `Block` de PTv2 encapsule `GroupedVectorAttention` dans une structure résiduelle similaire à ResNet, avec une innovation clé : **DropPath**.

{% include figure.liquid path="assets/img/poinTransformerV2/block.svg" class="img-fluid rounded z-depth-1" %}

### Comparaison avec PTv1

| Aspect | PTv1 (PointTransformerBlock) | PTv2 (Block) |
|--------|------------------------------|--------------|
| **Structure** | Linear → Attention → Linear + Skip | Linear → Attention → Linear + Skip |
| **Régularisation** | Dropout uniquement | **DropPath** + Dropout |
| **Normalisation** | 3× BatchNorm | 3× BatchNorm (identique) |
| **Skip connection** | Simple addition | Addition avec **DropPath** |

### Architecture Détaillée

```
Input features (N, C)
    ↓
[Linear + BatchNorm1d + ReLU]  ← Pre-activation (expansion)
    ↓
[GroupedVectorAttention]  ← Attention locale sur K voisins
    ↓
[BatchNorm1d + ReLU]  ← Post-attention normalization
    ↓
[Linear + BatchNorm1d]  ← Projection
    ↓
[DropPath]  ← Régularisation stochastique (NOUVEAU)
    ↓
[+ Skip Connection]  ← Connexion résiduelle
    ↓
[ReLU]  ← Activation finale
    ↓
Output features (N, C)
```

### DropPath : Stochastic Depth

**DropPath** (Stochastic Depth) est une technique de régularisation qui **dropout des chemins entiers** dans un réseau résiduel, plutôt que des neurones individuels.

**Dropout classique vs DropPath :**

```python
# Dropout classique (agit sur les features)
def dropout(x, p=0.5):
    mask = random(x.shape) > p  # Masque aléatoire par élément
    return x * mask / (1 - p)

output = x + dropout(f(x))
# Certaines features de f(x) sont mises à 0


# DropPath (agit sur le chemin entier)
def drop_path(x, p=0.1):
    if training and random() < p:
        return 0  # Tout le chemin est ignoré !
    return x

output = x + drop_path(f(x))
# Soit tout f(x) est gardé, soit tout est ignoré
```

**Fonctionnement en pratique :**

Durant l'entraînement, avec probabilité `drop_path_rate` (typiquement 0.1), on saute complètement le bloc transformé :

```python
# Sans DropPath (PTv1)
feat_transformed = Linear → Attention → Linear
output = identity + feat_transformed  # Toujours calculé

# Avec DropPath (PTv2)
feat_transformed = Linear → Attention → Linear

if training and random() < drop_path_rate:
    output = identity  # On saute feat_transformed complètement !
else:
    output = identity + feat_transformed

# À l'inférence
output = identity + feat_transformed  # Toujours actif
```

**Visualisation sur un réseau de 12 blocs :**

```
Avec drop_path_rate = 0.1 (10% de chance de drop par bloc)

Training iteration 1:
Input → [Block1] → [Block2] → [SKIP] → [Block4] → ... → [SKIP] → [Block12]
        ✓          ✓          ✗          ✓              ✗          ✓
        (réseau de ~10 blocs actifs)

Training iteration 2:
Input → [Block1] → [SKIP] → [Block3] → [Block4] → ... → [Block11] → [Block12]
        ✓          ✗        ✓          ✓                  ✓          ✓
        (réseau de ~11 blocs actifs)

Inference:
Input → [Block1] → [Block2] → [Block3] → [Block4] → ... → [Block11] → [Block12]
        ✓          ✓          ✓          ✓                  ✓          ✓
        (tous les 12 blocs actifs)
```


**Cependant, dans PTv2, le `drop_path_rate` est implémenté mais laissé à 0.0. Autrement dit, il n'est pas utilisé.**



## BlockSequence : Réutilisation du K-NN

`BlockSequence` empile plusieurs `Block` et introduit une optimisation majeure : **partage du reference_index**.

{% include figure.liquid path="assets/img/poinTransformerV2/blockSequence.svg" class="img-fluid rounded z-depth-1" %}

### Innovation Clé : K-NN Calculé Une Seule Fois

**Problème PTv1 :**

Dans PTv1, chaque `PointTransformerLayer` recalcule les K plus proches voisins via K-NN :

```python
# PTv1 - Dans PointTransformerLayer.forward()
def forward(self, pxo):
    p, x, o = pxo
    
    # K-NN calculé À CHAQUE COUCHE
    x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)
    x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)
    # ...
```

Pour un bloc avec 6 couches `PointTransformerLayer`, on fait **6 fois** la même recherche K-NN !

```
Bloc avec 6 couches PTv1:
Layer 1: K-NN (N points, find K=16 neighbors) → O(N log N)
Layer 2: K-NN (N points, find K=16 neighbors) → O(N log N)
Layer 3: K-NN (N points, find K=16 neighbors) → O(N log N)
Layer 4: K-NN (N points, find K=16 neighbors) → O(N log N)
Layer 5: K-NN (N points, find K=16 neighbors) → O(N log N)
Layer 6: K-NN (N points, find K=16 neighbors) → O(N log N)

Coût total: 6 × O(N log N)
```

**Solution PTv2 :**

Dans PTv2, `BlockSequence` calcule le K-NN **une seule fois** au début et tous les `Block` partagent le même `reference_index` :

```python
# PTv2 - Dans BlockSequence.forward()
def forward(self, points):
    coord, feat, offset = points
    
    # K-NN calculé UNE SEULE FOIS au début
    reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
    # reference_index: (N, K) - indices des K voisins pour chaque point
    
    # Tous les blocks partagent le même reference_index
    for block in self.blocks:
        points = block(points, reference_index)  # Pas de recalcul !
    
    return points
```

```
Bloc avec 6 couches PTv2:
K-NN (une fois): O(N log N)
Layer 1: Utilise reference_index → O(1) lookup
Layer 2: Utilise reference_index → O(1) lookup
Layer 3: Utilise reference_index → O(1) lookup
Layer 4: Utilise reference_index → O(1) lookup
Layer 5: Utilise reference_index → O(1) lookup
Layer 6: Utilise reference_index → O(1) lookup

Coût total: O(N log N)  ← 6× plus rapide !
```

### Pourquoi c'est Valide ?

**Question :** Peut-on vraiment réutiliser les mêmes voisins à travers toutes les couches ?

**Réponse :** **OUI**, car dans `BlockSequence`, les **positions ne changent pas** !

```python
# Dans Block.forward()
def forward(self, points, reference_index):
    coord, feat, offset = points
    
    # coord (positions) reste INCHANGÉ à travers le bloc
    feat = self.fc1(feat)  # Seulement les features changent
    feat = self.attn(feat, coord, reference_index)  # coord fixe
    feat = self.fc3(feat)
    # ...
    
    return [coord, feat, offset]  # coord identique en sortie
```

Les positions 3D (`coord`) sont **constantes** dans un `BlockSequence` - seules les **features** évoluent. Les K plus proches voisins restent donc identiques géométriquement !

**Cas où on DOIT recalculer le K-NN :**

Les positions changent uniquement lors des transitions entre niveaux de l'architecture (downsampling/upsampling) :

```python
# Encoder
points = BlockSequence(points)  # Positions fixes, K-NN partagé ✓
points = GridPool(points)        # Positions changent (downsampling) ✗
points = BlockSequence(points)  # Nouvelles positions → nouveau K-NN ✓

# Decoder
points = UnpoolWithSkip(points, skip)  # Positions changent (upsampling) ✗
points = BlockSequence(points)         # Nouvelles positions → nouveau K-NN ✓
```

---

## GVAPatchEmbed : Embedding Initial

Avant de downsampler, PTv2 applique un `GVAPatchEmbed` qui enrichit les features à pleine résolution.

{% include figure.liquid path="assets/img/poinTransformerV2/GVAPatchEmbed.svg" class="img-fluid rounded z-depth-1" %}

### Rôle

**GVAPatchEmbed** = Projection linéaire + BlockSequence (sans downsampling)

```python
Input: (N, in_channels)
    ↓
Linear + BatchNorm1d + ReLU
    ↓
(N, embed_channels)
    ↓
BlockSequence (depth blocks)
    ↓
Output: (N, embed_channels)
```

# GridPool : Downsampling par Voxelisation

## Vue d'ensemble

`GridPool` est l'une des innovations majeures de PTv2, remplaçant le **Furthest Point Sampling (FPS)** de PTv1 par une approche basée sur la **voxelisation**.

{% include figure.liquid path="assets/img/poinTransformerV2/gridPool.svg" class="img-fluid rounded z-depth-1" %}



####  Voxelisation

```python
# Normalisation des coordonnées par rapport au début de chaque nuage
coord_normalized = coord - start[batch]  # (N, 3)

# Assignation à une grille avec voxels de taille grid_size
cluster = voxel_grid(
    pos=coord_normalized, 
    size=grid_size,  # ex: 0.06m
    batch=batch,
    start=0
)
# cluster: (N,) - ID du voxel pour chaque point
```

**Exemple avec grid_size=1.0 :**

```python
# Points d'un nuage (après normalisation)
points = [
    [0.2, 0.3, 0.1],  # Voxel (0, 0, 0)
    [0.8, 0.5, 0.2],  # Voxel (0, 0, 0)
    [1.2, 0.4, 0.3],  # Voxel (1, 0, 0)
    [1.5, 1.8, 0.1],  # Voxel (1, 1, 0)
    [0.1, 1.1, 0.9],  # Voxel (0, 1, 0)
]

# Calcul du voxel ID
voxel_id = floor(coord / grid_size)

cluster = [
    0,  # (0,0,0) → ID unique du voxel
    0,  # (0,0,0) → même voxel
    1,  # (1,0,0)
    2,  # (1,1,0)
    3,  # (0,1,0)
]
```

#### Étape 4 : Identification des Voxels Uniques

```python
unique, cluster_inverse, counts = torch.unique(
    cluster, 
    sorted=True, 
    return_inverse=True, 
    return_counts=True
)
```

**Que retourne torch.unique ?**

```python
# Input cluster (exemple)
cluster = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3]
#          ↑─────↑  ↑──↑  ↑────────↑  ↑──────↑
#          3 pts   2 pts  4 points   3 points

unique = [0, 1, 2, 3]  # Les voxels uniques
# Nvoxel = 4

cluster_inverse = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3]
# Mapping: point i appartient au voxel unique[cluster_inverse[i]]

counts = [3, 2, 4, 3]  # Nombre de points par voxel
```

#### Étape 5 : Tri et Index Pointers

```python
# Tri les points par voxel
_, sorted_indices = torch.sort(cluster_inverse)
# sorted_indices: ordre pour regrouper les points du même voxel ensemble

# Création des pointeurs pour chaque voxel
idx_ptr = torch.cat([
    torch.zeros(1), 
    torch.cumsum(counts, dim=0)
])
# idx_ptr: (Nvoxel + 1,)
```

**Exemple :**

```python
# Après tri
sorted_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# Points triés par voxel

# Index pointers
counts = [3, 2, 4, 3]
idx_ptr = [0, 3, 5, 9, 12]
#          ↑  ↑  ↑  ↑  ↑
#          │  │  │  │  └─ Fin (12 points)
#          │  │  │  └──── Voxel 3 commence à l'indice 9
#          │  │  └─────── Voxel 2 commence à l'indice 5
#          │  └────────── Voxel 1 commence à l'indice 3
#          └───────────── Voxel 0 commence à l'indice 0
```

#### Étape 6 : Agrégation des Coordonnées (Moyenne)

```python
coord_pooled = segment_csr(
    coord[sorted_indices],  # Coordonnées triées par voxel
    idx_ptr, 
    reduce="mean"
)
# coord_pooled: (Nvoxel, 3)
# Position moyenne de tous les points dans chaque voxel
```

**Exemple :**

```python
# Voxel 0 contient 3 points aux positions:
points_voxel_0 = [[0.2, 0.3, 0.1], [0.8, 0.5, 0.2], [0.1, 0.2, 0.15]]
coord_pooled[0] = mean(points_voxel_0) = [0.37, 0.33, 0.15]

# Voxel 1 contient 2 points:
points_voxel_1 = [[1.2, 0.4, 0.3], [1.4, 0.6, 0.4]]
coord_pooled[1] = mean(points_voxel_1) = [1.3, 0.5, 0.35]
```

#### Étape 7 : Agrégation des Features (Max)

```python
feat_pooled = segment_csr(
    feat[sorted_indices],  # Features triées par voxel
    idx_ptr,
    reduce="max"
)
# feat_pooled: (Nvoxel, out_channels)
# Maximum des features dans chaque voxel
```

**Pourquoi Max au lieu de Mean ?**

```python
# Exemple avec 3 points dans un voxel

# Mean pooling
feat_mean = (feat1 + feat2 + feat3) / 3
# Peut "diluer" les features importantes

# Max pooling (utilisé par PTv2)
feat_max = max(feat1, feat2, feat3)
# Préserve les features dominantes de chaque canal
# Plus robuste au bruit et aux outliers
```

#### Étape 8 : Reconstruction des Offsets

```python
# Récupération du batch ID pour chaque voxel
# (prend le batch du premier point de chaque voxel)
batch_pooled = batch[idx_ptr[:-1]]
# batch_pooled: (Nvoxel,)

# Conversion batch → offset
offset_pooled = batch2offset(batch_pooled)
# offset_pooled: (B,)
```

#### Étape 9 : Retour du Cluster Mapping

```python
return [coord_pooled, feat_pooled, offset_pooled], cluster_inverse
```

Le `cluster_inverse` est **crucial** car il permet le **Map Unpooling** plus tard :

```python
# cluster_inverse: (N,) - pour chaque point, son voxel d'appartenance
cluster_inverse[point_i] = voxel_id

# Exemple
cluster_inverse = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3]
#                  ↑─────↑  ↑──↑  ↑────────↑  ↑──────↑
#                  Points du voxel 0, 1, 2, 3
```

Ce mapping sera réutilisé dans `UnpoolWithSkip` pour "dépooler" efficacement !

---


### 4. Map Unpooling Gratuit

Le `cluster_inverse` permet un unpooling **sans calcul** :

```python
# PTv1: doit recalculer K-NN pour l'interpolation
upsampled = knn_interpolation(low_res, high_res)  # Coûteux !

# PTv2: réutilise le cluster mapping
upsampled = feat_low_res[cluster_inverse]  # Lookup instantané !
```

# UnpoolWithSkip : Map Unpooling avec Skip Connections

## Vue d'ensemble

`UnpoolWithSkip` est le pendant de `GridPool` dans le décodeur, permettant de remonter en résolution tout en fusionnant l'information multi-échelle via les skip connections.

{% include figure.liquid path="assets/img/poinTransformerV2/unpoolWithSkip.svg" class="img-fluid rounded z-depth-1" %}


## Problème avec K-NN Interpolation (PTv1)

### Algorithme d'Interpolation PTv1

Dans PTv1, pour passer de M points (basse résolution) à N points (haute résolution), on utilise une **interpolation par K-NN** :

### Problèmes de l'Interpolation

**1. Coût computationnel :**

```python
# Pour chaque point haute résolution N:
#   - Calculer M distances
#   - Trier pour trouver les K plus proches
#   - Calculer la moyenne pondérée

Complexité: O(N × M log M)

# Exemple: M=25k, N=100k
Opérations: 100k × 25k × log(25k) ≈ 35 milliards !
```


## Solution : Map Unpooling (PTv2)

### Principe : Réutilisation du Cluster Mapping

L'idée géniale de PTv2 : **stocker le mapping lors du downsampling** et le **réutiliser lors de l'upsampling** !

**Rappel du GridPool :**

```python
# GridPool retourne le cluster_inverse
coord_pooled, feat_pooled, offset_pooled, cluster = GridPool(points)

# cluster: (N,) - pour chaque point original, son voxel d'appartenance
cluster = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3]
#          └──┬──┘  └─┬─┘  └────┬────┘  └──┬──┘
#          Voxel 0  Voxel 1  Voxel 2   Voxel 3
```

**Map Unpooling :**

```python
# Pour remonter en résolution, simple indexing !
feat_upsampled = feat_pooled[cluster]  # (N, C)

# Chaque point récupère les features de son voxel d'origine
```

**C'est tout !** Un simple lookup, complexité **O(1)** par point, donc **O(N)** total.

---

## Algorithme Détaillé

### Inputs

```python
# Points actuels (basse résolution)
coord_low: (M, 3)        # Positions des voxels
feat_low: (M, in_ch)     # Features des voxels
offset_low: (B,)

# Points skip (haute résolution - de l'encodeur)
coord_skip: (N, 3)       # Positions originales
feat_skip: (N, skip_ch)  # Features originales
offset_skip: (B,)

# Cluster mapping (du GridPool correspondant)
cluster: (N,)            # Pour chaque point, son voxel
```

### Étape 1 : Projection des Features Basse Résolution

```python
feat_low_proj = Linear(feat_low) → BatchNorm1d → ReLU
# feat_low_proj: (M, out_ch)
```

### Étape 2 : Map Unpooling

```python
# Lookup direct via cluster
feat_mapped = feat_low_proj[cluster]
# feat_mapped: (N, out_ch)
```

Chaque point récupère **exactement** les features de son voxel d'origine !

### Étape 3 : Projection des Features Skip

```python
feat_skip_proj = Linear(feat_skip) → BatchNorm1d → ReLU
# feat_skip_proj: (N, out_ch)
```

### Étape 4 : Fusion Skip Connection

```python
feat_fused = feat_mapped + feat_skip_proj
# feat_fused: (N, out_ch)
```

**Visualisation :**

```
Basse résolution (upsampled):        Skip (haute résolution):
    feat_mapped                           feat_skip_proj
         ↓                                      ↓
    [0.2, 0.5, 0.1, 0.8]              [0.3, 0.1, 0.6, 0.2]
         ↓                                      ↓
         └──────────────── + ────────────────┘
                             ↓
                    [0.5, 0.6, 0.7, 1.0]
                         feat_fused
```

### Étape 5 : Output

```python
return [coord_skip, feat_fused, offset_skip]
# On retourne les coordonnées skip (haute résolution)
# Avec les features fusionnées
```

---


## Encoder et Decoder : Vue Complète

### Encoder

{% include figure.liquid path="assets/img/poinTransformerV2/encoder.svg" class="img-fluid rounded z-depth-1" %}

```python
class Encoder:
    def forward(self, points):
        # Downsampling + enrichissement features
        points_pooled, cluster = GridPool(points)
        
        # Attention locale sur les voxels
        points_out = BlockSequence(points_pooled)
        
        return points_out, cluster
```


### Decoder

{% include figure.liquid path="assets/img/poinTransformerV2/decoder.svg" class="img-fluid rounded z-depth-1" %}

```python
class Decoder:
    def forward(self, points_low, points_skip, cluster):
        # Upsampling + fusion skip
        points_up = UnpoolWithSkip(points_low, points_skip, cluster)
        
        # Attention locale sur les points upsampled
        points_out = BlockSequence(points_up)
        
        return points_out
```

---


## Performance Globale : PTv1 vs PTv2


### Mémoire


{% include figure.liquid path="assets/img/poinTransformerV2/ptv2_time_diff.png" class="img-fluid rounded z-depth-1" %}

### Précision

{% include figure.liquid path="assets/img/poinTransformerV2/ptv2_s3dis_miou.png" class="img-fluid rounded z-depth-1" %}


---

