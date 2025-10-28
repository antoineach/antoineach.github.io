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

Commençons par l'architecture globale avant de détailler chaque composant.

---

## Architecture Globale

{% include figure.liquid 
   path="assets/img/pointTransformerv2/main_architecture.svg" 
   class="img-fluid rounded z-depth-1" 
   style="height:400px; object-fit:contain;"
%}

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
    temp = input * weight  # (N, 64)
    
    # Étape 2: Reshape en groupes
    temp = temp.reshape(N, groups, in_features/groups)
    # temp: (N, 8, 8)
    
    # Étape 3: Somme par groupe
    output = temp.sum(dim=-1)  # (N, 8)
    
    return output
```

### Exemple numérique concret

Prenons **N=1, in_features=8, groups=4** pour simplifier :

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

### Pourquoi ça fonctionne ?

**Inductive bias structuré :** GroupedLinear force le modèle à utiliser les mêmes poids pour tous les groupes, mais appliqués sur des portions différentes de l'input. C'est comme dire :

*"Les 8 premiers channels utilisent les poids w₀-w₇ pour former le poids d'attention du groupe 0, les 8 suivants utilisent les poids w₈-w₁₅ pour le groupe 1, etc."*

Cette contrainte :
- ✅ Réduit le risque d'overfitting (moins de paramètres)
- ✅ Force des représentations plus générales
- ✅ Maintient les performances (validé empiriquement dans le papier)

---

## GroupedVectorAttention : Attention Locale Enrichie

### Vue d'ensemble

`GroupedVectorAttention` est le cœur de PTv2, avec plusieurs améliorations par rapport à PTv1.

{% include figure.liquid path="assets/img/poinTransformerV2/groupedVectorAttention.svg" class="img-fluid rounded z-depth-1" %}

**Différences clés avec PTv1:**

| Aspect | PTv1 | PTv2 |
|--------|------|------|
| **Position Encoding sur values** | ❌ Non | ✅ Oui |
| **Masking voisins invalides** | ❌ Non | ✅ Oui |
| **Weight generation** | MLP standard | **GroupedLinear** (8× moins de params) |
| **Normalization** | BatchNorm après Linear | **BatchNorm + ReLU entre** Q/K/V |

### Innovation 1 : Position Encoding sur les Values

**Dans PTv1**, l'encodage de position n'était ajouté qu'à la relation Q-K :

```python
# PTv1
relation_qk = (key - query) + position_encoding(pos)
# Les values ne sont PAS affectées par la géométrie
```

**Dans PTv2**, on ajoute aussi l'encodage aux **values** :

```python
# PTv2
pe_bias = MLP(relative_positions)  # (N, K, 3) → (N, K, C)

# Sur la relation Q-K (comme PTv1)
relation_qk = (key - query) + pe_bias

# NOUVEAU: aussi sur les values !
value = value + pe_bias
```

**Pourquoi c'est important ?**

L'encodage de position sur les values permet d'**injecter directement l'information géométrique** dans les features qui seront agrégées.

**Exemple physique :**

Imaginons un point représentant le coin d'une table, avec 3 voisins :

```
Voisin 1: dessus de la table (Δpos = [0, 0, 0.05m])
    → pe_bias₁ = [0.8, 0.1, 0.1, ...]  # proche, même surface
    
Voisin 2: pied de table (Δpos = [0, 0, -0.8m])
    → pe_bias₂ = [0.2, 0.1, 0.9, ...]  # loin, objet différent
    
Voisin 3: air vide (Δpos = [0.5, 0, 0])
    → pe_bias₃ = [0.1, 0.8, 0.1, ...]  # éloigné latéralement
```

Ces encodages, ajoutés aux values, permettent au modèle de savoir **où se trouvent géométriquement** les features qu'il agrège, pas seulement leur importance relative (via les poids d'attention).

### Innovation 2 : Masking des Voisins Invalides

**Problème :** Dans un batch, certains points (au bord du nuage, ou dans des régions peu denses) ont moins de K voisins. Le K-NN "pad" avec des indices `-1`.

**Solution PTv2 :**

```python
# reference_index: (N, K)
# Contient -1 pour les voisins invalides (padding)

# Création du masque
mask = torch.sign(reference_index + 1)  # (N, K)
# Si reference_index[n, k] = -1  → sign(-1+1) = sign(0) = 0
# Si reference_index[n, k] ≥ 0   → sign(≥1) = 1

# Application sur les poids d'attention
attention_weights = attention_weights * mask.unsqueeze(-1)
# Shape: (N, K, groups) × (N, K, 1) → (N, K, groups)
```

**Exemple concret :**

```python
# Point isolé au bord du nuage avec seulement 3 vrais voisins
reference_index[point_42] = [15, 23, 8, -1, -1, -1, -1, -1, ...]  # K=16
mask[point_42] = [1, 1, 1, 0, 0, 0, 0, 0, ...]

# Poids d'attention avant masking (après softmax)
attention[point_42] = [
    [0.3, 0.2, ...],  # Voisin 15 (valide)
    [0.25, 0.18, ...], # Voisin 23 (valide)
    [0.2, 0.15, ...],  # Voisin 8 (valide)
    [0.08, 0.12, ...], # Invalide mais a des poids !
    [0.05, 0.10, ...], # Invalide
    ...
]

# Après masking
attention[point_42] = [
    [0.3, 0.2, ...],   # OK
    [0.25, 0.18, ...], # OK
    [0.2, 0.15, ...],  # OK
    [0, 0, ...],       # Annulé ✓
    [0, 0, ...],       # Annulé ✓
    ...
]
```

Sans ce masking, les voisins "padding" contribueraient avec des **features aléatoires/garbage**, polluant l'agrégation finale !

### Innovation 3 : GroupedLinear pour les Poids d'Attention

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

**Gain :** 8× moins de paramètres pour générer les poids d'attention, sans perte de performance.

### Flux Complet avec Exemple Numérique

Prenons un exemple complet avec **N=1000 points, K=16 voisins, C=64 channels, groups=8**.

#### Étape 1 : Projections Q, K, V

```python
# Input
feat: (1000, 64)

# Projections avec normalisation
query = Linear(feat) → BatchNorm1d → ReLU  # (1000, 64)
key = Linear(feat) → BatchNorm1d → ReLU    # (1000, 64)
value = Linear(feat)                        # (1000, 64)
```

#### Étape 2 : Grouping des Voisins

```python
# Récupération des K voisins via reference_index
key_neighbors = grouping(reference_index, key, coord, with_xyz=True)
# Shape: (1000, 16, 3+64) = (1000, 16, 67)
# Les 3 premières dims sont les positions relatives

value_neighbors = grouping(reference_index, value, coord, with_xyz=False)
# Shape: (1000, 16, 64)
```

**Note :** `reference_index` (N, K) contient les indices des K voisins pour chaque point, pré-calculés dans `BlockSequence`.

#### Étape 3 : Séparation Positions / Features

```python
relative_positions = key_neighbors[:, :, 0:3]  # (1000, 16, 3)
key_neighbors = key_neighbors[:, :, 3:]        # (1000, 16, 64)
```

#### Étape 4 : Encodage des Positions

```python
# MLP sur positions relatives
pe_bias = MLP(relative_positions)  # (1000, 16, 3) → (1000, 16, 64)
# Le MLP transforme les positions 3D en features de dimension C
```

**Exemple pour un point :**
```python
# Positions relatives de ses 16 voisins
relative_positions[point_0] = [
    [0.05, 0.02, 0.01],   # Voisin très proche
    [0.15, -0.10, 0.05],  # Voisin moyen
    [0.50, 0.30, -0.20],  # Voisin lointain
    ...
]

# Encodage via MLP
pe_bias[point_0] = [
    [0.8, 0.2, 0.1, ...],   # Features pour voisin proche
    [0.5, 0.4, 0.3, ...],   # Features pour voisin moyen
    [0.2, 0.1, 0.9, ...],   # Features pour voisin lointain
    ...
]
```

#### Étape 5 : Application Position Encoding

```python
# Sur la relation Q-K
relation_qk = (key_neighbors - query.unsqueeze(1)) + pe_bias
# Shape: (1000, 16, 64)

# Sur les values (NOUVEAU dans PTv2 !)
value_with_pos = value_neighbors + pe_bias
# Shape: (1000, 16, 64)
```

#### Étape 6 : Génération des Poids d'Attention

```python
# MLP contenant GroupedLinear
attention_scores = weight_encoding(relation_qk)
# Shape: (1000, 16, 64) → (1000, 16, 8)

# Normalization: softmax sur les voisins
attention_weights = softmax(attention_scores, dim=1)
# Shape: (1000, 16, 8)
# Pour chaque point, les poids des 16 voisins somment à 1 (par groupe)
```

#### Étape 7 : Masking

```python
mask = sign(reference_index + 1)  # (1000, 16)
attention_weights = attention_weights * mask.unsqueeze(-1)
# Shape: (1000, 16, 8)
# Les poids des voisins invalides (-1) sont mis à 0
```

#### Étape 8 : Agrégation par Groupes

```python
# Reshape values en groupes
value_grouped = value_with_pos.view(1000, 16, 8, 8)
# Shape: (N, K, groups, C/groups)

# Préparation des poids pour broadcasting
attention_exp = attention_weights.unsqueeze(-1)
# Shape: (1000, 16, 8, 1)

# Multiplication groupe par groupe
weighted = value_grouped * attention_exp
# Shape: (1000, 16, 8, 8)

# Agrégation sur les K=16 voisins
aggregated = weighted.sum(dim=1)
# Shape: (1000, 8, 8)

# Flatten
output = aggregated.reshape(1000, 64)
# Shape: (1000, 64)
```

**Visualisation pour un point avec 3 voisins :**

```
Point central ◉ avec 3 voisins:

Voisin 1 ●₁: value = [v₁⁰, v₁¹, ..., v₁⁶³]
    → Découpe en 8 groupes de 8 dims
    → Poids: [w₁⁰=0.5, w₁¹=0.3, ..., w₁⁷=0.2]
    
Voisin 2 ●₂: value = [v₂⁰, v₂¹, ..., v₂⁶³]
    → Découpe en 8 groupes de 8 dims
    → Poids: [w₂⁰=0.3, w₂¹=0.4, ..., w₂⁷=0.5]
    
Voisin 3 ●₃: value = [v₃⁰, v₃¹, ..., v₃⁶³]
    → Découpe en 8 groupes de 8 dims
    → Poids: [w₃⁰=0.2, w₃¹=0.3, ..., w₃⁷=0.3]

Agrégation pour le groupe g=0 (dims 0-7):
output[groupe_0] = w₁⁰ × value₁[0:8] + w₂⁰ × value₂[0:8] + w₃⁰ × value₃[0:8]
                 = 0.5 × [v₁⁰,...,v₁⁷] + 0.3 × [v₂⁰,...,v₂⁷] + 0.2 × [v₃⁰,...,v₃⁷]

... répété pour les 8 groupes
```

---
Voici la section comparative enrichie pour GroupedVectorAttention :

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
| **PE sur values** | ❌ Non | ✅ **Oui** |
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

```python
# Sans normalisation (PTv1)
query = Linear(feat)  # Peut avoir de grandes variations
key = Linear(feat)    
relation_qk = key - query  # Peut exploser en magnitude !

# Avec normalisation (PTv2)
query = Linear(feat) → BatchNorm → ReLU  # Contrôlé et stable
key = Linear(feat) → BatchNorm → ReLU
relation_qk = key - query  # Magnitude stable
```

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

# Agrégation
output = sum(attention_weights * value)
```

**PTv2 :** L'encodage est ajouté à la relation Q-K **ET aux values**

```python
# Code PTv2
pe_bias = MLP(relative_positions)  # (N, K, C)

# Sur la relation Q-K (comme PTv1)
relation_qk = (key - query) + pe_bias

# NOUVEAU: aussi sur les values !
value = value + pe_bias

# Agrégation (values contiennent maintenant l'info géométrique)
output = sum(attention_weights * value)
```

**Exemple physique comparatif :**

Imaginons un point représentant le coin d'une table avec 3 voisins :

```
Voisin 1: dessus de la table    (Δpos = [0, 0, 0.05m])
Voisin 2: pied de table         (Δpos = [0, 0, -0.8m])
Voisin 3: air environnant       (Δpos = [0.5, 0, 0])
```

**Avec PTv1 :**
```python
# Encodage position
pe = MLP([0, 0, 0.05]) → [0.8, 0.1, 0.1, ...]  # proche

# Application sur attention seulement
relation_qk[voisin_1] = (key - query) + pe
# → Le poids d'attention capture la géométrie

# Mais la value reste inchangée !
value[voisin_1] = [semantic_features...]  # Pas d'info géométrique

# Agrégation
output = 0.6 × value[voisin_1] + ...
#        ↑ poids tient compte de la géométrie
#            ↑ mais la value non !
```

**Avec PTv2 :**
```python
# Encodage position
pe = MLP([0, 0, 0.05]) → [0.8, 0.1, 0.1, ...]

# Application sur attention (comme PTv1)
relation_qk[voisin_1] = (key - query) + pe

# NOUVEAU: aussi sur la value !
value[voisin_1] = value_original + pe
# → La value contient maintenant l'info : "je suis proche et au-dessus"

# Agrégation
output = 0.6 × value[voisin_1] + ...
#        ↑ poids géométrique
#            ↑ value aussi géométrique !
```

**Intuition :** PTv2 permet au modèle d'apprendre des patterns du type :

*"Quand j'agrège des features de voisins proches au-dessus de moi (dessus de table), leurs features doivent être modifiées différemment que des voisins lointains en-dessous (pied de table)"*

PTv1 ne pouvait capturer cela que via les poids d'attention - les features agrégées étaient "aveugles" à la géométrie.

---

### Innovation 3 : Masking des Voisins Invalides

**Problème commun :** Dans un batch, certains points ont moins de K voisins.

**PTv1 : Pas de masking explicite**

```python
# PTv1 assume que tous les voisins sont valides
# Si un point a seulement 10 voisins au lieu de 16 :
# - Les 10 vrais voisins sont dans reference_index
# - Les 6 restants sont des duplicates du dernier voisin valide
#   (comportement de queryandgroup avec padding)

attention_weights = softmax(attention_scores)  # (N, K, C/G)
# Les poids des voisins "padding" ne sont PAS mis à zéro
# → Ils contribuent avec des features dupliquées
```

**PTv2 : Masking explicite avec -1**

```python
# PTv2 utilise -1 pour marquer les voisins invalides
reference_index[point_42] = [15, 23, 8, -1, -1, -1, ...]
#                            ↑ valides  ↑ invalides (padding)

# Création du masque
mask = torch.sign(reference_index + 1)  # (N, K)
# sign(-1 + 1) = sign(0) = 0  → voisin invalide
# sign(i + 1)  = sign(>0) = 1 → voisin valide

# Application sur les poids
attention_weights = attention_weights * mask.unsqueeze(-1)
# Les poids des voisins invalides deviennent exactement 0
```

**Comparaison sur un exemple :**

```python
# Point isolé avec 3 vrais voisins sur K=8

# PTv1 behavior
reference_index = [15, 23, 8, 8, 8, 8, 8, 8]  # duplicates du dernier
attention_weights = [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05]
# Les voisins "padding" (indices 3-7) ont des poids non-nuls
# → Agrégation polluée par 5× la même feature (voisin 8)

# PTv2 behavior
reference_index = [15, 23, 8, -1, -1, -1, -1, -1]  # explicit invalid
mask = [1, 1, 1, 0, 0, 0, 0, 0]
attention_weights = [0.25, 0.20, 0.15, 0, 0, 0, 0, 0]
# Les voisins invalides sont complètement ignorés ✓
```

**Impact :** Plus robuste pour les nuages avec densité variable ou points isolés.

---

### Innovation 4 : GroupedLinear vs MLP Standard

**PTv1 : MLP standard pour générer les poids d'attention**

```python
# PTv1
self.linear_w = nn.Sequential(
    nn.BatchNorm1d(mid_planes),
    nn.ReLU(inplace=True),
    nn.Linear(mid_planes, mid_planes // share_planes),  # C → C/G
    nn.BatchNorm1d(mid_planes // share_planes),
    nn.ReLU(inplace=True),
    nn.Linear(out_planes // share_planes, out_planes // share_planes)  # C/G → C/G
)

# Paramètres totaux pour C=64, G=8:
# Première Linear: 64 × 8 = 512 paramètres
# Deuxième Linear: 8 × 8 = 64 paramètres
# Total: ~576 paramètres
```

**PTv2 : GroupedLinear**

```python
# PTv2
self.weight_encoding = nn.Sequential(
    GroupedLinear(embed_channels, groups, groups),  # C → G
    PointBatchNorm(groups),
    nn.ReLU(inplace=True),
    nn.Linear(groups, groups)  # G → G
)

# Paramètres totaux pour C=64, G=8:
# GroupedLinear: 64 paramètres (vecteur partagé)
# Linear: 8 × 8 = 64 paramètres
# Total: ~128 paramètres
```

**Réduction : 576 → 128 paramètres (4.5× moins !)**

---

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

### Flux Complet Comparatif

**PTv1 :**
```
1. Q, K, V = Linear(feat)
2. Grouper K voisins (K-NN à chaque couche)
3. PE = MLP(relative_pos)
4. relation = (K - Q) + PE
5. weights = MLP_standard(relation) → softmax
6. output = sum(weights × V)
```

**PTv2 :**
```
1. Q, K = Linear(feat) → BatchNorm → ReLU
   V = Linear(feat)
2. Grouper K voisins (référence pré-calculée)
3. PE = MLP_normalized(relative_pos)
4. relation = (K - Q) + PE
5. V = V + PE  ← NOUVEAU
6. weights = GroupedLinear(relation) → softmax
7. mask = sign(ref_index + 1)  ← NOUVEAU
8. weights = weights × mask
9. output = sum(weights × V)
```

---

### Tableau Récapitulatif

| Innovation | PTv1 | PTv2 | Gain |
|------------|------|------|------|
| **Normalisation Q/K/V** | ❌ Non | ✅ Oui | Stabilité training |
| **PE sur values** | ❌ Non | ✅ Oui | Features géométriques |
| **Masking invalides** | ❌ Non | ✅ Oui | Robustesse densité variable |
| **Paramètres weight gen** | ~576 | ~128 | 4.5× réduction |
| **K-NN par couche** | Oui | Non (pré-calc) | ~6× speedup |

PTv2 améliore donc **significativement** l'attention locale tout en réduisant les paramètres et le coût computationnel ! 🎯

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

**Pourquoi ça marche ?**

1. **Régularisation :** Force chaque bloc à être utile indépendamment
2. **Gradient flow :** Crée des "chemins courts" pendant l'entraînement
3. **Ensemble implicite :** Entraîne effectivement plusieurs sous-réseaux de profondeurs différentes
4. **Réduit l'overfitting :** Les blocs ne peuvent pas trop dépendre les uns des autres

**Drop Path Rate Scheduler :**

Dans PTv2, le `drop_path_rate` augmente progressivement à travers les couches :

```python
# Configuration PTv2 avec drop_path_rate = 0.3
enc_depths = [2, 2, 6, 2]  # 12 couches au total

drop_path_rates = linspace(0, 0.3, sum(enc_depths))
# [0.00, 0.03, 0.05, 0.08, 0.11, 0.14, 0.16, 0.19, 0.22, 0.24, 0.27, 0.30]

# Les premières couches ont drop_path_rate faible (plus stables)
# Les dernières couches ont drop_path_rate élevé (plus régularisées)
```

**Intuition :** Les couches profondes bénéficient plus de la régularisation car elles ont tendance à overfitter.

---

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

### Comparaison des Coûts

**Pour un Encoder avec 4 niveaux × 6 couches chacun (24 couches totales) :**

| Opération | PTv1 | PTv2 | Speedup |
|-----------|------|------|---------|
| **K-NN queries** | 24 fois | 4 fois | 6× |
| **K-NN cost** | 24 × O(N log N) | 4 × O(N log N) | 6× |
| **Memory** | Recalculé chaque fois | Stocké et réutilisé | - |

**Note :** Le speedup réel dépend du ratio (coût K-NN / coût attention), mais empiriquement PTv2 est ~2-3× plus rapide en pratique sur cette optimisation seule.

### Gestion du reference_index

**Structure du reference_index :**

```python
# reference_index: (N, K)
# Pour chaque point n ∈ [0, N), contient les indices de ses K voisins

# Exemple avec N=5 points, K=3 voisins
reference_index = [
    [1, 2, 4],    # Point 0: ses 3 voisins sont les points 1, 2, 4
    [0, 2, 3],    # Point 1: ses 3 voisins sont les points 0, 2, 3
    [0, 1, 3],    # Point 2: ...
    [1, 2, 4],    # Point 3: ...
    [0, 2, 3]     # Point 4: ...
]
```

**Gestion des voisins invalides (padding) :**

```python
# Point isolé avec seulement 2 vrais voisins (K=3)
reference_index[point_42] = [15, 23, -1]
#                                    ↑ Pas assez de voisins → -1 (invalide)

# Le masking dans GroupedVectorAttention gère automatiquement
mask = sign(reference_index + 1)  # [1, 1, 0]
attention_weights = attention_weights * mask.unsqueeze(-1)
# Le voisin invalide (-1) est ignoré ✓
```

### Exemple Complet

**Configuration :**
- BlockSequence avec `depth=6` (6 blocks)
- `neighbours=16` (K=16 voisins)
- N=1000 points, C=64 channels

**Flux :**

```python
# Input
coord: (1000, 3)
feat: (1000, 64)
offset: (B,)

# Étape 1: K-NN une fois
reference_index = knn_query(K=16, coord, offset)
# reference_index: (1000, 16)
# Pour chaque point: indices de ses 16 voisins

# Étape 2: Block 1
coord, feat, offset = Block_1(coord, feat, offset, reference_index)
# coord inchangé: (1000, 3)
# feat transformé: (1000, 64)

# Étape 3: Block 2 (réutilise reference_index)
coord, feat, offset = Block_2(coord, feat, offset, reference_index)
# coord toujours inchangé: (1000, 3)
# feat transformé: (1000, 64)

# Étapes 4-6: Blocks 3-6 (tous réutilisent reference_index)
...

# Output
coord: (1000, 3)  ← Identique à l'input
feat: (1000, 64)  ← Transformé par 6 couches d'attention
offset: (B,)      ← Identique à l'input
```

**Visualisation :**

```
Point central ◉ à la position (x, y, z)

Au début de BlockSequence:
  K-NN → Trouve ses 16 voisins: ●₁, ●₂, ..., ●₁₆
  Stocke dans reference_index[◉] = [idx₁, idx₂, ..., idx₁₆]

Block 1:
  Attention sur ●₁, ●₂, ..., ●₁₆ (lookup via reference_index)
  → Features de ◉ mises à jour
  → Position de ◉ INCHANGÉE

Block 2:
  Attention sur les MÊMES ●₁, ●₂, ..., ●₁₆ (lookup via reference_index)
  → Features de ◉ encore mises à jour
  → Position de ◉ toujours INCHANGÉE

...

Block 6:
  Attention sur les MÊMES voisins
  → Features finales de ◉
```

Les voisins géométriques restent identiques, mais leurs **features évoluent** à chaque couche !

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

### Comparaison avec PTv1

| Aspect | PTv1 | PTv2 |
|--------|------|------|
| **Initial embedding** | ❌ Aucun | ✅ GVAPatchEmbed |
| **Première opération** | TransitionDown (downsampling immédiat) | GVAPatchEmbed (attention à pleine résolution) |
| **Philosophy** | Downsample vite | Apprendre des features riches d'abord |

**PTv1 :**
```
Input (N, in_channels)
    ↓
TransitionDown (stride=1)  ← Simple Linear + BN + ReLU
    ↓
PointTransformerBlock
    ↓
TransitionDown (stride=4)  ← Downsampling immédiat
```

**PTv2 :**
```
Input (N, in_channels)
    ↓
GVAPatchEmbed:
  - Linear + BN + ReLU
  - depth × Block (GroupedVectorAttention)
    ↓
(N, embed_channels)  ← Features riches avant downsampling
    ↓
Encoder 1 (GridPool)  ← Premier downsampling
```

### Pourquoi c'est Important ?

**Analogie avec les CNNs :**

Dans les CNNs modernes (ResNet, EfficientNet), on a un **"stem"** initial qui traite l'image à haute résolution avant le pooling :

```python
# ResNet stem
Input (224×224)
    ↓
Conv 7×7, stride=2  → (112×112)
    ↓
MaxPool 3×3, stride=2 → (56×56)
    ↓
ResNet blocks...
```

**PTv2 adopte la même philosophie :**
- Apprendre des features riches à **pleine résolution** avant de downsampler
- Permet de capturer des détails fins dès le début
- Les features initiales de meilleure qualité aident tout le réseau

### Configuration Typique

```python
# PTv2 default config
GVAPatchEmbed(
    in_channels=6,          # xyz + rgb
    embed_channels=48,      # Dimension d'embedding
    groups=6,               # 6 groupes pour l'attention
    depth=1,                # 1 block (peut être plus)
    neighbours=8            # K=8 voisins
)
```

Avec `depth=1`, c'est modeste, mais déjà bénéfique. Certaines variantes utilisent `depth=2` pour des features encore plus riches.

# GridPool : Downsampling par Voxelisation

## Vue d'ensemble

`GridPool` est l'une des innovations majeures de PTv2, remplaçant le **Furthest Point Sampling (FPS)** de PTv1 par une approche basée sur la **voxelisation**.

{% include figure.liquid path="assets/img/poinTransformerV2/gridPool.svg" class="img-fluid rounded z-depth-1" %}

### Comparaison : FPS vs Grid Pooling

| Aspect | PTv1 (FPS) | PTv2 (Grid Pooling) |
|--------|------------|---------------------|
| **Complexité** | O(N²) | **O(N)** |
| **Méthode** | Sélection itérative des points les plus éloignés | Voxelisation + agrégation |
| **Déterminisme** | Non (ordre dépend du point de départ) | Oui (basé sur la grille) |
| **Speedup** | - | **3-5× plus rapide** |
| **Couverture spatiale** | Maximise les distances | Uniforme par design |
| **Unpooling** | K-NN interpolation (coûteux) | **Map unpooling** (gratuit) |

---

## Problème avec FPS (PTv1)

### Algorithme FPS

**Furthest Point Sampling** sélectionne itérativement les points les plus éloignés :

```python
def furthest_point_sampling(points, num_samples):
    # 1. Choisir un point de départ aléatoire
    selected = [random_point]
    
    # 2. Répéter jusqu'à avoir num_samples points
    for i in range(num_samples - 1):
        # Pour chaque point non sélectionné:
        #   - Calculer sa distance au point sélectionné le plus proche
        distances = [min_distance_to_selected(p) for p in points]
        
        # Sélectionner le point le plus éloigné
        farthest = argmax(distances)
        selected.append(farthest)
    
    return selected
```

**Exemple visuel :**

```
Nuage de 16 points, on veut en garder 4:

Étape 1: Point aléatoire
    ●────●────●────●
    │    │    │    │
    ◉────●────●────●   ← Point de départ
    │    │    │    │
    ●────●────●────●
    │    │    │    │
    ●────●────●────●

Étape 2: Point le plus éloigné
    ◉────●────●────●
    │    │    │    │
    ◉────●────●────●
    │    │    │    │
    ●────●────●────●
    │    │    │    │
    ●────●────●────◉   ← Coin opposé (le plus loin)

Étape 3: Encore le plus éloigné
    ◉────●────●────◉
    │    │    │    │
    ◉────●────●────●
    │    │    │    │
    ●────●────●────●
    │    │    │    │
    ●────●────●────◉

Étape 4: Dernier point
    ◉────●────●────◉
    │    │    │    │
    ◉────●────●────●
    │    │    │    │
    ●────●────●────●
    │    │    │    │
    ◉────●────●────◉   ← 4 points bien espacés
```

**Problèmes :**

1. **Complexité O(N²)** : Pour sélectionner M points parmi N, on doit calculer O(M × N) distances
2. **Coût élevé** : Pour N=100k points → M=25k points, c'est 2.5 milliards de calculs de distance !
3. **Non-déterministe** : Dépend du point de départ aléatoire
4. **Pas de mapping** : On perd l'information de correspondance pour l'unpooling

```python
# Coût FPS pour downsampler 4 niveaux dans PTv1
Level 1: FPS(N → N/4)     → O(N²)
Level 2: FPS(N/4 → N/16)  → O((N/4)²)
Level 3: FPS(N/16 → N/64) → O((N/16)²)
Level 4: FPS(N/64 → N/256)→ O((N/64)²)

Total: O(N²) dominant pour les premiers niveaux
```

---

## Solution : Grid Pooling (PTv2)

### Principe : Voxelisation

Au lieu de sélectionner des points, on **partitionne l'espace en voxels** (cubes 3D) et on agrège tous les points d'un même voxel.

```
Nuage de points + grille 3D:

        grid_size = 0.5m
        
    ┌─────┬─────┬─────┬─────┐
    │ ●   │     │  ●  │     │
    │   ● │     │     │  ●  │
    ├─────┼─────┼─────┼─────┤
    │     │ ●●  │     │     │
    │ ●   │  ●  │  ●  │     │
    ├─────┼─────┼─────┼─────┤
    │  ●  │     │ ●   │  ●  │
    │     │  ●  │   ● │     │
    ├─────┼─────┼─────┼─────┤
    │     │     │     │ ●●  │
    │  ●  │  ●  │  ●  │  ●  │
    └─────┴─────┴─────┴─────┘

Après voxelisation (1 point par voxel):

    ┌─────┬─────┬─────┬─────┐
    │  ◉  │     │  ◉  │  ◉  │
    │     │     │     │     │
    ├─────┼─────┼─────┼─────┤
    │  ◉  │  ◉  │  ◉  │     │
    │     │     │     │     │
    ├─────┼─────┼─────┼─────┤
    │  ◉  │  ◉  │  ◉  │  ◉  │
    │     │     │     │     │
    ├─────┼─────┼─────┼─────┤
    │  ◉  │  ◉  │  ◉  │  ◉  │
    │     │     │     │     │
    └─────┴─────┴─────┴─────┘
```

### Algorithme Étape par Étape

#### Étape 1 : Projection des Features

```python
# Input
coord: (N, 3)
feat: (N, in_channels)

# Projection linéaire + normalisation
feat = Linear(feat)  # (N, in_channels) → (N, out_channels)
feat = BatchNorm1d(feat) + ReLU(feat)
# feat: (N, out_channels)
```

#### Étape 2 : Calcul du Point de Départ par Nuage

Chaque nuage dans le batch a besoin d'un point de référence (coin minimal) :

```python
# Conversion offset → batch
batch = offset2batch(offset)  # (N,)
# batch[i] indique à quel nuage appartient le point i

# Calcul du coin minimal de chaque nuage
start = segment_csr(coord, batch_ptr, reduce="min")
# start: (B, 3) - coin minimal (x_min, y_min, z_min) de chaque nuage
```

**Exemple :**

```python
# 2 nuages dans le batch
Nuage 1: points aux positions [[1,2,3], [2,3,4], [1.5,2.5,3.5]]
    → start[0] = [1, 2, 3]  # Minimum de chaque dimension

Nuage 2: points aux positions [[5,6,7], [6,7,8]]
    → start[1] = [5, 6, 7]
```

#### Étape 3 : Voxelisation

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

**Exemple concret :**

```python
# Voxel contient 3 points avec features:
feat1 = [0.8, 0.2, 0.5, 0.1]
feat2 = [0.3, 0.9, 0.4, 0.6]
feat3 = [0.5, 0.4, 0.7, 0.3]

# Max pooling (canal par canal)
feat_pooled = [
    max(0.8, 0.3, 0.5) = 0.8,  # Canal 0
    max(0.2, 0.9, 0.4) = 0.9,  # Canal 1
    max(0.5, 0.4, 0.7) = 0.7,  # Canal 2
    max(0.1, 0.6, 0.3) = 0.6   # Canal 3
]
= [0.8, 0.9, 0.7, 0.6]
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

## Exemple Complet Numérique

**Configuration :**
- N = 12 points
- grid_size = 1.0
- in_channels = 3, out_channels = 4

**Input :**

```python
coord = [
    [0.2, 0.3, 0.1],  # Point 0
    [0.8, 0.5, 0.2],  # Point 1
    [0.1, 0.2, 0.15], # Point 2
    [1.2, 0.4, 0.3],  # Point 3
    [1.4, 0.6, 0.4],  # Point 4
    [1.5, 1.8, 0.1],  # Point 5
    [0.1, 1.1, 0.9],  # Point 6
    [2.1, 0.3, 0.2],  # Point 7
    [2.3, 0.5, 0.4],  # Point 8
    [0.9, 1.9, 0.3],  # Point 9
    [1.1, 1.8, 0.5],  # Point 10
    [2.2, 1.1, 0.6],  # Point 11
]  # (12, 3)

feat = [...] # (12, 3)
offset = [12]  # 1 nuage avec 12 points
```

**Étape 1 : Projection**

```python
feat = Linear(feat) → BatchNorm → ReLU
# feat: (12, 4)
```

**Étape 2 : Start point**

```python
start = [0.1, 0.2, 0.1]  # Minimum de chaque dimension
```

**Étape 3 : Voxelisation**

```python
# Coordonnées normalisées
coord_norm = coord - start

# Assignation aux voxels (floor(coord_norm / 1.0))
cluster = [
    0,  # (0,0,0)  Points 0, 1, 2
    0,  # (0,0,0)
    0,  # (0,0,0)
    1,  # (1,0,0)  Points 3, 4
    1,  # (1,0,0)
    2,  # (1,1,0)  Points 5, 9, 10
    3,  # (0,1,0)  Point 6
    4,  # (2,0,0)  Points 7, 8
    4,  # (2,0,0)
    2,  # (1,1,0)
    2,  # (1,1,0)
    5,  # (2,1,0)  Point 11
]
```

**Étape 4 : Unique**

```python
unique = [0, 1, 2, 3, 4, 5]  # 6 voxels
counts = [3, 2, 3, 1, 2, 1]  # Points par voxel
cluster_inverse = [0, 0, 0, 1, 1, 2, 3, 4, 4, 2, 2, 5]
```

**Étape 5 : Tri**

```python
sorted_indices = [0, 1, 2, 3, 4, 5, 9, 10, 6, 7, 8, 11]
idx_ptr = [0, 3, 5, 8, 9, 11, 12]
```

**Étape 6 : Agrégation Coordonnées**

```python
coord_pooled = [
    mean(coord[0,1,2]) = [0.37, 0.33, 0.15],   # Voxel 0
    mean(coord[3,4])   = [1.3, 0.5, 0.35],     # Voxel 1
    mean(coord[5,9,10])= [1.17, 1.83, 0.3],    # Voxel 2
    coord[6]           = [0.1, 1.1, 0.9],      # Voxel 3
    mean(coord[7,8])   = [2.2, 0.4, 0.3],      # Voxel 4
    coord[11]          = [2.2, 1.1, 0.6],      # Voxel 5
]  # (6, 3)
```

**Étape 7 : Agrégation Features**

```python
feat_pooled = [
    max(feat[0,1,2], dim=0),   # (4,)
    max(feat[3,4], dim=0),     # (4,)
    max(feat[5,9,10], dim=0),  # (4,)
    feat[6],                   # (4,)
    max(feat[7,8], dim=0),     # (4,)
    feat[11],                  # (4,)
]  # (6, 4)
```

**Output :**

```python
coord_pooled: (6, 3)     # 6 voxels au lieu de 12 points
feat_pooled: (4, 4)      # 6 voxels avec 4 channels
offset_pooled: [6]       # 1 nuage avec 6 points
cluster_inverse: (12,)   # Mapping points → voxels
```

**Réduction :** 12 points → 6 voxels (2× downsampling)

---

## Complexité et Performance

### Complexité Algorithmique

```python
# GridPool
1. Linear: O(N × C)
2. Start computation: O(N)
3. Voxel assignment: O(N)
4. Unique: O(N log N)
5. Sort: O(N log N)
6. Aggregation: O(N)

Total: O(N log N)  ← Dominé par le tri
```

### Comparaison avec FPS

| Opération | FPS (PTv1) | Grid Pooling (PTv2) |
|-----------|------------|---------------------|
| **Complexité** | O(N²) | O(N log N) |
| **N=10k** | 100M ops | ~133k ops |
| **N=100k** | 10B ops | ~1.7M ops |
| **Speedup empirique** | - | **3-5×** |

### Visualisation du Speedup

```
Temps de downsampling (N→N/4):

FPS (PTv1):
N=10k:  ████████████ 120ms
N=50k:  ████████████████████████████████ 3000ms
N=100k: ████████████████████████████████████████████████████ 12000ms

Grid Pooling (PTv2):
N=10k:  ███ 30ms
N=50k:  ████████ 200ms
N=100k: ██████████████ 350ms

Speedup: ~4×  ~15×  ~34×
```

---

## Avantages de Grid Pooling

### 1. Vitesse

**3-5× plus rapide** que FPS, surtout pour les grands nuages.

### 2. Déterminisme

```python
# FPS: résultat dépend du point de départ aléatoire
run1 = FPS(points, seed=42)
run2 = FPS(points, seed=43)
# run1 ≠ run2  (points sélectionnés différents)

# Grid Pooling: toujours le même résultat
run1 = GridPool(points, grid_size=0.06)
run2 = GridPool(points, grid_size=0.06)
# run1 == run2  (voxels identiques)
```

### 3. Couverture Spatiale Uniforme

```python
# FPS: peut "rater" des régions
    ●─────●─────────●
    │              │
    │   ●●●●       │  ← Zone dense peu échantillonnée
    │              │
    ●─────────────●

# Grid Pooling: couverture garantie
    ●─────●─────●─────●
    │     │     │     │
    ├─────┼─────┼─────┤
    │  ●  │ ●●  │     │  ← Chaque voxel représenté
    ├─────┼─────┼─────┤
    ●─────●─────●─────●
```

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

### Comparaison : Interpolation vs Map Unpooling

| Aspect | PTv1 (K-NN Interpolation) | PTv2 (Map Unpooling) |
|--------|---------------------------|----------------------|
| **Méthode** | K-NN + weighted average | Lookup direct via cluster |
| **Complexité** | O(M log N) K-NN search | **O(1) lookup** |
| **Information utilisée** | Distances géométriques | Mapping du downsampling |
| **Coût** | Coûteux (recherche K-NN) | **Gratuit** (indexing) |
| **Précision** | Interpolation lisse | Mapping exact |

---

## Problème avec K-NN Interpolation (PTv1)

### Algorithme d'Interpolation PTv1

Dans PTv1, pour passer de M points (basse résolution) à N points (haute résolution), on utilise une **interpolation par K-NN** :

```python
def interpolation(coord_low, coord_high, feat_low, K=3):
    """
    Args:
        coord_low: (M, 3) - positions basse résolution
        coord_high: (N, 3) - positions haute résolution (cibles)
        feat_low: (M, C) - features basse résolution
    Returns:
        feat_high: (N, C) - features interpolées
    """
    N = coord_high.shape[0]
    
    for n in range(N):
        # 1. Trouver les K voisins les plus proches dans coord_low
        distances = ||coord_low - coord_high[n]||  # (M,)
        k_nearest = argsort(distances)[:K]  # K indices
        
        # 2. Poids inversement proportionnels aux distances
        dists = distances[k_nearest]  # (K,)
        weights = 1.0 / (dists + ε)
        weights = weights / sum(weights)  # Normalisation
        
        # 3. Moyenne pondérée
        feat_high[n] = Σ weights[k] × feat_low[k_nearest[k]]
    
    return feat_high
```

**Visualisation :**

```
Basse résolution (M=4 points):        Haute résolution (N=9 cibles):
    
    ◉₁         ◉₂                         ●₁    ●₂    ●₃
                                          
                                          ●₄    ●₅    ●₆
                                          
    ◉₃         ◉₄                         ●₇    ●₈    ●₉

Pour interpoler ●₅ (centre):
1. K-NN: trouver les 3 plus proches parmi {◉₁, ◉₂, ◉₃, ◉₄}
   → ◉₁ (dist=1.4), ◉₂ (dist=1.4), ◉₃ (dist=1.4), ◉₄ (dist=1.4)
   → Tous équidistants !

2. Poids: w₁=w₂=w₃=w₄ = 0.25

3. Interpolation:
   feat[●₅] = 0.25×feat[◉₁] + 0.25×feat[◉₂] + 0.25×feat[◉₃] + 0.25×feat[◉₄]
```

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

**2. Perte d'information :**

L'interpolation crée de **nouvelles** features qui n'existaient pas dans la résolution d'origine :

```python
# GridPool: Point original → Voxel A
point_42: [x, y, z], feat_original

# Après downsampling
voxel_A: [x_mean, y_mean, z_mean], feat_pooled

# Après interpolation (PTv1)
point_42: [x, y, z], feat_interpolated  # ≠ feat_original !
# L'interpolation "invente" des features
```

**3. Non-déterminisme pour les cas ambigus :**

```python
# Point équidistant de plusieurs voisins
distances = [1.0, 1.0, 1.0, 1.0]  # 4 voisins équidistants

# Avec K=3, lesquels choisir ?
# → Dépend de l'ordre dans le tableau (non déterministe)
```

---

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

**Explication :**

```python
# Exemple avec M=4 voxels, N=12 points
feat_low_proj = [
    [f₀⁰, f₀¹, f₀², f₀³],  # Features du voxel 0
    [f₁⁰, f₁¹, f₁², f₁³],  # Features du voxel 1
    [f₂⁰, f₂¹, f₂², f₂³],  # Features du voxel 2
    [f₃⁰, f₃¹, f₃², f₃³],  # Features du voxel 3
]  # (4, 4)

cluster = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3]

# Map unpooling
feat_mapped = [
    feat_low_proj[0],  # Point 0 → voxel 0
    feat_low_proj[0],  # Point 1 → voxel 0
    feat_low_proj[0],  # Point 2 → voxel 0
    feat_low_proj[1],  # Point 3 → voxel 1
    feat_low_proj[1],  # Point 4 → voxel 1
    feat_low_proj[2],  # Point 5 → voxel 2
    feat_low_proj[2],  # Point 6 → voxel 2
    feat_low_proj[2],  # Point 7 → voxel 2
    feat_low_proj[2],  # Point 8 → voxel 2
    feat_low_proj[3],  # Point 9 → voxel 3
    feat_low_proj[3],  # Point 10 → voxel 3
    feat_low_proj[3],  # Point 11 → voxel 3
]  # (12, 4)
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

## Exemple Complet Numérique

**Configuration :**
- M = 4 voxels (basse résolution)
- N = 12 points (haute résolution)
- in_ch = 3, skip_ch = 3, out_ch = 4

**Inputs :**

```python
# Basse résolution (voxels du GridPool)
coord_low = [
    [0.37, 0.33, 0.15],  # Voxel 0 (moyenne de 3 points)
    [1.3, 0.5, 0.35],    # Voxel 1 (moyenne de 2 points)
    [1.17, 1.83, 0.3],   # Voxel 2 (moyenne de 4 points)
    [0.1, 1.1, 0.9],     # Voxel 3 (moyenne de 3 points)
]  # (4, 3)

feat_low = [
    [0.8, 0.9, 0.7],  # Features voxel 0
    [0.6, 0.5, 0.8],  # Features voxel 1
    [0.9, 0.7, 0.6],  # Features voxel 2
    [0.4, 0.6, 0.9],  # Features voxel 3
]  # (4, 3)

# Haute résolution (points originaux de l'encodeur)
coord_skip = [
    [0.2, 0.3, 0.1],   # Point 0 (était dans voxel 0)
    [0.8, 0.5, 0.2],   # Point 1 (était dans voxel 0)
    [0.1, 0.2, 0.15],  # Point 2 (était dans voxel 0)
    [1.2, 0.4, 0.3],   # Point 3 (était dans voxel 1)
    [1.4, 0.6, 0.4],   # Point 4 (était dans voxel 1)
    [1.5, 1.8, 0.1],   # Point 5 (était dans voxel 2)
    [0.1, 1.1, 0.9],   # Point 6 (était dans voxel 3)
    [2.1, 0.3, 0.2],   # Point 7 (était dans voxel 2)
    [2.3, 0.5, 0.4],   # Point 8 (était dans voxel 2)
    [0.9, 1.9, 0.3],   # Point 9 (était dans voxel 2)
    [1.1, 1.8, 0.5],   # Point 10 (était dans voxel 3)
    [2.2, 1.1, 0.6],   # Point 11 (était dans voxel 3)
]  # (12, 3)

feat_skip = [...] # (12, 3)

# Cluster mapping (du GridPool correspondant)
cluster = [0, 0, 0, 1, 1, 2, 3, 2, 2, 2, 3, 3]
```

**Étape 1 : Projection feat_low**

```python
feat_low_proj = Linear(feat_low) → BatchNorm → ReLU
# feat_low_proj: (4, 4) par exemple
feat_low_proj = [
    [0.9, 0.8, 0.6, 0.7],  # Voxel 0
    [0.7, 0.6, 0.9, 0.5],  # Voxel 1
    [0.8, 0.7, 0.5, 0.9],  # Voxel 2
    [0.5, 0.7, 0.8, 0.6],  # Voxel 3
]
```

**Étape 2 : Map Unpooling**

```python
feat_mapped = feat_low_proj[cluster]  # Simple indexing !

# Point 0 → cluster[0]=0 → feat_low_proj[0]
feat_mapped[0] = [0.9, 0.8, 0.6, 0.7]

# Point 1 → cluster[1]=0 → feat_low_proj[0]
feat_mapped[1] = [0.9, 0.8, 0.6, 0.7]

# Point 2 → cluster[2]=0 → feat_low_proj[0]
feat_mapped[2] = [0.9, 0.8, 0.6, 0.7]

# Point 3 → cluster[3]=1 → feat_low_proj[1]
feat_mapped[3] = [0.7, 0.6, 0.9, 0.5]

# Point 4 → cluster[4]=1 → feat_low_proj[1]
feat_mapped[4] = [0.7, 0.6, 0.9, 0.5]

# Point 5 → cluster[5]=2 → feat_low_proj[2]
feat_mapped[5] = [0.8, 0.7, 0.5, 0.9]

# Point 6 → cluster[6]=3 → feat_low_proj[3]
feat_mapped[6] = [0.5, 0.7, 0.8, 0.6]

# Point 7 → cluster[7]=2 → feat_low_proj[2]
feat_mapped[7] = [0.8, 0.7, 0.5, 0.9]

# Point 8 → cluster[8]=2 → feat_low_proj[2]
feat_mapped[8] = [0.8, 0.7, 0.5, 0.9]

# Point 9 → cluster[9]=2 → feat_low_proj[2]
feat_mapped[9] = [0.8, 0.7, 0.5, 0.9]

# Point 10 → cluster[10]=3 → feat_low_proj[3]
feat_mapped[10] = [0.5, 0.7, 0.8, 0.6]

# Point 11 → cluster[11]=3 → feat_low_proj[3]
feat_mapped[11] = [0.5, 0.7, 0.8, 0.6]

# Résultat: (12, 4)
```

**Étape 3 : Projection feat_skip**

```python
feat_skip_proj = Linear(feat_skip) → BatchNorm → ReLU
# feat_skip_proj: (12, 4)
```

**Étape 4 : Fusion**

```python
feat_fused = feat_mapped + feat_skip_proj
# feat_fused: (12, 4)

# Exemple pour point 0
feat_fused[0] = [0.9, 0.8, 0.6, 0.7] + [0.3, 0.4, 0.5, 0.2]
              = [1.2, 1.2, 1.1, 0.9]
```

**Output :**

```python
coord_skip: (12, 3)   # Positions haute résolution
feat_fused: (12, 4)   # Features fusionnées
offset_skip: (B,)
```

---

## Comparaison Visuelle : Interpolation vs Map Unpooling

### PTv1 : K-NN Interpolation

```
Downsampling (FPS):              Upsampling (K-NN Interpolation):
                                 
16 points → 4 points              4 points → 16 points
                                  
●●●●                              ●───●───●───●
●●●●    → FPS →    ◉   ◉         │   │   │   │
●●●●                              ●───●───●───●   ← K-NN pour chacun
●●●●               ◉   ◉         │   │   │   │
                                  ●───●───●───●
                                  │   │   │   │
                                  ●───●───●───●

Chaque ● cible:
  1. Cherche K=3 voisins parmi les 4 ◉
  2. Calcule les distances
  3. Moyenne pondérée

Coût: 16 × O(4 log 4) = O(N × M log M)
```

### PTv2 : Map Unpooling

```
Downsampling (GridPool):         Upsampling (Map Unpooling):

16 points → 4 voxels              4 voxels → 16 points
+ cluster mapping

●●●●                              ●●●●
●●●●  → Grid →  [0,0,0,0,         [0,0,0,0,  → ●●●●
●●●●             1,1,1,1,         1,1,1,1,     ●●●●
●●●●             2,2,2,2,         2,2,2,2,     ●●●●
                 3,3,3,3]         3,3,3,3]

Voxel 0 → 4 points                feat[●₀] = feat_voxel[0]
Voxel 1 → 4 points                feat[●₁] = feat_voxel[0]
Voxel 2 → 4 points                feat[●₂] = feat_voxel[0]
Voxel 3 → 4 points                feat[●₃] = feat_voxel[0]
                                  ...

Coût: 16 × O(1) = O(N)  ← Lookup direct !
```

---

## Avantages du Map Unpooling

### 1. Vitesse

**Complexité :**

```python
# PTv1 : K-NN Interpolation
Complexité: O(N × M log M)

# PTv2 : Map Unpooling
Complexité: O(N)  ← Juste un indexing !

# Exemple: N=100k, M=25k
PTv1: 100k × 25k × log(25k) ≈ 35 milliards d'ops
PTv2: 100k ≈ 100k ops

Speedup: ~350,000× sur cette opération !
```

**En pratique, le speedup global est ~10-20× car l'interpolation n'est qu'une partie du décodeur.**

### 2. Exactitude

```python
# PTv1 : Interpolation
point_original → voxel_A → interpolation
feat_final ≈ feat_original  # Approximation

# PTv2 : Map exact
point_original → voxel_A → map unpooling
feat_final = feat_voxel[A]  # Exact (pas d'interpolation)
```

Les points récupèrent **exactement** les features de leur voxel d'origine, sans interpolation artificielle.

### 3. Mémoire

```python
# PTv1 : Doit stocker les K-NN indices temporaires
knn_indices: (N, K)

# PTv2 : Cluster mapping déjà stocké du downsampling
cluster: (N,)  # Déjà en mémoire, réutilisé
```

### 4. Déterminisme

```python
# PTv1 : K-NN peut être ambigu
distances = [1.0, 1.0, 1.0, 1.0]  # 4 équidistants, K=3
# → Quel trio choisir ? Non déterministe

# PTv2 : Mapping exact du downsampling
cluster[point_i] = voxel_id  # Toujours le même
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

**Flow :**
```
Input: N points, in_ch
    ↓
GridPool (voxelisation)
    ↓
Nvoxel points, embed_ch
+ cluster mapping (N,)
    ↓
BlockSequence (depth blocks)
    ↓
Output: Nvoxel points, embed_ch
+ cluster (N,)  ← Stocké pour le décodeur !
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

**Flow :**
```
Input basse résolution: M points, in_ch
Input skip (encodeur): N points, skip_ch
cluster mapping: (N,)
    ↓
UnpoolWithSkip (map unpooling + fusion)
    ↓
N points, embed_ch
    ↓
BlockSequence (depth blocks)
    ↓
Output: N points, embed_ch
```

---

## Tableau Récapitulatif des Innovations

| Composant | PTv1 | PTv2 | Gain |
|-----------|------|------|------|
| **Downsampling** | FPS O(N²) | **GridPool O(N log N)** | 3-5× speedup |
| **Upsampling** | K-NN Interpolation O(NM log M) | **Map Unpooling O(N)** | 10-20× speedup |
| **Mapping stocké** | ❌ Non | ✅ **cluster** réutilisé | Mémoire efficient |
| **Déterminisme** | ❌ Non (FPS aléatoire) | ✅ Oui (grille fixe) | Reproductibilité |
| **Exactitude** | Interpolation approximative | **Mapping exact** | Plus précis |

---

## Performance Globale : PTv1 vs PTv2

### Speedup par Composant

```
Component                PTv1        PTv2        Speedup
──────────────────────────────────────────────────────
K-NN queries            24×         4×          6×
Downsampling (FPS)      O(N²)       O(N log N)  3-5×
Upsampling (Interp)     O(NM log M) O(N)        10-20×
Attention weights       576 params  128 params  4.5×
──────────────────────────────────────────────────────
Overall                 Baseline    2-3× faster
```

### Mémoire

```python
# PTv1
- Pas de cluster mapping
- K-NN temporaire à chaque couche
Total: ~1.2× baseline

# PTv2
- cluster mapping stocké (N,) par niveau
- K-NN une fois par BlockSequence
Total: ~1.0× baseline  (plus efficient !)
```

### Précision

```
Dataset: S3DIS (segmentation sémantique)

PTv1: 70.4% mIoU
PTv2: 72.5% mIoU  (+2.1 points)

Speedup + meilleure précision ! 🎯
```

---

Voilà ! Nous avons couvert toute l'architecture de PTv2 :

✅ **GroupedLinear** : Réduction paramétrique  
✅ **GroupedVectorAttention** : Attention enrichie  
✅ **Block & BlockSequence** : Architecture résiduelle + K-NN partagé  
✅ **GVAPatchEmbed** : Embedding initial  
✅ **GridPool** : Downsampling par voxelisation  
✅ **UnpoolWithSkip** : Map unpooling + skip connections  
✅ **Encoder & Decoder** : Architecture U-Net complète  



