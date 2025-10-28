---

layout: post
title: "Point Transformer v2: Architecture and Implementation Details"
date: 2025-10-26
description: Detailed analysis of the Point Transformer v2 architecture for point-cloud segmentation and classification
tags: deep-learning point-cloud transformer architecture
categories: computer-vision
---------------------------

# Point Transformer v2: Architecture and Improvements

## Introduction

**Point Transformer v2** significantly improves its predecessor in computational efficiency and performance. Key innovations include:

* **Grid Pooling** instead of Furthest Point Sampling (3–5× faster)
* **Map Unpooling** that reuses downsampling information
* **GroupedLinear** to drastically reduce parameter count
* **Enriched vector attention** with positional encoding on the values
* **Masking of invalid neighbors** to handle point clouds of varying sizes

Before diving into the overall architecture, we first explain two fundamental innovations: GroupedLinear and GroupedVectorAttention.

---

## Overall Architecture

{% include figure.liquid path="assets/img/poinTransformerV2/main_architecture.svg" class="img-fluid rounded z-depth-1" %}

PTv2 follows a U-Net architecture with:

**Encoder (Downsampling):**

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

**Decoder (Upsampling):**

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

**Key points:**

* Each **Encoder** reduces the number of points via **GridPool** (voxelization).
* Each **Decoder** increases resolution via **Map Unpooling** + skip connection.
* **Clusters** store voxelization mapping for unpooling.
* **No Furthest Point Sampling** → much faster.

---

## GroupedLinear: Smart Parameter Reduction

### The problem with classic Linear

In a deep network, generating attention weights via standard Linear layers quickly accumulates parameters:

```python
# Classic Linear to generate 8 attention weights from 64 features
Linear(in_features=64, out_features=8)
# Parameters: 64 × 8 = 512 weights + 8 bias = 520 parameters
```

### The GroupedLinear innovation

{% include figure.liquid path="assets/img/poinTransformerV2/groupedLinear.svg" class="img-fluid rounded z-depth-1" %}

**GroupedLinear** replaces the weight matrix with a **shared weight vector**:

```python
# GroupedLinear
weight: (1, 64)  # A SINGLE vector instead of a matrix
# Parameters: 64 (no bias)
```

### Step-by-step operation

```python
def forward(self, input):
    # input: (N, in_features) = (N, 64)
    # weight: (1, in_features) = (1, 64)
    
    # Step 1: Element-wise multiplication
    temp = input * weight  # (N, in_features)
    
    # Step 2: Reshape into groups
    temp = temp.reshape(N, groups, in_features/groups)
    # temp: (N, groups, in_features/groups)
    
    # Step 3: Sum per group
    output = temp.sum(dim=-1)  # (N, groups) = (N, out_features)
    
    return output
```

### Concrete numeric example

Take **N=1, in_features=8, groups=out_features=4** for simplicity:

```python
# Input
x = [2, 3, 1, 4, 5, 2, 3, 1]  # (8,)

# Weight (shared vector)
w = [0.5, 1.0, 0.2, 0.8, 0.3, 0.9, 0.4, 0.7]  # (8,)

# Step 1: Element-wise multiplication
temp = [2×0.5, 3×1.0, 1×0.2, 4×0.8, 5×0.3, 2×0.9, 3×0.4, 1×0.7]
     = [1.0, 3.0, 0.2, 3.2, 1.5, 1.8, 1.2, 0.7]

# Step 2: Reshape into 4 groups of 2 dimensions
temp_grouped = [
    [1.0, 3.0],     # Group 0
    [0.2, 3.2],     # Group 1
    [1.5, 1.8],     # Group 2
    [1.2, 0.7]      # Group 3
]

# Step 3: Sum per group
output = [
    1.0 + 3.0 = 4.0,    # Group 0
    0.2 + 3.2 = 3.4,    # Group 1
    1.5 + 1.8 = 3.3,    # Group 2
    1.2 + 0.7 = 1.9     # Group 3
]
# Result: [4.0, 3.4, 3.3, 1.9]
```

### Parameter comparison

| Configuration | Classic Linear    | GroupedLinear | Reduction |
| ------------- | ----------------- | ------------- | --------- |
| 64 → 8        | 64×8 = **512**    | **64**        | 8×        |
| 128 → 16      | 128×16 = **2048** | **128**       | 16×       |
| 256 → 32      | 256×32 = **8192** | **256**       | 32×       |

GroupedLinear forces the model to use the same weights across groups, applied to different portions of the input.

---

## GroupedVectorAttention: Enriched Local Attention

### Overview

`GroupedVectorAttention` is the core of PTv2. It includes several improvements over PTv1.

{% include figure.liquid path="assets/img/poinTransformerV2/groupedVectorAttention.svg" class="img-fluid rounded z-depth-1" %}

### Detailed comparison with PTv1

| Aspect                          | PTv1 (PointTransformerLayer) | PTv2 (GroupedVectorAttention)        |
| ------------------------------- | ---------------------------- | ------------------------------------ |
| **Q, K, V projections**         | Simple Linear                | Linear + **BatchNorm1d + ReLU**      |
| **Position encoding**           | Additive only                | Additive (+ optional multiplicative) |
| **Position encoding on values** | ❌ No                         | ✅ **Yes**                            |
| **Masking invalid neighbors**   | ❌ No (assumes all valid)     | ✅ **Yes**                            |
| **Weight generation**           | Standard MLP (C×C/G params)  | **GroupedLinear** (C params only)    |
| **Normalization**               | After weight encoding        | **Before and after** attention       |

### Innovation 1: Normalization of Q, K, V projections

**PTv1:**

```python
# Simple projections without normalization
self.linear_q = nn.Linear(in_planes, mid_planes)
self.linear_k = nn.Linear(in_planes, mid_planes)
self.linear_v = nn.Linear(in_planes, out_planes)

# Usage
query = self.linear_q(feat)  # (N, C)
key = self.linear_k(feat)    # (N, C)
value = self.linear_v(feat)  # (N, C)
```

**PTv2:**

```python
# Projections with normalization and activation
self.linear_q = nn.Sequential(
    nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
    PointBatchNorm(embed_channels),  # Normalization
    nn.ReLU(inplace=True)            # Activation
)
# Same for linear_k

# Usage
query = self.linear_q(feat)  # (N, C) - normalized and activated
```

**Why it matters**

Normalization of Q and K stabilizes training by avoiding extreme values in the Q-K relation.

**Impact:** Faster convergence and more stable training.

---

### Innovation 2: Positional Encoding on the Values

**PTv1:** Positional encoding is added only to the Q-K relation.

```python
# PTv1 (simplified)
relative_positions = neighbor_positions - query_position  # (N, K, 3)
encoded_positions = MLP(relative_positions)               # (N, K, out_dim)

# Applied ONLY to Q-K relation
relation_qk = (key - query) + encoded_positions
# Values are NOT modified by geometry
```

**PTv2:** The encoding is added to Q-K **and** to the values.

```python
# PTv2
pe_bias = MLP(relative_positions)  # (N, K, C)

# On the Q-K relation (as in PTv1)
relation_qk = (key - query) + pe_bias

# NEW: also on the values!
value = value + pe_bias

# (values now contain geometric information)
```

---

### Innovation 3: Masking Invalid Neighbors

**Context: Fundamental difference between PTv1 and PTv2**

#### PTv1: K-NN always guarantees K neighbors

In PTv1 neighbors are found via **K-Nearest Neighbors (K-NN)**:

```python
# PTv1 - In each PointTransformerLayer
x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)
# Returns EXACTLY K neighbors (via K-NN search)
```

PTv1 therefore does not need masking. All K neighbors are valid (though some may be far).

---

#### PTv2: Grid Pooling can yield < K neighbors

In PTv2 neighbors are determined by **Grid Pooling** (voxelization), which can produce regions with fewer than K points.

**Reminder: What is Grid Pooling?**

**Grid Pooling** partitions space into **voxels** and aggregates all points in the same voxel.

(ASCII art omitted here but same concept as original)

**Consequence:** After Grid Pooling some regions can be **sparse**.

```
Configuration: K=8 neighbors requested

Dense area:                  Sparse area (cloud border):
    ◉  ◉  ◉                      ◉
    ◉  ●  ◉                         
    ◉  ◉  ◉                          ◉
    
Point ● has 8 neighbors ✓      Point ◉ has only 2 neighbors ✗
```

**How does PTv2 handle missing neighbors?**

When performing K-NN on pooled voxels, if a voxel has fewer than K neighbors available, missing indices are marked with **-1**:

```python
# PTv2 - K-NN on voxels after Grid Pooling
reference_index = knn_query(K=8, coord_pooled, offset)
# reference_index: (M, K)

# Example for an isolated voxel
reference_index[voxel_42] = [15, 23, -1, -1, -1, -1, -1, -1]
#                            ↑───↑   ↑──────────────────────↑
#                            2 neighbors    6 invalid indices (-1)
```

**Why -1 and not fewer indices?**

To keep a uniform shape `(M, K)` compatible with matrix operations:

* All tensors keep the same shape.
* Enables efficient GPU batching.
* Padding with -1 allows explicit masking.

---

**PTv2 solution: Explicit Masking**

**Step 1: Create the mask**

```python
# reference_index contains -1 for invalid neighbors
mask = torch.sign(reference_index + 1)  # (M, K)

# Behavior of sign(x+1):
# If reference_index[i] = -1  → sign(-1+1) = sign(0) = 0  ← invalid
# If reference_index[i] ≥ 0   → sign(≥1) = 1             ← valid
```

**Step 2: Apply on attention weights**

```python
# In GroupedVectorAttention, after softmax
attention_weights = softmax(attention_scores)  # (M, K, groups)

# Apply the mask
attention_weights = attention_weights * mask.unsqueeze(-1)
# Shape: (M, K, groups) × (M, K, 1) → (M, K, groups)
```

**Visualization:**

```python
# Before masking (after softmax over K neighbors)
attention_weights[voxel_42] = [
    [0.20, 0.15, 0.10, ...],  # Neighbor 15 (valid)
    [0.18, 0.12, 0.09, ...],  # Neighbor 23 (valid)
    [0.12, 0.14, 0.11, ...],  # Padding -1 (invalid but has weights!)
    [0.11, 0.13, 0.10, ...],  # Padding -1 (invalid)
    [0.10, 0.12, 0.12, ...],  # Padding -1 (invalid)
    [0.09, 0.11, 0.13, ...],  # Padding -1 (invalid)
    [0.10, 0.12, 0.18, ...],  # Padding -1 (invalid)
    [0.10, 0.11, 0.17, ...],  # Padding -1 (invalid)
]

# After masking
mask = [1, 1, 0, 0, 0, 0, 0, 0]

attention_weights[voxel_42] = [
    [0.20, 0.15, 0.10, ...],  # Neighbor 15 ✓
    [0.18, 0.12, 0.09, ...],  # Neighbor 23 ✓
    [0, 0, 0, ...],            # Zeroed ✓
    [0, 0, 0, ...],            # Zeroed ✓
    [0, 0, 0, ...],            # Zeroed ✓
    [0, 0, 0, ...],            # Zeroed ✓
    [0, 0, 0, ...],            # Zeroed ✓
    [0, 0, 0, ...],            # Zeroed ✓
]
```

**Step 3: Aggregation**

```python
# Final aggregation (weighted sum)
output = (value_grouped * attention_weights.unsqueeze(-1)).sum(dim=1)
# Invalid neighbors (weights=0) do not contribute ✓
```

---

**Why this is crucial**

**Without masking**, padding neighbors would contribute with **arbitrary features**.

---

### Innovation 4: GroupedLinear for Attention Weights

Instead of a standard MLP `Linear(C, groups)` with C×groups parameters, PTv2 uses `GroupedLinear(C, groups)` with only C parameters.

```python
# PTv1: Standard MLP
self.linear_w = nn.Sequential(
    nn.Linear(mid_planes, mid_planes // share_planes),  # C × C/G parameters
    ...
)

# PTv2: with GroupedLinear
self.weight_encoding = nn.Sequential(
    GroupedLinear(embed_channels, groups, groups),  # Only C parameters
    ...
)
```

**Gain:** fewer parameters to generate attention weights without performance loss.

### Innovation 5: Normalization Architecture

**PTv1:** Minimal normalization

```python
# PTv1 - No normalization on projections Q, K, V
query = Linear(x)  # Not normalized
key = Linear(x)
value = Linear(x)

# Normalization only in the weight MLP
attention_scores = MLP_with_BatchNorm(relation_qk)
```

**PTv2:** Extensive normalization

```python
# PTv2 - Normalization everywhere
query = Linear(x) → BatchNorm → ReLU  # Normalized
key = Linear(x) → BatchNorm → ReLU
value = Linear(x)  # No activation (remains linear)

# Positional encoding also normalized
pe_bias = Linear(pos) → BatchNorm → ReLU → Linear

# Weight encoding also normalized
attention_scores = GroupedLinear → BatchNorm → ReLU → Linear
```

**Impact:** More stable training, faster convergence, less sensitive to hyperparameters.

---

# Block and BlockSequence: Residual Architecture

## Block: Residual Block with DropPath

The `Block` in PTv2 encapsulates `GroupedVectorAttention` in a residual structure similar to ResNet, with a key innovation: **DropPath**.

{% include figure.liquid path="assets/img/poinTransformerV2/block.svg" class="img-fluid rounded z-depth-1" %}

### Comparison with PTv1

| Aspect              | PTv1 (PointTransformerBlock)       | PTv2 (Block)                       |
| ------------------- | ---------------------------------- | ---------------------------------- |
| **Structure**       | Linear → Attention → Linear + Skip | Linear → Attention → Linear + Skip |
| **Regularization**  | Dropout only                       | **DropPath** + Dropout             |
| **Normalization**   | 3× BatchNorm                       | 3× BatchNorm (same)                |
| **Skip connection** | Simple addition                    | Addition with **DropPath**         |

### Detailed architecture

```
Input features (N, C)
    ↓
[Linear + BatchNorm1d + ReLU]  ← Pre-activation (expansion)
    ↓
[GroupedVectorAttention]  ← Local attention over K neighbors
    ↓
[BatchNorm1d + ReLU]  ← Post-attention normalization
    ↓
[Linear + BatchNorm1d]  ← Projection
    ↓
[DropPath]  ← Stochastic regularization (NEW)
    ↓
[+ Skip Connection]  ← Residual connection
    ↓
[ReLU]  ← Final activation
    ↓
Output features (N, C)
```

### DropPath: Stochastic Depth

**DropPath** (Stochastic Depth) is a regularization technique that **drops entire paths** in a residual network rather than individual neurons.

**Classic Dropout vs DropPath:**

```python
# Classic dropout (acts on features)
def dropout(x, p=0.5):
    mask = random(x.shape) > p  # Random per-element mask
    return x * mask / (1 - p)

output = x + dropout(f(x))
# Some features of f(x) are zeroed


# DropPath (acts on entire path)
def drop_path(x, p=0.1):
    if training and random() < p:
        return 0  # Entire path ignored
    return x

output = x + drop_path(f(x))
# Either all of f(x) is kept or all is ignored
```

**Practical behavior**

During training with `drop_path_rate` (typically 0.1), a block may be skipped entirely:

```python
# Without DropPath (PTv1)
feat_transformed = Linear → Attention → Linear
output = identity + feat_transformed  # Always computed

# With DropPath (PTv2)
feat_transformed = Linear → Attention → Linear

if training and random() < drop_path_rate:
    output = identity  # feat_transformed fully skipped
else:
    output = identity + feat_transformed

# At inference
output = identity + feat_transformed  # Always active
```

**Visualization on a 12-block network**

```
With drop_path_rate = 0.1

Training iteration 1:
Input → [Block1] → [Block2] → [SKIP] → [Block4] → ... → [SKIP] → [Block12]
        ✓          ✓          ✗          ✓              ✗          ✓
        (~10 active blocks)

Training iteration 2:
Input → [Block1] → [SKIP] → [Block3] → [Block4] → ... → [Block11] → [Block12]
        ✓          ✗        ✓          ✓                  ✓          ✓
        (~11 active blocks)

Inference:
Input → [Block1] → [Block2] → [Block3] → [Block4] → ... → [Block11] → [Block12]
        ✓          ✓          ✓          ✓                  ✓          ✓
        (all 12 blocks active)
```

**However, in PTv2 the `drop_path_rate` is implemented but set to 0.0. In other words it is not used.**

## BlockSequence: Reuse of K-NN

`BlockSequence` stacks several `Block` modules and introduces a major optimization: **sharing the reference_index**.

{% include figure.liquid path="assets/img/poinTransformerV2/blockSequence.svg" class="img-fluid rounded z-depth-1" %}

### Key innovation: K-NN computed once

**PTv1 problem:**

In PTv1 each `PointTransformerLayer` recomputes K nearest neighbors via K-NN:

```python
# PTv1 - In PointTransformerLayer.forward()
def forward(self, pxo):
    p, x, o = pxo
    
    # K-NN computed AT EACH LAYER
    x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)
    x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)
    # ...
```

For a block with 6 layers, K-NN is computed **6 times**.

```
Block with 6 PTv1 layers:
Layer 1: K-NN (N points, K=16 neighbors) → O(N log N)
Layer 2: K-NN (N points, K=16 neighbors) → O(N log N)
...
Total cost: 6 × O(N log N)
```

**PTv2 solution:**

In PTv2, `BlockSequence` computes K-NN **once** at the start. All `Block`s share the same `reference_index`:

```python
# PTv2 - In BlockSequence.forward()
def forward(self, points):
    coord, feat, offset = points
    
    # K-NN computed ONCE at the beginning
    reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
    # reference_index: (N, K) - indices of K neighbors per point
    
    # All blocks share reference_index
    for block in self.blocks:
        points = block(points, reference_index)  # No recalculation!
    
    return points
```

```
Block with 6 PTv2 layers:
K-NN (once): O(N log N)
Layer 1: uses reference_index → O(1) lookup
Layer 2: uses reference_index → O(1) lookup
...
Total cost: O(N log N)  ← 6× faster
```

### Why this is valid?

**Question:** Can the same neighbors be reused across layers?

**Answer:** **YES**, because in `BlockSequence` the **positions do not change**.

```python
# In Block.forward()
def forward(self, points, reference_index):
    coord, feat, offset = points
    
    # coord (positions) remain UNCHANGED across the block
    feat = self.fc1(feat)  # Only features change
    feat = self.attn(feat, coord, reference_index)  # coord fixed
    feat = self.fc3(feat)
    # ...
    
    return [coord, feat, offset]  # coord identical in output
```

3D positions (`coord`) are constant in a `BlockSequence`. Only features evolve. The K nearest neighbors remain geometrically identical.

**When must K-NN be recomputed:**

Positions change only at level transitions (downsampling/upsampling):

```python
# Encoder
points = BlockSequence(points)  # Positions fixed, K-NN shared ✓
points = GridPool(points)        # Positions change (downsampling) ✗
points = BlockSequence(points)  # New positions → new K-NN ✓

# Decoder
points = UnpoolWithSkip(points, skip)  # Positions change (upsampling) ✗
points = BlockSequence(points)         # New positions → new K-NN ✓
```

---

## GVAPatchEmbed: Initial Embedding

Before downsampling, PTv2 applies a `GVAPatchEmbed` that enriches features at full resolution.

{% include figure.liquid path="assets/img/poinTransformerV2/GVAPatchEmbed.svg" class="img-fluid rounded z-depth-1" %}

### Role

**GVAPatchEmbed** = Linear projection + BlockSequence (without downsampling)

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

# GridPool: Voxel-based Downsampling

## Overview

`GridPool` is a major PTv2 innovation. It replaces **Furthest Point Sampling (FPS)** used in PTv1 with a voxelization-based approach.

{% include figure.liquid path="assets/img/poinTransformerV2/gridPool.svg" class="img-fluid rounded z-depth-1" %}

#### Voxelization

```python
# Normalize coordinates relative to the start of each cloud
coord_normalized = coord - start[batch]  # (N, 3)

# Assign to a grid with voxels of size grid_size
cluster = voxel_grid(
    pos=coord_normalized, 
    size=grid_size,  # e.g. 0.06m
    batch=batch,
    start=0
)
# cluster: (N,) - voxel ID per point
```

**Example with grid_size=1.0:**

```python
# Points after normalization
points = [
    [0.2, 0.3, 0.1],  # Voxel (0, 0, 0)
    [0.8, 0.5, 0.2],  # Voxel (0, 0, 0)
    [1.2, 0.4, 0.3],  # Voxel (1, 0, 0)
    [1.5, 1.8, 0.1],  # Voxel (1, 1, 0)
    [0.1, 1.1, 0.9],  # Voxel (0, 1, 0)
]

# Voxel ID calculation
voxel_id = floor(coord / grid_size)

cluster = [
    0,  # (0,0,0) → unique voxel ID
    0,  # (0,0,0) → same voxel
    1,  # (1,0,0)
    2,  # (1,1,0)
    3,  # (0,1,0)
]
```

#### Step 4: Identify Unique Voxels

```python
unique, cluster_inverse, counts = torch.unique(
    cluster, 
    sorted=True, 
    return_inverse=True, 
    return_counts=True
)
```

**What does torch.unique return?**

```python
# Input cluster (example)
cluster = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3]
#          ↑─────↑  ↑──↑  ↑────────↑  ↑──────↑
#          3 pts   2 pts  4 points   3 points

unique = [0, 1, 2, 3]  # Unique voxels
# Nvoxel = 4

cluster_inverse = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3]
# Mapping: point i belongs to unique[cluster_inverse[i]]

counts = [3, 2, 4, 3]  # Number of points per voxel
```

#### Step 5: Sorting and Index Pointers

```python
# Sort points by voxel
_, sorted_indices = torch.sort(cluster_inverse)
# sorted_indices: order to group points of the same voxel together

# Create pointers for each voxel
idx_ptr = torch.cat([
    torch.zeros(1), 
    torch.cumsum(counts, dim=0)
])
# idx_ptr: (Nvoxel + 1,)
```

**Example:**

```python
# After sort
sorted_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# Points sorted by voxel

# Index pointers
counts = [3, 2, 4, 3]
idx_ptr = [0, 3, 5, 9, 12]
#          ↑  ↑  ↑  ↑  ↑
#          │  │  │  │  └─ End (12 points)
#          │  │  │  └──── Voxel 3 starts at index 9
#          │  │  └─────── Voxel 2 starts at index 5
#          │  └────────── Voxel 1 starts at index 3
#          └───────────── Voxel 0 starts at index 0
```

#### Step 6: Aggregate Coordinates (Mean)

```python
coord_pooled = segment_csr(
    coord[sorted_indices],  # Coordinates sorted by voxel
    idx_ptr, 
    reduce="mean"
)
# coord_pooled: (Nvoxel, 3)
# Mean position of all points in each voxel
```

**Example:**

```python
# Voxel 0 has 3 points:
points_voxel_0 = [[0.2, 0.3, 0.1], [0.8, 0.5, 0.2], [0.1, 0.2, 0.15]]
coord_pooled[0] = mean(points_voxel_0) = [0.37, 0.33, 0.15]

# Voxel 1 has 2 points:
points_voxel_1 = [[1.2, 0.4, 0.3], [1.4, 0.6, 0.4]]
coord_pooled[1] = mean(points_voxel_1) = [1.3, 0.5, 0.35]
```

#### Step 7: Aggregate Features (Max)

```python
feat_pooled = segment_csr(
    feat[sorted_indices],  # Features sorted by voxel
    idx_ptr,
    reduce="max"
)
# feat_pooled: (Nvoxel, out_channels)
# Channel-wise maximum per voxel
```

**Why Max instead of Mean?**

```python
# Example with 3 points in a voxel

# Mean pooling
feat_mean = (feat1 + feat2 + feat3) / 3
# Can "dilute" important features

# Max pooling (used by PTv2)
feat_max = max(feat1, feat2, feat3)
# Preserves dominant features per channel
# More robust to noise and outliers
```

#### Step 8: Reconstruct Offsets

```python
# Retrieve batch ID for each voxel
# (takes batch of first point in each voxel)
batch_pooled = batch[idx_ptr[:-1]]
# batch_pooled: (Nvoxel,)

# Convert batch → offset
offset_pooled = batch2offset(batch_pooled)
# offset_pooled: (B,)
```

#### Step 9: Return Cluster Mapping

```python
return [coord_pooled, feat_pooled, offset_pooled], cluster_inverse
```

`cluster_inverse` is **crucial** because it enables **Map Unpooling** later:

```python
# cluster_inverse: (N,) - voxel id for each original point
cluster_inverse[point_i] = voxel_id

# Example
cluster_inverse = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3]
#                  ↑─────↑  ↑──↑  ↑────────↑  ↑──────↑
#                  Points of voxel 0, 1, 2, 3
```

This mapping will be reused in `UnpoolWithSkip` for efficient unpooling.

---

### 4. Free Map Unpooling

`cluster_inverse` allows unpooling **without computation**:

```python
# PTv1: must recompute K-NN for interpolation
upsampled = knn_interpolation(low_res, high_res)  # Costly!

# PTv2: reuse cluster mapping
upsampled = feat_low_res[cluster_inverse]  # Instant lookup!
```

# UnpoolWithSkip: Map Unpooling with Skip Connections

## Overview

`UnpoolWithSkip` is the decoder counterpart to `GridPool`. It upsamples resolution and merges multi-scale information via skip connections.

{% include figure.liquid path="assets/img/poinTransformerV2/unpoolWithSkip.svg" class="img-fluid rounded z-depth-1" %}

## Problem with K-NN Interpolation (PTv1)

### PTv1 Interpolation algorithm

In PTv1, to go from M points (low res) to N points (high res), K-NN interpolation is used.

### Interpolation problems

**1. Computational cost:**

```python
# For each high-resolution point N:
#   - Compute M distances
#   - Sort to find the K nearest
#   - Compute weighted average

Complexity: O(N × M log M)

# Example: M=25k, N=100k
Ops: 100k × 25k × log(25k) ≈ 35 billion!
```

## Solution: Map Unpooling (PTv2)

### Principle: Reuse the Cluster Mapping

PTv2's idea: **store the mapping during downsampling** and **reuse it during upsampling**.

**Recall GridPool:**

```python
# GridPool returns cluster_inverse
coord_pooled, feat_pooled, offset_pooled, cluster = GridPool(points)

# cluster: (N,) - voxel id for each original point
cluster = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3]
#          └──┬──┘  └─┬─┘  └────┬────┘  └──┬──┘
#          Voxel 0  Voxel 1  Voxel 2   Voxel 3
```

**Map Unpooling:**

```python
# Upsample by direct indexing
feat_upsampled = feat_pooled[cluster]  # (N, C)

# Each point gets features of its original voxel
```

That is it. A simple lookup. Complexity **O(1)** per point. Total **O(N)**.

---

## Detailed Algorithm

### Inputs

```python
# Current low-resolution points
coord_low: (M, 3)        # Voxel positions
feat_low: (M, in_ch)     # Voxel features
offset_low: (B,)

# Skip points (high-resolution - from encoder)
coord_skip: (N, 3)       # Original positions
feat_skip: (N, skip_ch)  # Original features
offset_skip: (B,)

# Cluster mapping (from corresponding GridPool)
cluster: (N,)            # Voxel id per original point
```

### Step 1: Project low-resolution features

```python
feat_low_proj = Linear(feat_low) → BatchNorm1d → ReLU
# feat_low_proj: (M, out_ch)
```

### Step 2: Map Unpooling

```python
# Direct lookup via cluster
feat_mapped = feat_low_proj[cluster]
# feat_mapped: (N, out_ch)
```

Each point recovers exactly the features of its original voxel.

### Step 3: Project skip features

```python
feat_skip_proj = Linear(feat_skip) → BatchNorm1d → ReLU
# feat_skip_proj: (N, out_ch)
```

### Step 4: Skip fusion

```python
feat_fused = feat_mapped + feat_skip_proj
# feat_fused: (N, out_ch)
```

**Visualization:**

```
Low-res (upsampled):        Skip (high-res):
    feat_mapped                   feat_skip_proj
         ↓                              ↓
    [0.2, 0.5, 0.1, 0.8]         [0.3, 0.1, 0.6, 0.2]
         ↓                              ↓
         └──────────── + ──────────────┘
                       ↓
              [0.5, 0.6, 0.7, 1.0]
                   feat_fused
```

### Step 5: Output

```python
return [coord_skip, feat_fused, offset_skip]
# Return skip coordinates (high resolution)
# With fused features
```

---

## Encoder and Decoder: Full view

### Encoder

{% include figure.liquid path="assets/img/poinTransformerV2/encoder.svg" class="img-fluid rounded z-depth-1" %}

```python
class Encoder:
    def forward(self, points):
        # Downsampling + feature enrichment
        points_pooled, cluster = GridPool(points)
        
        # Local attention on voxels
        points_out = BlockSequence(points_pooled)
        
        return points_out, cluster
```

### Decoder

{% include figure.liquid path="assets/img/poinTransformerV2/decoder.svg" class="img-fluid rounded z-depth-1" %}

```python
class Decoder:
    def forward(self, points_low, points_skip, cluster):
        # Upsampling + skip fusion
        points_up = UnpoolWithSkip(points_low, points_skip, cluster)
        
        # Local attention on upsampled points
        points_out = BlockSequence(points_up)
        
        return points_out
```

---

## Overall Performance: PTv1 vs PTv2

### Memory

{% include figure.liquid path="assets/img/poinTransformerV2/ptv2_time_diff.png" class="img-fluid rounded z-depth-1" %}

### Accuracy

{% include figure.liquid path="assets/img/poinTransformerV2/ptv2_s3dis_miou.png" class="img-fluid rounded z-depth-1" %}

---
