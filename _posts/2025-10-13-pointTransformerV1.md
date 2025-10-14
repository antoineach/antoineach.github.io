---
layout: post
title: "Point Transformer v1: Architecture and Implementation Details"
date: 2025-10-13
description: Detailed analysis of the Point Transformer v1 architecture for point-cloud segmentation and classification
tags: deep-learning point-cloud transformer architecture
categories: computer-vision
---

# Point Transformer v1: Architecture and Implementation Details

## Introduction

**Point Transformer v1** is a model for segmentation and classification of 3D point clouds that adapts the Transformer mechanism to unstructured 3D data while respecting point-cloud specific constraints. Published in 2021, it adapts attention to local neighborhoods and the irregular nature of point clouds.

The model follows a **U-Net**-like architecture composed of three main layer types:

- **PointTransformerLayer**: local attention over the K nearest neighbors  
- **TransitionDown**: spatial downsampling using Furthest Point Sampling (FPS)  
- **TransitionUp**: upsampling with skip connections

---

## Overall Architecture

The network follows an encoder–decoder (U-Net) design:

{% include figure.liquid path="assets/img/pointTransformerv1/pointTransformerV1_architecture.svg" class="img-fluid rounded z-depth-1" %}

**Key features:**

- **Local attention**: attention is computed only over K nearest neighbors (typically K = 16) rather than globally.  
- **Permutation invariance**: the architecture respects the lack of natural ordering in point clouds.  
- **Skip connections**: U-Net style skip connections preserve spatial details.

---

### 🧱 Input Reminder — Batched Point Clouds

Before diving into PointTransformer internals, recall that we handle **batches of point clouds** by concatenating them into a single tensor:

$$
X \in \mathbb{R}^{N \times C}, \quad \text{where } N = N_1 + N_2 + \dots + N_B
$$

and we keep **offsets** to delimit each cloud’s boundaries:
$$
\text{offsets} = [N_1,, N_1{+}N_2,, \dots,, N_1{+}N_2{+}\dots{+}N_B]
$$

Each row of (X) corresponds to one 3D point and its features —
so linear layers act point-wise, without mixing points from different objects.

For a detailed explanation of this batching strategy, see
👉 [Batching Point Clouds]({{ '/blog/2025/batchingPointclouds/' | relative_url }}).

---
## PointTransformerBlock: Residual Block

`PointTransformerBlock` wraps the `PointTransformerLayer` inside a residual block (ResNet-style).

{% include figure.liquid path="assets/img/pointTransformerv1/pointTransformerBlock.svg" class="img-fluid rounded z-depth-1" %}


Residual connections improve gradient flow, help learn residual mappings, and preserve initial information.

---

## PointTransformerLayer: Vectorial Local Attention

### Overview

The `PointTransformerLayer` implements a **local vector attention** mechanism inspired by Transformers, but adapted to point clouds.

{% include figure.liquid path="assets/img/pointTransformerv1/pointTransformerLayer.svg" class="img-fluid rounded z-depth-1" %}



### Why use Q - K instead of Q·Kᵀ?

The batching constraint is central here. In standard Transformers you compute:

```python
attention_scores = Q @ K.T  # shape (N, N) -> global attention
```

But with concatenated point clouds:

```
points = [ pc1_points | pc2_points | pc3_points ]
          ←    N_1   → ←    N_2   → ←    N_3   →
```

A full $$N \times N$$ attention matrix would include cross-cloud scores (e.g. between pc1 and pc2), which is **invalid**.

Point Transformer avoids this by:

1. **Local attention only**: compute attention over the K nearest neighbors within the same cloud.
2. **Neighbor search respecting offsets**: `query_and_group` or neighbor routines use offsets to restrict neighbor search to the same cloud.
3. **Using Q − K (relative vector) rather than a global dot product**:

```python
# For each query point, consider its K neighbors (guaranteed same cloud)
attention_input = key_neighbors - query_expanded  # shape (N, K, out_dim)
# A vector difference rather than a scalar product
```

This vector difference captures relative relationships without producing a full N×N matrix and without creating invalid cross-cloud attention.

### Position encoding

Positions are explicitly encoded and added to the attention input:

```python
relative_positions = neighbor_positions - query_position  # (N, K, 3)
encoded_positions = MLP(relative_positions)              # (N, K, out_dim)
attention_input = (Q - K) + encoded_positions
```


### Vectorial attention with groups

Instead of a single scalar weight per neighbor, Point Transformer produces **`num_groups` weights per neighbor**. Let's understand why and how this works.


#### Visual Diagram

Here's what happens for **one point** with **K=3 neighbors** and **num_groups=4, out_dim=16**:

```
Each neighbor's value vector (16 dims):
┌─────┬─────┬─────┬─────┐
│ G0  │ G1  │ G2  │ G3  │  ← 4 groups of 4 dimensions
│ [4] │ [4] │ [4] │ [4] │
└─────┴─────┴─────┴─────┘

Attention weights for each neighbor (4 weights):
Neighbor 1: [w₁⁽⁰⁾, w₁⁽¹⁾, w₁⁽²⁾, w₁⁽³⁾]
Neighbor 2: [w₂⁽⁰⁾, w₂⁽¹⁾, w₂⁽²⁾, w₂⁽³⁾]
Neighbor 3: [w₃⁽⁰⁾, w₃⁽¹⁾, w₃⁽²⁾, w₃⁽³⁾]

Weighted multiplication:
            ┌─────────────────────────────────┐
Neighbor 1: │w₁⁽⁰⁾·G0│w₁⁽¹⁾·G1│w₁⁽²⁾·G2│w₁⁽³⁾·G3│
            ├─────────────────────────────────┤
Neighbor 2: │w₂⁽⁰⁾·G0│w₂⁽¹⁾·G1│w₂⁽²⁾·G2│w₂⁽³⁾·G3│
            ├─────────────────────────────────┤
Neighbor 3: │w₃⁽⁰⁾·G0│w₃⁽¹⁾·G1│w₃⁽²⁾·G2│w₃⁽³⁾·G3│
            └─────────────────────────────────┘
                        ↓ sum over neighbors
            ┌─────────────────────────────────┐
Output:     │  G0   │  G1   │  G2   │  G3   │  (16 dims)
            └─────────────────────────────────┘
```


The shape `(N, K, num_groups, dim_per_group)` represents:
- For each of N points
- For each of K neighbors
- We have num_groups separate feature groups
- Each group has dim_per_group dimensions

And each group gets its own attention weight, allowing fine-grained control over feature aggregation.


---



## TransitionDown: Spatial Downsampling

`TransitionDown` reduces the number of points (analogous to strided conv).

### Case 1: stride = 1 (no downsampling)

{% include figure.liquid path="assets/img/pointTransformerv1/transitionDown_stride=1.svg" class="img-fluid rounded z-depth-1" %}

Simple projection:

```
(N, in_dim) → Linear → BN → ReLU → (N, out_dim)
```

### Case 2: stride > 1 (downsampling)

{% include figure.liquid path="assets/img/pointTransformerv1/transitionDown_stride!=1.svg" class="img-fluid rounded z-depth-1" %}

Pipeline (high-level):

1. **Compute new counts**: for each cloud, new_count = old_count // stride.
2. **Furthest Point Sampling (FPS)**: choose M ≈ N/stride representative points that maximize minimal distance; ensures spatial coverage.
3. **K-NN grouping**: for each sampled point, gather its K neighbors in the original cloud (with relative positions if `use_xyz=True`). Result: `(M, K, 3 + in_dim)`.
4. **Projection + normalization**: linear on neighbor features, BatchNorm + ReLU → `(M, out_dim, K)`.
5. **MaxPooling**: aggregate K neighbors by channel-wise max → `(M, out_dim)`.

Result: reduce N points to M points (M ≈ N/stride) with locally aggregated features.

---

## TransitionUp: Upsampling with Skip Connections

`TransitionUp` increases resolution and fuses multi-scale information.

### Case 1: no skip connection (`pxo2 = None`)

{% include figure.liquid path="assets/img/pointTransformerv1/transitionUp_without_pxo.svg" class="img-fluid rounded z-depth-1" %}


### Case 2: with skip connection (`pxo2` provided)

{% include figure.liquid path="assets/img/pointTransformerv1/transitionUp_with_pxoo.svg" class="img-fluid rounded z-depth-1" %}




## References

* Point Transformer paper (ICCV 2021): [https://arxiv.org/abs/2012.09164](https://arxiv.org/abs/2012.09164)
* Official code: [https://github.com/POSTECH-CVLab/point-transformer](https://github.com/POSTECH-CVLab/point-transformer)
* See also my post on [Batching of Point Clouds](/blog/2025/batchingPointclouds/)


