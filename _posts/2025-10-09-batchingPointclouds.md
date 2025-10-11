---
layout: post
title:  Batching PointClouds
date: 2025-10-09
description:  A comprehensive explanation of Batching PointClouds.
thumbnail: assets/img/batch_thumbnail.jpg
---


## ☁️ Characteristics of Point Clouds

1. **Variable size** – each point cloud contains a different number of points $$ N $$.  
2. **Unordered** – permuting the points does not change the represented object.  
3. **Irregular** – there is no fixed neighborhood structure like in images.  
4. **Continuous** – each point lives in continuous 3D space:  
   $$
   (x, y, z) \in \mathbb{R}^3
   $$

---

## ⚠️ The Variable Number of Points Problem

The fact that each point cloud has a different number of points $$ N $$ prevents **batch parallelization** like in image-based neural networks.

```python
# Classic computer vision: easy batching
images = torch.randn(batch_size=4, channels=3, height=224, width=224)
# ✅ All images have the same shape → can be stacked together

# With point clouds — impossible!
obj1 = 1523 points   # chair
obj2 = 3891 points   # table
obj3 = 892 points    # lamp

batch = torch.stack([obj1, obj2, obj3])  # ❌ Different sizes → error!
```

---

## 🧩 Common Strategies to Handle Variable Point Counts

### 1️⃣ Batch Size = 1

Process each object individually.
→ **Drawback:** training is extremely slow, and statistical relations between samples in a batch are lost.

---

### 2️⃣ **Downsampling**

Randomly sample each point cloud to reach a fixed size (e.g. 1024 points).
→ **Pros:** Simple to implement
→ **Cons:** Loss of geometric detail, especially when point counts differ greatly
(e.g. from 1k to 10k → 90% data loss).

---

### 3️⃣ **Oversampling**

Duplicate some points to reach the target size ( $$N' = N + \Delta N$$ ).
This works for architectures like **PointNet**, since each point is independently projected via a shared MLP, then aggregated with **max pooling**.

<div class="row justify-content-center">
  <div class="col-md-8 text-center">
    {% include figure.liquid path="assets/img/maxpool.png" class="img-fluid rounded z-depth-1 shadow-sm" style="max-width:80%;height:auto;" %}
    <div class="caption mt-2 text-muted">Example of PointNet using shared MLP + max pooling.</div>
  </div>
</div>

However, if we replaced max pooling with **mean pooling**, duplicates would bias the average and distort the representation.

---

### 4️⃣ **Sparse Tensor Representation (Practical Solution)**

In practice, frameworks like **Torch Scatter**  allow concatenation of all points from a batch while preserving object boundaries.

```python
# Instead of stacking → concatenate all points
p = torch.cat([obj1_points, obj2_points, obj3_points], dim=0)  # [6306, 3]  (x, y, z)
x = torch.cat([obj1_features, obj2_features, obj3_features], dim=0)  # [6306, 64]  (features)

# Track where each object ends using offsets
o = torch.tensor([1523, 5414, 6306])  # cumulative end indices
```

```
|----------obj1----------|---------------obj2--------------|---obj3---|
0                       1523                             5414        6306
                         ↑                                 ↑          ↑
                      offset[0]                       offset[1]  offset[2]
```

These **offsets** let the model know where each object starts and ends in the concatenated tensor.


<div class="row justify-content-center">
  <div class="col-md-8 text-center">
    {% include figure.liquid path="assets/img/torch_scatter.svg" class="img-fluid rounded z-depth-1 shadow-sm" style="max-width:80%;height:auto;" %}
    <div class="caption mt-2 text-muted">Example of Torch Scatter add operator.</div>
  </div>
</div>


---

## 🔗 Why concatenating B point clouds and applying a single `nn.Linear` does **not** mix objects


### Formal statement

Let `Linear(F -> D)` be a pointwise linear projection with weight matrix $$W \in \mathbb{R}^{D\times F}$$ and bias $$b\in\mathbb{R}^{D}$$. Applied to the concatenated feature tensor:

$$
Y = \text{Linear}(x) = x W^\top + \mathbf{1} b^\top \quad\text{with } x \in \mathbb{R}^{N\times F},\ Y\in\mathbb{R}^{N\times D}.
$$

Partition $$x$$ according to the original point clouds:
$$
x = \begin{bmatrix} x^{(1)} \\[4pt] x^{(2)} \\[4pt] \vdots \\[4pt] x^{(B)} \end{bmatrix},
\qquad x^{(i)}\in\mathbb{R}^{N_i\times F},\quad \sum_i N_i = N.
$$

Because matrix multiplication is linear and row-wise here, we get:
$$
Y = \begin{bmatrix} x^{(1)} W^\top + \mathbf{1} b^\top \\[4pt] x^{(2)} W^\top + \mathbf{1} b^\top \\[4pt] \vdots \\[4pt] x^{(B)} W^\top + \mathbf{1} b^\top \end{bmatrix}
= \begin{bmatrix} \text{Linear}(x^{(1)}) \\[4pt] \text{Linear}(x^{(2)}) \\[4pt] \vdots \\[4pt] \text{Linear}(x^{(B)}) \end{bmatrix}.
$$

Thus applying `Linear` to `x` (concatenated) is **mathematically identical** to applying `Linear` independently to each `x^{(i)}` and stacking the results: **no cross-object mixing** occurs at this step.



