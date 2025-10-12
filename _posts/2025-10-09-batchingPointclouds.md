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
# Other operator that supports the concatenation of pointclouds : nn.Linear

## Definition

The `torch.nn.Linear` layer applies an affine linear transformation:

$$y = xA^T + b$$

Where:
- $$x \in \mathbb{R}^{n \times d_{in}}$$ is the input matrix (n samples, $$d_{in}$$ input features)
- $$A \in \mathbb{R}^{d_{out} \times d_{in}}$$ is the weight matrix
- $$b \in \mathbb{R}^{d_{out}}$$ is the bias vector
- $$y \in \mathbb{R}^{n \times d_{out}}$$ is the output matrix

The bias $$b$$ is broadcast and added to each row of $$xA^T$$.

## Application to Concatenated Point Clouds

### Setup

Consider two point clouds with different numbers of points:
- Point cloud 1: $N_1 = 2$ points
- Point cloud 2: $N_2 = 3$ points  
- Each point has 3 coordinates (x, y, z)

**Point Cloud 1:**
$$X_1 = \begin{bmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

**Point Cloud 2:**
$$X_2 = \begin{bmatrix} x_{31} & x_{32} & x_{33} \\ x_{41} & x_{42} & x_{43} \\ x_{51} & x_{52} & x_{53} \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

### Concatenation

Concatenate along the point dimension:

$$X = \begin{bmatrix} X_1 \\ X_2 \end{bmatrix} = \begin{bmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \\ x_{31} & x_{32} & x_{33} \\ x_{41} & x_{42} & x_{43} \\ x_{51} & x_{52} & x_{53} \end{bmatrix} \in \mathbb{R}^{5 \times 3}$$

### Linear Transformation

Apply `nn.Linear(in_features=3, out_features=2)`:

**Weight matrix:**
$$A = \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

**Bias vector:**
$$b = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} \in \mathbb{R}^{2}$$

**Transpose of weight matrix:**
$$A^T = \begin{bmatrix} w_{11} & w_{21} \\ w_{12} & w_{22} \\ w_{13} & w_{23} \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

### Matrix Multiplication: $Y = XA^T + b$

$$Y = \begin{bmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \\ x_{31} & x_{32} & x_{33} \\ x_{41} & x_{42} & x_{43} \\ x_{51} & x_{52} & x_{53} \end{bmatrix} \begin{bmatrix} w_{11} & w_{21} \\ w_{12} & w_{22} \\ w_{13} & w_{23} \end{bmatrix} + \begin{bmatrix} b_1 & b_2 \end{bmatrix}$$

### Row-by-Row Computation

Each output row is computed independently:

$$y_1 = \begin{bmatrix} x_{11}w_{11} + x_{12}w_{12} + x_{13}w_{13} + b_1 & x_{11}w_{21} + x_{12}w_{22} + x_{13}w_{23} + b_2 \end{bmatrix}$$

$$y_2 = \begin{bmatrix} x_{21}w_{11} + x_{22}w_{12} + x_{23}w_{13} + b_1 & x_{21}w_{21} + x_{22}w_{22} + x_{23}w_{23} + b_2 \end{bmatrix}$$

$$y_3 = \begin{bmatrix} x_{31}w_{11} + x_{32}w_{12} + x_{33}w_{13} + b_1 & x_{31}w_{21} + x_{32}w_{22} + x_{33}w_{23} + b_2 \end{bmatrix}$$

$$y_4 = \begin{bmatrix} x_{41}w_{11} + x_{42}w_{12} + x_{43}w_{13} + b_1 & x_{41}w_{21} + x_{42}w_{22} + x_{43}w_{23} + b_2 \end{bmatrix}$$

$$y_5 = \begin{bmatrix} x_{51}w_{11} + x_{52}w_{12} + x_{53}w_{13} + b_1 & x_{51}w_{21} + x_{52}w_{22} + x_{53}w_{23} + b_2 \end{bmatrix}$$

## Key Property: No Mixing Between Points

Each output row $y_i$ depends **only** on its corresponding input row $$x_i$$:

$$y_i = x_i A^T + b, \quad i = 1, 2, \ldots, 5$$

Therefore:
- Rows 1-2 (from PC1) are transformed independently
- Rows 3-5 (from PC2) are transformed independently
- **No information is mixed between different points or different point clouds**


