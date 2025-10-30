---
layout: post
title: "From 3D Clouds to 1D Sequences — Understanding Point Cloud Serialization"
date: 2025-10-29
description: "How Morton codes (Z-order curves) transform unordered 3D point clouds into ordered sequences for modern architectures like PointTransformerV3 and PointMamba."
categories: [3D Deep Learning, Transformers, Point Clouds]
tags: [Morton code, Z-order, PointTransformerV3, PointMamba, 3D vision]
---

## 1. Why serialization matters

Traditional point cloud models like **PointTransformerV1** or **PointNet++** operate on **unordered sets** of points.

To define local neighborhoods, they rely on:
- FPS (Farthest Point Sampling),
- KNN (K-Nearest Neighbors),
- or Grid/Voxel pooling.

Those operations create small groups (patches) of points, shaped like `[N, K, C]`, where:
- `N` = number of center points,
- `K` = number of neighbors,
- `C` = number of features.

But in **PointTransformerV3** or **PointMamba**, this paradigm shifts:
> The entire point cloud becomes a single ordered **sequence** `[L, C]`.

So we need a **consistent way to order points in 3D**.  
That’s what **serialization** does.

---

## 2. The Z-order (Morton order) curve

We can’t directly sort 3D coordinates `(x, y, z)` — it’s not one-dimensional.  
But we can **map** each 3D point to a **single integer key** that roughly preserves spatial proximity.

This mapping is called the **Morton code** or **Z-order**.

### Example in 2D (for visualization)

Below is a Z-order traversal of a 4×4 grid (2 bits per axis):

{% include figure.liquid loading="eager" path="assets/img/pointTransformerV3/z_order.png" class="img-fluid rounded z-depth-1 shadow-sm" style="max-width: 80%; height: auto;" %} 

In this curve:
- Each cell `(x, y)` is assigned a **Z-shaped index**.
- Cells that are close in space get **close indices**.
- When we sort points by this index, we obtain a **spatially coherent sequence**.

---

## 3. Bit interleaving explained

Let’s take 3D integer coordinates `x, y, z` in binary.

Suppose:
- `x = 5 → 101₂`
- `y = 3 → 011₂`
- `z = 2 → 010₂`

Each number has 3 bits: `x₂x₁x₀`, `y₂y₁y₀`, `z₂z₁z₀`.

The **Morton encoding** takes these bits and **interleaves them**:

```

Morton(x,y,z) = z2 y2 x2  z1 y1 x1  z0 y0 x0

```

Then you interpret this as a single binary number.

| Bit group | z | y | x | Resulting value |
|------------|---|---|---|-----------------|
| 2 | 0 | 0 | 1 | (001) |
| 1 | 1 | 1 | 0 | (110) |
| 0 | 0 | 1 | 1 | (011) |

Concatenating them → `001110011₂ = 115` (decimal).  
So the point `(x=5,y=3,z=2)` gets key `115`.

Two nearby points will have binary patterns that differ in only a few low bits,  
so their Morton keys are numerically close → **locality is preserved**.


---

## 4. Handling multiple batches

In training, we often have **multiple point clouds** in the same batch.  

For a detailed explanation of this batching strategy, see
👉 [Batching Point Clouds]({{ '/blog/2025/batchingPointclouds/' | relative_url }}).

If we serialize them all, their Morton keys could **collide** (same value).

To prevent this, PointTransformerV3 reserves **different numeric intervals** for each batch.

### Example

Let’s say:
- Batch 0 → keys range `[0, 2⁴⁸ - 1]`
- Batch 1 → keys range `[2⁴⁸, 2×2⁴⁸ - 1]`
- Batch 2 → keys range `[2×2⁴⁸, 3×2⁴⁸ - 1]`

This is done by **embedding the batch ID** into the highest bits:

```

global_key = (batch_id << 48) | morton_key

```

So the final 64-bit key looks like this:

| Bits | Meaning |
|------|----------|
| [63:48] | Batch ID |
| [47:0]  | Morton-encoded position |

Now, when you sort all points in the dataset together,  
each batch occupies a **separate contiguous interval**, so there is no interference.

---

## 5. Putting it all together

### Step 1. Discretize the coordinates

To use Morton encoding, we convert floating coordinates `(x, y, z)`  
into **integer voxel coordinates**:

```python
grid_coord = floor((coord - coord.min(0)[0]) / grid_size)
````

This ensures the cloud fits into an integer cube, e.g. `[0..255]^3`.

---

### Step 2. Compute Morton keys

Using the **bit interleaving** method:

```python
def morton_encode(x, y, z, depth=16):
    key = 0
    for i in range(depth):
        bit = 1 << i
        key |= ((x & bit) << (2*i + 2)) \
             | ((y & bit) << (2*i + 1)) \
             | ((z & bit) << (2*i))
    return key
```

Each bit of `x`, `y`, and `z` is placed at positions `0,3,6,…`, `1,4,7,…`, and `2,5,8,…`.

---

### Step 3. Add batch ID and sort

```python
key = (batch_id << 48) | morton_encode(x, y, z)
order = torch.argsort(key)
points = points[order]
```

After sorting, the entire cloud becomes a **1D spatial sequence**.

---

## 6. What serialization enables

By turning 3D space into an ordered sequence,
we can now feed the point cloud to **sequence-based models** like Transformers or Mamba.

| Operation           | Old approach                | New approach                        |
| ------------------- | --------------------------- | ----------------------------------- |
| Define local region | KNN or voxel pooling        | Z-order window (index range)        |
| Input structure     | Set of patches `[N,K,C]`    | Sequence `[L,C]`                    |
| Model type          | Point-based CNN/Transformer | State-space or sequence transformer |

---

## 7. Summary 

1. Discretize coordinates
2. Interleave bits (Morton encoding)
3. Add batch ID to avoid collisions
4. Sort → obtain sequence
5. Feed into Transformer/Mamba


## References

* [OCNN: Octree-based Sparse CNNs (Wang et al., 2022)](https://github.com/microsoft/OCNN)
* [PointTransformerV3 (Zhao et al., 2024)](https://arxiv.org/abs/2408.XXXXX)
* [Morton order (Wikipedia)](https://en.wikipedia.org/wiki/Z-order_curve)
* [PointMamba (2025)](https://arxiv.org/abs/2502.XXXXX)
