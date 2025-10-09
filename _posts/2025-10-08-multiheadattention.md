---
layout: post
title:  MultiHead Attention Visualized
date: 2025-10-08
description:  A comprehensive explanation of MultiHead Attention with dimensions.
thumbnail: assets/img/multihead.jpg
---

This post is adapted from [GeeksforGeeks](https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism/)’s article on the Multi-Head Attention Mechanism by *sanjulika_sharma*.  
It provides an intuitive understanding of how multiple attention heads work in parallel to capture different representation subspaces.

---

## 🧠 What is Multi-Head Attention?

The Multi-Head Attention mechanism allows a model to **focus on different parts of a sequence simultaneously**.  
Each head learns different contextual relationships — for example, one might focus on word order while another captures long-range dependencies.

---

## 📊 Visualization

Below is a simple diagram illustrating how queries (Q), keys (K), and values (V) interact across multiple heads.

<div class="row justify-content-center mt-4">
  <div class="col-md-8 text-center">
    {% include figure.liquid loading="eager" path="assets/img/multiheadAttention.svg" class="img-fluid rounded z-depth-1 shadow-sm" style="max-width: 80%; height: auto;" %}
    <div class="caption mt-2 text-muted">
      Multi-Head Attention — each head performs scaled dot-product attention in parallel.
    </div>
  </div>
</div>

---

## ⚙️ Implementation Example

Below is a PyTorch implementation of **Multi-Head Attention**.  
It combines several attention heads computed in parallel, each with its own query, key, and value subspace.

```python
class MultiheadAttention(nn.Module):
    def __init__(self, in_dim, D, H):
        super().__init__()
        self.in_dim = in_dim      # Input embedding size
        self.D = D                # Model embedding size
        self.H = H                # Number of attention heads

        # Compute Q, K, V for all heads at once
        self.qkv_layer = nn.Linear(in_dim, 3 * D)
        # Final projection layer
        self.linear_layer = nn.Linear(D, D)

    def forward(self, x, mask=None):
        B, N, in_dim = x.size()

        # 1️⃣ Compute concatenated Q, K, V
        qkv = self.qkv_layer(x)  # (B, N, 3*D)

        # 2️⃣ Split heads
        qkv = qkv.reshape(B, N, self.H, 3 * self.D // self.H)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, H, N, D//H)

        # 3️⃣ Scaled dot-product attention
        d_k = q.size(-1)
        scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            scaled += mask
        attention = F.softmax(scaled, dim=-1)

        # 4️⃣ Apply attention to values
        values = torch.matmul(attention, v)

        # 5️⃣ Concatenate heads
        values = values.reshape(B, N, self.D)

        # 6️⃣ Final linear projection
        return self.linear_layer(values)
```
---

## 🧩 Key Takeaway

> Multi-Head Attention enhances a model’s representational capacity by letting it attend to information from **different representation subspaces** simultaneously — leading to richer contextual understanding.

---
