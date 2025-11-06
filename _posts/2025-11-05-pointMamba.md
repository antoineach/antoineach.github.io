---
layout: post
title: "PointMamba: A Simple State Space Model for Point Cloud Analysis"
date: 2025-11-06
description: Overview of Point-Mamba, a point cloud model using Mamba state-space blocks for efficient classification.
tags: point-cloud, deep-learning, Mamba, classification, state-space-model, 3D-vision, superpoints, MAE, PointNet
categories: computer-vision
mermaid:
  enabled: true
  zoomable: true
---
# Point-Mamba: Understanding the Architecture

{% include figure.liquid path="assets/img/pointMamba/pointMamba_architecture.png" class="img-fluid rounded z-depth-1" %}


## Introduction

**Point-Mamba** introduces a novel approach to 3D point cloud understanding by combining **space-filling curve serialization** with **Mamba (State Space Models)** instead of traditional Transformers. The architecture operates on **superpoint-level** representations rather than individual points, making it fundamentally designed for **classification tasks**, not segmentation.

> **Important Note:** Point-Mamba processes point clouds at the **patch/superpoint level** (M tokens), not at the point level (N tokens). This design makes it inherently unsuitable for dense prediction tasks like segmentation without significant architectural modifications.

---

## Two-Phase Training Strategy

Point-Mamba follows a standard two-phase approach:

1. **Pre-training Phase**: Self-supervised learning via Masked Autoencoding (MAE) on ShapeNetCore
2. **Fine-tuning Phase**: Supervised classification on downstream tasks (ModelNet40, ScanObjectNN)

---

## Core Pipeline: From Point Cloud to Superpoints

### Step 1: Superpoint Sampling (FPS + KNN)

The first critical step transforms the unstructured point cloud into a structured set of **superpoints** (also called patches or groups).

```mermaid
graph TB
    subgraph SUPERPOINT["🎯 Superpoint Creation"]
        direction TB
        
        INPUT["Input Point Cloud<br/>(B, N, 3)<br/>N = 1024 (ModelNet40)<br/>N = 2048 (ScanObjectNN)"]
        
        FPS["Farthest Point Sampling (FPS)<br/>Sample M centers<br/>(B, N, 3) → (B, M, 3)<br/>M = 64 typical"]
        
        KNN["K-Nearest Neighbors (KNN)<br/>Group k neighbors per center<br/>k = 32 typical"]
        
        NORM["Normalize Neighborhoods<br/>Subtract center position<br/>neighborhood -= center"]
        
        OUTPUT["Superpoints<br/>Neighborhoods: (B, M, k, 3)<br/>Centers: (B, M, 3)"]
        
        INPUT --> FPS
        FPS --> KNN
        KNN --> NORM
        NORM --> OUTPUT
    end
    
    classDef inputStyle fill:#e1f5e1,stroke:#4caf50,stroke-width:3px
    classDef procStyle fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef outputStyle fill:#fff9c4,stroke:#fbc02d,stroke-width:3px
    
    class INPUT inputStyle
    class FPS,KNN,NORM procStyle
    class OUTPUT outputStyle
```

**Key Point:** After this step, we work with **M superpoints** (e.g., M=64), not N points (e.g., N=1024). Each superpoint represents a local region of ~32 points.

**Why this matters:** The entire Mamba encoder processes only M tokens, not N tokens. This dramatically reduces computational cost but **prevents dense per-point predictions** needed for segmentation.

---

### Step 2: Superpoint Feature Encoding (PointNet)

Each superpoint is encoded independently using a lightweight PointNet encoder.

```mermaid
graph TB
    subgraph ENCODER["📦 PointNet Superpoint Encoder"]
        direction TB
        
        ENC_IN["Input: One Superpoint<br/>(k, 3) = (32, 3)<br/>k points in local coords"]
        
        CONV1["Conv1d: 3 → 128<br/>+ BatchNorm1d<br/>+ ReLU"]
        
        CONV2["Conv1d: 128 → 256<br/>(k, 128) → (k, 256)"]
        
        POOL1["Max Pooling<br/>(k, 256) → (1, 256)<br/>Global feature"]
        
        CONCAT["Concatenate<br/>Global (1,256) + Local (k,256)<br/>→ (k, 512)"]
        
        CONV3["Conv1d: 512 → 512<br/>+ BatchNorm1d<br/>+ ReLU"]
        
        CONV4["Conv1d: 512 → 1024"]
        
        POOL2["Max Pooling<br/>(k, 1024) → (1, 1024)<br/>Superpoint feature"]
        
        ENC_OUT["Output: Feature Vector<br/>(1024,) per superpoint"]
        
        ENC_IN --> CONV1
        CONV1 --> CONV2
        CONV2 --> POOL1
        POOL1 --> CONCAT
        CONV2 -.->|broadcast| CONCAT
        CONCAT --> CONV3
        CONV3 --> CONV4
        CONV4 --> POOL2
        POOL2 --> ENC_OUT
    end
    
    classDef inputStyle fill:#e1f5e1,stroke:#4caf50,stroke-width:2px
    classDef convStyle fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef poolStyle fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef outputStyle fill:#fff9c4,stroke:#fbc02d,stroke-width:3px
    
    class ENC_IN inputStyle
    class CONV1,CONV2,CONV3,CONV4,CONCAT convStyle
    class POOL1,POOL2 poolStyle
    class ENC_OUT outputStyle
```

**Result:** Each of the M superpoints is now represented by a feature vector of dimension C .

**Shape transformation:**
```
Input:  (B, M, k, 3) = (B, 64, 32, 3)
         ↓ PointNet applied per superpoint
Output: (B, M, C) = (B, 64, 1024)
```

---

## Phase 1: Pre-training with Masked Autoencoding (MAE)

### Architecture Overview

```mermaid
graph TB
    subgraph PRETRAIN["🎭 Point-MAE-Mamba Pre-training"]
        direction TB
        
        PT_IN["ShapeNetCore Dataset<br/>(B, N, 3)"]
        
        SUPER_PT["Create Superpoints<br/>FPS + KNN<br/>(B, M=64, k=32, 3)"]
        
        SERIAL_PT["Serialization<br/>Random choice:<br/>Hilbert OR Hilbert-trans<br/>Reorder M superpoints"]
        
        ORDER_PT["OrderScale<br/>Learnable γ, β<br/>Modulate features<br/>based on chosen order"]
        
        MASK_PT["Random/Block Masking<br/>mask_ratio = 0.6<br/>Keep only 40% visible"]
        
        ENC_POINTNET["PointNet Encoder<br/>Only on VISIBLE superpoints<br/>(B, M_vis, C)"]
        
        POS_PT["Position Encoding<br/>MLP on center coords<br/>(B, M_vis, C)"]
        
        MAMBA_ENC["Mamba Encoder<br/>N=12 blocks, C=384<br/>Process visible tokens"]
        
        VIS_FEAT["Visible Features<br/>(B, M_vis, C)"]
        
        RECONSTRUCT["Reconstruction Preparation<br/>visible + mask_token → full<br/>(B, M, C)"]
        
        MAMBA_DEC["Mamba Decoder<br/>N=4 blocks, C=384<br/>Process full sequence"]
        
        EXTRACT_MASK["Extract Masked Tokens<br/>decoder_out[mask]<br/>(B, M_mask, C)"]
        
        PRED_HEAD["Prediction Head<br/>Conv1d: C → 3·k<br/>Reconstruct coordinates"]
        
        RECON_OUT["Reconstructed Patches<br/>(B·M_mask, k, 3)"]
        
        LOSS_PT["Chamfer Distance Loss<br/>CD(reconstructed, ground_truth)<br/>Only on masked patches"]
        
        PT_IN --> SUPER_PT
        SUPER_PT --> SERIAL_PT
        SERIAL_PT --> ORDER_PT
        ORDER_PT --> MASK_PT
        MASK_PT --> ENC_POINTNET
        MASK_PT --> POS_PT
        ENC_POINTNET --> MAMBA_ENC
        POS_PT --> MAMBA_ENC
        MAMBA_ENC --> VIS_FEAT
        VIS_FEAT --> RECONSTRUCT
        RECONSTRUCT --> MAMBA_DEC
        MAMBA_DEC --> EXTRACT_MASK
        EXTRACT_MASK --> PRED_HEAD
        PRED_HEAD --> RECON_OUT
        RECON_OUT --> LOSS_PT
        SUPER_PT -.->|GT masked| LOSS_PT
    end
    
    classDef inputStyle fill:#e1f5e1,stroke:#4caf50,stroke-width:3px
    classDef prepStyle fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef maskStyle fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
    classDef encStyle fill:#ffebee,stroke:#f44336,stroke-width:2px
    classDef decStyle fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef lossStyle fill:#ffcdd2,stroke:#e57373,stroke-width:3px
    
    class PT_IN inputStyle
    class SUPER_PT,SERIAL_PT,ORDER_PT prepStyle
    class MASK_PT maskStyle
    class ENC_POINTNET,POS_PT,MAMBA_ENC,VIS_FEAT encStyle
    class RECONSTRUCT,MAMBA_DEC,EXTRACT_MASK,PRED_HEAD,RECON_OUT decStyle
    class LOSS_PT lossStyle
```

### Key Pre-training Mechanism

**Masking Strategy:**
- **Random masking**: Randomly select 60% of superpoints to mask
- **Block masking**: Mask a contiguous spatial block (selected by distance from a random center)

**What gets masked:** Entire superpoints (patches), not individual points within patches.

```
Example with M=64 superpoints, mask_ratio=0.6:

Visible: 26 superpoints → Encoded by Mamba
Masked:  38 superpoints → Replaced by learnable mask_token

Decoder reconstructs the 38 masked patches (38 × 32 = 1216 points)
```

**Training objective:**
```python
loss = ChamferDistance(
    reconstructed_patches,  # (B·M_mask, k, 3)
    ground_truth_patches    # (B·M_mask, k, 3)
)
```

---

## Phase 2: Fine-tuning for Classification

### Architecture Overview

```mermaid
graph TB
    subgraph FINETUNE["🎯 Point-Mamba Classification"]
        direction TB
        
        FT_IN["Input Point Cloud<br/>(B, N, 3)<br/>ModelNet40 or ScanObjectNN"]
        
        SUPER_FT["Create Superpoints<br/>FPS + KNN<br/>(B, M=64, k=32, 3)"]
        
        ENC_FT["PointNet Encoder<br/>All superpoints<br/>(B, M, C=384)"]
        
        subgraph DUAL_SERIAL["🔄 Dual-Order Serialization"]
            S1["Hilbert Serialization<br/>Reorder M superpoints<br/>+ OrderScale γ₁, β₁"]
            S2["Hilbert-Trans Serialization<br/>Reorder M superpoints<br/>+ OrderScale γ₂, β₂"]
            S3["Concatenate Both Orders<br/>(B, 2M, C)"]
        end
        
        POS_FT["Position Encoding<br/>MLP on center coords<br/>For both orders<br/>(B, 2M, C)"]
        
        ADD_POS["Add Features + Position<br/>x = tokens + pos<br/>(B, 2M, C)"]
        
        MAMBA_FT["Mamba Encoder<br/>N=12 blocks, C=384<br/>Process 2M tokens"]
        
        FEAT_FT["Final Features<br/>(B, 2M, C)"]
        
        subgraph POOLING["Global Pooling"]
            P1["Max Pooling<br/>x.max(1)[0]"]
            P2["Average Pooling<br/>x.mean(1)"]
            P3["Concatenate<br/>(optional)"]
        end
        
        GLOBAL["Global Feature<br/>(B, C) or (B, 2C)"]
        
        CLS_HEAD["Classification Head<br/>MLP: C → 256 → 256 → num_classes<br/>+ BatchNorm + Dropout(0.5)"]
        
        FT_OUT["Class Logits<br/>(B, num_classes)"]
        
        FT_IN --> SUPER_FT
        SUPER_FT --> ENC_FT
        ENC_FT --> S1
        ENC_FT --> S2
        S1 --> S3
        S2 --> S3
        SUPER_FT --> POS_FT
        S3 --> ADD_POS
        POS_FT --> ADD_POS
        ADD_POS --> MAMBA_FT
        MAMBA_FT --> FEAT_FT
        FEAT_FT --> P1
        FEAT_FT --> P2
        P1 --> P3
        P2 --> P3
        P3 --> GLOBAL
        GLOBAL --> CLS_HEAD
        CLS_HEAD --> FT_OUT
    end
    
    classDef inputStyle fill:#e1f5e1,stroke:#4caf50,stroke-width:3px
    classDef superStyle fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef serialStyle fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
    classDef mambaStyle fill:#ffebee,stroke:#f44336,stroke-width:2px
    classDef poolStyle fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef headStyle fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px
    classDef outputStyle fill:#c8e6c9,stroke:#4caf50,stroke-width:3px
    
    class FT_IN inputStyle
    class SUPER_FT,ENC_FT superStyle
    class S1,S2,S3,POS_FT,ADD_POS serialStyle
    class MAMBA_FT,FEAT_FT mambaStyle
    class P1,P2,P3,GLOBAL poolStyle
    class CLS_HEAD headStyle
    class FT_OUT outputStyle
```

### Dual-Order Serialization Strategy

Point-Mamba's key innovation for classification is processing superpoints in **two different spatial orders simultaneously**.

```mermaid
graph TB
    subgraph DUAL_ORDER["🔄 Dual-Order Serialization Detail"]
        direction TB
        
        TOKENS["Superpoint Tokens<br/>(B, M=64, C=384)"]
        
        CENTERS["Superpoint Centers<br/>(B, M, 3)"]
        
        subgraph FORWARD["Forward Direction: Hilbert"]
            F1["Apply Hilbert Curve<br/>serialization on centers"]
            F2["Reorder tokens<br/>tokens[hilbert_order]"]
            F3["OrderScale γ₁, β₁<br/>γ₁ · tokens + β₁<br/>Learnable modulation"]
        end
        
        subgraph BACKWARD["Backward Direction: Hilbert-Trans"]
            B1["Apply Hilbert-Trans Curve<br/>serialization on centers"]
            B2["Reorder tokens<br/>tokens[hilbert_trans_order]"]
            B3["OrderScale γ₂, β₂<br/>γ₂ · tokens + β₂<br/>Different learnable params"]
        end
        
        CONCAT["Concatenate Both<br/>(B, 2M=128, C=384)"]
        
        POS_FORWARD["Position Embed<br/>Forward (B, M, C)"]
        POS_BACKWARD["Position Embed<br/>Backward (B, M, C)"]
        
        POS_CONCAT["Concatenate Positions<br/>(B, 2M, C)"]
        
        FINAL["Final Input to Mamba<br/>tokens + positions"]
        
        TOKENS --> F1
        TOKENS --> B1
        CENTERS --> F1
        CENTERS --> B1
        
        F1 --> F2
        F2 --> F3
        F3 --> CONCAT
        
        B1 --> B2
        B2 --> B3
        B3 --> CONCAT
        
        CENTERS --> POS_FORWARD
        CENTERS --> POS_BACKWARD
        POS_FORWARD --> POS_CONCAT
        POS_BACKWARD --> POS_CONCAT
        
        CONCAT --> FINAL
        POS_CONCAT --> FINAL
    end
    
    classDef tokenStyle fill:#e1f5e1,stroke:#4caf50,stroke-width:2px
    classDef forwardStyle fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef backwardStyle fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef concatStyle fill:#f3e5f5,stroke:#9c27b0,stroke-width:3px
    
    class TOKENS,CENTERS tokenStyle
    class F1,F2,F3,POS_FORWARD forwardStyle
    class B1,B2,B3,POS_BACKWARD backwardStyle
    class CONCAT,POS_CONCAT,FINAL concatStyle
```

**Why dual-order?**
- Different space-filling curves capture different spatial patterns
- Processing the same data in two orders enriches representations
- OrderScale (γ, β) allows the model to learn order-specific feature modulations

**OrderScale mechanism:**
```python
# Learnable parameters per order
gamma_1, beta_1 = nn.Parameter(torch.ones(C)), nn.Parameter(torch.zeros(C))
gamma_2, beta_2 = nn.Parameter(torch.ones(C)), nn.Parameter(torch.zeros(C))

# Apply to forward/backward serialized tokens
tokens_forward_scaled = gamma_1 * tokens_forward + beta_1
tokens_backward_scaled = gamma_2 * tokens_backward + beta_2
```

---

### From Superpoints to Classification: The Aggregation

The critical step for classification is aggregating the 2M superpoint features into a **single global feature vector**.

```
Mamba Output: (B, 2M=128, C=384)
    ↓
Global Pooling: (B, C) or (B, 2C) if both max and avg
    ↓
Classification Head: (B, num_classes)
```

**Pooling strategies (ablation study findings):**

| Strategy | Shape | Performance |
|----------|-------|-------------|
| Max pooling only | (B, C) | Good |
| Avg pooling only | (B, C) | **Best** |
| Max + Avg concat | (B, 2C) | Good |
| CLS token | (B, C) | Worse |

> **Important finding from paper:** "Without [CLS] token and utilizing only average pooling of the final block's output yields the best results for Point-Mamba."

---

## Mamba Block: The Core Processing Unit

```mermaid
graph TB
    subgraph MAMBA_BLOCK["🐍 Mamba Block Detail"]
        direction TB
        
        MB_IN["Input<br/>(B, 2M, C)"]
        
        NORM["RMSNorm<br/>(optional)"]
        
        subgraph SSM["State Space Model (Mamba Core)"]
            direction TB
            
            PROJ_IN["Input Projection<br/>Linear: C → 2C"]
            
            CONV["Causal Conv1D<br/>kernel_size=4<br/>Temporal modeling"]
            
            SSM_CORE["Selective Scan<br/>State-space dynamics<br/>Context-dependent"]
            
            GATE["Gating Mechanism<br/>SiLU activation<br/>x · σ(gate)"]
            
            PROJ_OUT["Output Projection<br/>Linear: C → C"]
        end
        
        DROP["DropPath<br/>Stochastic depth<br/>(regularization)"]
        
        RESIDUAL["Residual Connection<br/>out = input + drop(ssm(input))"]
        
        MB_OUT["Output<br/>(B, 2M, C)"]
        
        MB_IN --> NORM
        NORM --> PROJ_IN
        PROJ_IN --> CONV
        CONV --> SSM_CORE
        SSM_CORE --> GATE
        GATE --> PROJ_OUT
        PROJ_OUT --> DROP
        DROP --> RESIDUAL
        MB_IN -.->|skip| RESIDUAL
        RESIDUAL --> MB_OUT
    end
    
    classDef inputStyle fill:#e1f5e1,stroke:#4caf50,stroke-width:2px
    classDef normStyle fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef ssmStyle fill:#ffebee,stroke:#f44336,stroke-width:2px
    classDef outputStyle fill:#c8e6c9,stroke:#4caf50,stroke-width:2px
    
    class MB_IN inputStyle
    class NORM normStyle
    class PROJ_IN,CONV,SSM_CORE,GATE,PROJ_OUT,DROP ssmStyle
    class RESIDUAL,MB_OUT outputStyle
```

**Key difference from Transformer:**
- Transformer uses **attention** (Q @ K^T): O(sequence_length²) complexity
- Mamba uses **selective state-space**: O(sequence_length) complexity
- Mamba maintains a **hidden state** that evolves along the sequence (similar to RNNs but more efficient)

## Complete Shape Transformation Flow

### Pre-training (MAE)

```
(B, N, 3)                    Input point cloud
    ↓ FPS
(B, M, 3)                    Superpoint centers
    ↓ KNN
(B, M, k, 3)                 Neighborhoods
    ↓ PointNet per superpoint
(B, M, C)                    Superpoint features
    ↓ Serialization (1 order)
(B, M, C)                    Reordered features
    ↓ OrderScale
(B, M, C)                    Modulated features
    ↓ Masking (60%)
(B, M_vis, C)                Visible features only
    ↓ + Position
(B, M_vis, C)                Input to encoder
    ↓ Mamba Encoder (12 blocks)
(B, M_vis, C)                Encoded visible
    ↓ Add mask tokens
(B, M, C)                    Full sequence
    ↓ Mamba Decoder (4 blocks)
(B, M, C)                    Decoded features
    ↓ Extract masked
(B, M_mask, C)               Masked features
    ↓ Conv1d prediction head
(B·M_mask, k, 3)             Reconstructed patches
```

### Fine-tuning (Classification)

```
(B, N, 3)                    Input point cloud
    ↓ FPS
(B, M, 3)                    Superpoint centers
    ↓ KNN
(B, M, k, 3)                 Neighborhoods
    ↓ PointNet per superpoint
(B, M, C)                    Superpoint features
    ↓ Dual serialization
(B, M, C) × 2                Forward + Backward
    ↓ OrderScale × 2
(B, M, C) × 2                Modulated × 2
    ↓ Concatenate
(B, 2M, C)                   Combined features
    ↓ + Position
(B, 2M, C)                   Input to encoder
    ↓ Mamba Encoder (12 blocks)
(B, 2M, C)                   Encoded features
    ↓ Global pooling (avg)
(B, C)                       Global feature
    ↓ Classification head
(B, num_classes)             Class logits
```

---

## Summary: Why Superpoint-Level Processing?

**Advantages:**
1. **Local structure preservation**: Each superpoint captures a meaningful local region
2. **Hierarchical representation**: Superpoints act as abstracted representations

**Limitations:**
1. **Cannot do dense prediction**: No mechanism to return to N points
2. **Information loss**: k=32 points compressed into single vector

**Point-Mamba is fundamentally a** ***classification*** **architecture, not a segmentation architecture.** Adapting it for segmentation would require significant modifications including upsampling modules, multi-scale skip connections, and point-level feature extraction—none of which are present in the provided codebase.