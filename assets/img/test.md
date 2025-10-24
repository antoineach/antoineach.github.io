```mermaid
graph LR
    A[p, x, o] -->|N,B| B[GVAPatchEmbed]
    B -->|N₁,C₁| C1[Encoder Stage 1]
    C1 -->|N₂,C₂| C2[Encoder Stage 2]
    C2 -->|N₃,C₃| C3[Encoder Stage 3]
    C3 -->|N₄,C₄| C4[Encoder Stage 4]
    C4 -->|N₄,C₄| D1[Decoder Stage 4]
    D1 -->|N₃,C₃| D2[Decoder Stage 3]
    D2 -->|N₂,C₂| D3[Decoder Stage 2]
    D3 -->|N₁,C₁| D4[Decoder Stage 1]
    D4 -->|N,B| E[SegHead (linear + norm + relu + linear)]
    E -->|N,num_classes| F[x_seg_logits]
```
