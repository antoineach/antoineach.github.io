```mermaid
---
config:
  layout: elk
---
graph LR
    subgraph InputPXO[" "]
        p["p"]
        x["x"]
        o["o"]
        ref["reference_index"]
    end
    
    x -->|N x C| FC1["Linear<br>+ BatchNorm1d + ReLU"]
    
    FC1 -->|N x C| GVA["GroupedVectorAttention"]
    
    p -->|N x 3| GVA
    ref -->|N x K| GVA
    
    GVA -->|N x C| Norm2["BatchNorm1d + ReLU"]
    Norm2 -->|N x C| FC3["Linear<br>+ BatchNorm1d"]
    
    FC3 -->|N x C| DropPath["DropPath"]
    DropPath -->|N x C| Add["+"]
    x -.skip N x C.-> Add
    
    Add -->|N x C| ReLU["ReLU"]
    
    subgraph OutputPXO[" "]
        pout["p"]
        xout["x"]
        oout_t["o"]
    end
    
    p -->|N x 3| pout
    ReLU -->|N x C| xout

    o --> |B| oout_t
    
    style p fill:#f3f4f6,stroke:#6b7280,stroke-width:2px
    style x fill:#f3f4f6,stroke:#6b7280,stroke-width:2px
    style o fill:#f3f4f6,stroke:#6b7280,stroke-width:2px
    style ref fill:#f3f4f6,stroke:#6b7280,stroke-width:2px
    
    style FC1 fill:#dbeafe,stroke:#3b82f6,stroke-width:2px
    
    style GVA fill:#fce7f3,stroke:#db2777,stroke-width:3px
    
    style Norm2 fill:#dbeafe,stroke:#3b82f6,stroke-width:2px
    style FC3 fill:#dbeafe,stroke:#3b82f6,stroke-width:2px
    style DropPath fill:#dbeafe,stroke:#3b82f6,stroke-width:2px
    
    style Add fill:#fed7aa,stroke:#ea580c,stroke-width:3px
    style ReLU fill:#dbeafe,stroke:#3b82f6,stroke-width:2px
    
    style pout fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    style xout fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    style oout_t fill:#dcfce7,stroke:#16a34a,stroke-width:2px
```
