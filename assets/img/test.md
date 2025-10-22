---
config:
  layout: dagre
---
flowchart TD
    Input["Input (pxo)"] --> P["positions<br>(N, 3)"]
    Input --> X["features<br>(N, in_dim)"]
    Input --> O["offsets<br>(B,)"]

    %% --- Stage 1 : QKV projections ---
    X -- linear_q --> Q["query<br>(N, out_dim)"]
    X -- linear_k --> K["key<br>(N, out_dim)"]
    X -- linear_v --> V["value<br>(N, out_dim)"]

    %% --- Stage 2 : Neighbor grouping ---
    P --> Grouping
    O --> Grouping
    K --> Grouping
    Grouping["pointops.grouping<br>(nsample=K)"] --> KNeighbors["key_neighbors<br>(N, K, out_dim)"]
    V --> GroupingV
    GroupingV["pointops.grouping<br>(nsample=K)"] --> VNeighbors["value_neighbors<br>(N, K, out_dim)"]

    %% --- Stage 3 : Relative positions ---
    P --> GroupingPos
    GroupingPos["pointops.grouping<br>(nsample=K, use_xyz=True)"] --> PNeighbors["neighbor_positions<br>(N, K, 3)"]
    PNeighbors --> RelPos["relative_positions<br>(N, K, 3)"]
    RelPos -- pos_enc MLP --> EncPos["encoded_positions<br>(N, K, out_dim)"]

    %% --- Stage 4 : Attention weight computation ---
    Q -- unsqueeze(1) --> QExp["query_expanded<br>(N, 1, out_dim)"]
    KNeighbors --> Diff["K - Q"]
    QExp --> Diff
    Diff --> QKDiff["qk_diff<br>(N, K, out_dim)"]
    EncPos --> Combine
    QKDiff --> Combine
    Combine["qk_diff + encoded_pos"] --> AttnInput["attention_input<br>(N, K, out_dim)"]
    AttnInput -- linear_w --> AttnPre["attn_pre<br>(N, K, mid_dim)"]
    AttnPre -- Softmax(dim=2) --> AttnWeights["attention_weights<br>(N, K, mid_dim)"]

    %% --- Stage 5 : Feature aggregation ---
    VNeighbors --> CombFeats["V + encoded_pos"]
    EncPos --> CombFeats
    CombFeats --> Combined["combined_features<br>(N, K, out_dim)"]
    Combined -- "project via shared planes" --> CombinedProj["(N, K, nshare, out_dim/nshare)"]
    AttnWeights -- unsqueeze(2) --> AttnExp["attention_expanded<br>(N, K, 1, mid_dim)"]
    CombinedProj --> Multiply
    AttnExp --> Multiply
    Multiply["weighted sum<br>(N, K, nshare, out_dim/nshare)"]
      --> Aggregated["sum(dim=1)<br>(N, nshare, out_dim/nshare)"]
    Aggregated -- "reshape" --> Output["Output<br>(N, out_dim)"]

    %% --- Styles ---
    style Input fill:#f3f4f6,stroke:#6b7280,stroke-width:2px
    style P fill:#e0f2fe,stroke:#2563eb,stroke-width:2px
    style X fill:#e0f2fe,stroke:#2563eb,stroke-width:2px
    style O fill:#e0f2fe,stroke:#2563eb,stroke-width:2px

    style Q fill:#dbeafe,stroke:#3b82f6,stroke-width:2px
    style K fill:#dbeafe,stroke:#3b82f6,stroke-width:2px
    style V fill:#dbeafe,stroke:#3b82f6,stroke-width:2px

    style Grouping fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style GroupingV fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style GroupingPos fill:#fef3c7,stroke:#d97706,stroke-width:2px

    style KNeighbors fill:#e0e7ff,stroke:#6366f1,stroke-width:2px
    style VNeighbors fill:#e0e7ff,stroke:#6366f1,stroke-width:2px
    style PNeighbors fill:#e0e7ff,stroke:#6366f1,stroke-width:2px
    style RelPos fill:#e0e7ff,stroke:#6366f1,stroke-width:2px
    style EncPos fill:#fce7f3,stroke:#db2777,stroke-width:2px

    style QKDiff fill:#fed7aa,stroke:#ea580c,stroke-width:2px
    style AttnInput fill:#fed7aa,stroke:#ea580c,stroke-width:2px
    style AttnPre fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style AttnWeights fill:#dcfce7,stroke:#16a34a,stroke-width:2px

    style Combined fill:#e0e7ff,stroke:#6366f1,stroke-width:2px
    style CombinedProj fill:#ddd6fe,stroke:#7c3aed,stroke-width:2px
    style Aggregated fill:#ddd6fe,stroke:#7c3aed,stroke-width:2px
    style Output fill:#dcfce7,stroke:#16a34a,stroke-width:3px
