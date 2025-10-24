```mermaid
graph LR
    IN["p:(N0,3)\nx:(N0,in_dim)\no:(B,)"] -->|p:(N0,3)\nx:(N0,in_dim)\no:(B,)| PE["GVAPatchEmbed\noutputs p:(N1,3)\nx:(N1,C_pe)\no:(B,)"]

    PE -->|p:(N1,3)\nx:(N1,C_e1)\no:(B,)| ENC1["Encoder Stage 1\noutputs p:(N1,C_e1?)\nx:(N1,C_e1)\ncluster:(M1,)"]
    ENC1 -->|p:(N2,3)\nx:(N2,C_e2)\no:(B,)| ENC2["Encoder Stage 2\noutputs cluster:(M2,)"]
    ENC2 -->|p:(N3,3)\nx:(N3,C_e3)\no:(B,)| ENC3["Encoder Stage 3\noutputs cluster:(M3,)"]
    ENC3 -->|p:(N4,3)\nx:(N4,C_e4)\no:(B,)| ENC4["Encoder Stage 4\noutputs cluster:(M4,)"]

    %% collect skip points for decoders (explicit skip arrows)
    ENC1 -->|skip p:(N1,3)\nskip x:(N1,C_e1)| SK1["skip1"]
    ENC2 -->|skip p:(N2,3)\nskip x:(N2,C_e2)| SK2["skip2"]
    ENC3 -->|skip p:(N3,3)\nskip x:(N3,C_e3)| SK3["skip3"]
    ENC4 -->|skip p:(N4,3)\nskip x:(N4,C_e4)| SK4["skip4"]

    %% Decoder chain (reverse order). Each decoder consumes (points, skip, cluster)
    ENC4 -->|p:(N4,3)\nx:(N4,C_e4)\ncluster:(M4,)| DEC4["Decoder Stage 4\ninputs (points, skip4, cluster4)\noutputs p:(N3,3)\nx:(N3,C_d3)\no:(B,)"]
    DEC4 -->|p:(N3,3)\nx:(N3,C_d3)| DEC3["Decoder Stage 3\ninputs (points, skip3, cluster3)\noutputs p:(N2,3)\nx:(N2,C_d2)\no:(B,)"]
    DEC3 -->|p:(N2,3)\nx:(N2,C_d2)| DEC2["Decoder Stage 2\ninputs (points, skip2, cluster2)\noutputs p:(N1,3)\nx:(N1,C_d1)\no:(B,)"]
    DEC2 -->|p:(N1,3)\nx:(N1,C_d1)| DEC1["Decoder Stage 1\ninputs (points, skip1, cluster1)\noutputs p:(N0,3)\nx:(N0,C_d0)\no:(B,)"]

    %% connect skip nodes into decoder inputs (visual aid)
    SK4 -->|skip| DEC4
    SK3 -->|skip| DEC3
    SK2 -->|skip| DEC2
    SK1 -->|skip| DEC1

    %% segmentation head
    DEC1 -->|p:(N0,3)\nx:(N0,C_d0)\no:(B,)| SH["SegHead\n[linear + PointBatchNorm + ReLU + linear]\ninputs x:(N0,C_d0)\noutputs logits:(N0,num_classes)"]

    SH -->|logits:(N0,num_classes)| OUT["seg_logits:(N0,num_classes)"]


```
