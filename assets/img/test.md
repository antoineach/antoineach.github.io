```mermaid
graph LR
    IN["p[N0,3], x[N0,in_dim], o[B]"] -->|p[N0,3], x[N0,in_dim], o[B]| PE["GVAPatchEmbed<br>→ p[N1,3], x[N1,Cpe], o[B]"]

    PE -->|p[N1,3], x[N1,Ce1], o[B]| ENC1["Encoder Stage 1<br>→ p[N2,3], x[N2,Ce1], cluster[M1]"]
    ENC1 -->|p[N2,3], x[N2,Ce2], o[B]| ENC2["Encoder Stage 2<br>→ cluster[M2]"]
    ENC2 -->|p[N3,3], x[N3,Ce3], o[B]| ENC3["Encoder Stage 3<br>→ cluster[M3]"]
    ENC3 -->|p[N4,3], x[N4,Ce4], o[B]| ENC4["Encoder Stage 4<br>→ cluster[M4]"]

    %% skip connections
    ENC1 -.-> SK1["skip1<br>p[N1,3], x[N1,Ce1]"]
    ENC2 -.-> SK2["skip2<br>p[N2,3], x[N2,Ce2]"]
    ENC3 -.-> SK3["skip3<br>p[N3,3], x[N3,Ce3]"]
    ENC4 -.-> SK4["skip4<br>p[N4,3], x[N4,Ce4]"]

    %% decoders
    ENC4 -->|p[N4,3], x[N4,Ce4], cluster[M4]| DEC4["Decoder Stage 4<br>→ p[N3,3], x[N3,Cd3], o[B]"]
    DEC4 -->|p[N3,3], x[N3,Cd3]| DEC3["Decoder Stage 3<br>→ p[N2,3], x[N2,Cd2], o[B]"]
    DEC3 -->|p[N2,3], x[N2,Cd2]| DEC2["Decoder Stage 2<br>→ p[N1,3], x[N1,Cd1], o[B]"]
    DEC2 -->|p[N1,3], x[N1,Cd1]| DEC1["Decoder Stage 1<br>→ p[N0,3], x[N0,Cd0], o[B]"]

    SK4 -.-> DEC4
    SK3 -.-> DEC3
    SK2 -.-> DEC2
    SK1 -.-> DEC1

    %% seg head
    DEC1 -->|p[N0,3], x[N0,Cd0], o[B]| SH["SegHead<br>[linear + norm + relu + linear]<br>→ logits[N0,num_classes]"]
    SH -->|logits[N0,num_classes]| OUT["seg_logits[N0,num_classes]"]

```
