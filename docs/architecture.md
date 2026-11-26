# System Architecture: Custom GNN–LLM–VLM Pipeline

## 1. End-to-End Multimodal Data Flow

```text
[OAK-1 RGB Camera / KLT Simulator]
        │  (640x640 frames -> YOLO detection -> pos_x, pos_y, vx, vy)
        ▼
[UDP Telemetry Stream (JSON packets)]
        │
        ├──► [Sliding-Window Spatial Graph Builder] (omega_intra = 1 / dist)
        │
        ▼
[Multi-Flight PI-GNN (h1–h5 & Fused Pairs h1_h2–h1_h5)]
        │  (2-layer GCNConv + Physics-Informed Non-Negativity Loss)
        ▼
[SpatialGraphProjector / MATLABGraphProjector]
        │  (MLP mapping graph representation -> [1, num_tokens, llm_dim] soft tokens)
        ▼
[Frozen LLM / VLM Backbone (Qwen2.5-7B-Instruct / Multimodal VLM)]
        │  (Concatenated [soft_graph_tokens || prompt_embeddings])
        ▼
[Real-Time Flight Stability & Diagnostic Telemetry Report]
```

## 2. Physics-Informed GNN Formulation

Given concatenated node feature matrices $\mathbf{X}$ across flights `h1`–`h5` and fused pairs `h1_h2`–`h1_h5`, the two-layer `GCNConv` network computes latent node embeddings $\hat{\mathbf{Y}}$ optimized via:

$$\mathcal{L}_{\text{PI}} = \mathcal{L}_{\text{MSE}}(\hat{\mathbf{Y}}, \mathbf{Y}) + \lambda \cdot \frac{1}{N} \sum_{i} \text{ReLU}(-\hat{Y}_i)$$

where $\lambda = 0.1$ penalizes non-physical negative state predictions.

## 3. VLM Extension Roadmap

The `src/vlm/` module extends this soft-token graph prefix architecture by pairing:
1. Direct visual crops/frames from the OAK-1 stream,
2. Projected spatial-temporal GNN soft tokens, and
3. Structured mission prompts for joint visual-topological reasoning.
