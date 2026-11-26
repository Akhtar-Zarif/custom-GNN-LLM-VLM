# Custom GNN–LLM–VLM Framework for Multi-Flight Telemetry & Edge Diagnostics

Custom GNN-LLM-VLM research repository integrating **real-time edge vision & telemetry streaming (OAK-1 / YOLO / KLT)**, **multi-flight (`h1`–`h5`) and pairwise-fused (`h1_h2`–`h1_h5`) Physics-Informed Graph Neural Networks (PI-GNN)**, and **Graph-to-LLM/VLM soft-token projection** (`SpatialGraphProjector` / `MATLABGraphProjector` + `Qwen/Qwen2.5-7B-Instruct`) for autonomous UAV flight state reasoning and diagnostics.

## Repository Structure

```text
custom-GNN-LLM-VLM/
├── README.md                     # Architecture overview, setup, workflows & Hugging Face links
├── CITATION.cff                  # Citation metadata for joint doctorate research
├── LICENSE                       # MIT License
├── .gitignore                    # Ignores large checkpoints, datasets & local caches
├── requirements.txt              # Python dependencies
├── configs/
│   └── pipeline.yaml             # Unified UDP ports, flight IDs, model IDs & graph settings
├── docs/
│   ├── architecture.md           # End-to-end multimodal pipeline & soft-token projection notes
│   └── figures/                  # Diagnostic plots (multi-flight loss, topologies & embeddings)
├── src/
│   ├── __init__.py
│   ├── gnn/
│   │   ├── __init__.py
│   │   ├── train_pi_gnn_fusion.py     # Multi-flight (h1-h5) + all fused pairs (h1_h2..h1_h5) PI-GNN
│   │   ├── train_pi_gnn_fusion_v1.py  # Multi-flight (h1-h5) + primary fused master PI-GNN
│   │   └── test_gnn_pipeline.py       # Sliding-window spatial graph & omega_intra attention test
│   ├── bridge/
│   │   ├── __init__.py
│   │   ├── gnn_llm_projector.py       # SpatialGraphProjector (GNN manifold -> soft LLM tokens)
│   │   ├── gnn_llm_bridge.py          # Multimodal GNN + LLM prefix-embedding bridge
│   │   ├── run_gnn_llm.py             # Offline batch diagnostic report generator (Qwen2.5-7B)
│   │   └── gnn_server_live.py         # Real-time UDP listener + Qwen2.5-7B telemetry server
│   └── vlm/
│       ├── __init__.py
│       └── README.md                  # VLM visual-textual-graph integration roadmap & hooks
├── edge/
│   ├── __init__.py
│   ├── oak1_realtime.py               # DepthAI OAK-1 + YOLO detection & UDP telemetry streamer
│   ├── test_oak1_realtime.py          # OAK-1 + YOLO + trajectory trail & UDP HUD streamer
│   └── simulate_oak1_tracker.py       # Simulated circular UAV trajectory UDP streamer
├── data/
│   └── sample_graphs/                 # Place MATLAB-exported intra_nodes_*.csv & omegaintra_edges_*.csv here
└── checkpoints/                       # Local model weights (gitignored; host on Hugging Face)
```

## Architecture Overview

1. **Edge Perception & Telemetry Streaming (`edge/`):**
   - Captures RGB frames from a Luxonis OAK-1 camera via DepthAI (`oak1_realtime.py`, `test_oak1_realtime.py`) or simulates UAV kinematics (`simulate_oak1_tracker.py`).
   - Runs YOLO detection, computes optical-flow velocity vectors (`vx`, `vy`), and streams JSON telemetry packets over UDP.
2. **Multi-Flight Physics-Informed GNN (`src/gnn/`):**
   - Loads trajectory node features (`intra_nodes_*.csv`) and edge topologies (`omegaintra_edges_*.csv`) across base flights (`h1`–`h5`) and pairwise fused vocabularies (`h1_h2`–`h1_h5`).
   - Trains a 2-layer `GCNConv` model with a physics-informed non-negativity penalty (`train_pi_gnn_fusion.py`) to learn unified maneuver embeddings and diagnostic topology plots.
   - Supports real-time sliding-window spatial graph construction with inverse-distance edge weights ($\omega_{\text{intra}}$) (`test_gnn_pipeline.py`).
3. **Graph-to-LLM / VLM Soft-Token Bridge (`src/bridge/` & `src/vlm/`):**
   - Projects graph-level representations into soft prompt tokens (`[batch, num_tokens, llm_dim]`) via `SpatialGraphProjector` / `MATLABGraphProjector` and prepends them to tokenized prompt embeddings.
   - Generates real-time operational stability and diagnostic reports using `Qwen/Qwen2.5-7B-Instruct` (`run_gnn_llm.py`, `gnn_server_live.py`).

## Installation

```bash
git clone https://github.com/Akhtar-Zarif/custom-GNN-LLM-VLM.git
cd custom-GNN-LLM-VLM
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Start

### 1. Train Multi-Flight PI-GNN (`h1`–`h5` + Fused Pairs)
Place your `intra_nodes_*.csv` and `omegaintra_edges_*.csv` files inside `data/sample_graphs/` (or pass `--data-dir`), then run:
```bash
python src/gnn/train_pi_gnn_fusion.py --data-dir data/sample_graphs
```

### 2. Test Live Sliding-Window GNN Telemetry (Simulated or OAK-1)
In Terminal 1 (GNN UDP receiver):
```bash
python src/gnn/test_gnn_pipeline.py --port 5005
```
In Terminal 2 (Simulated tracker or OAK-1 camera):
```bash
python edge/simulate_oak1_tracker.py --port 5005
# Or with OAK-1 hardware:
python edge/test_oak1_realtime.py --model-path checkpoints/best.pt --port 5005
```

### 3. Run GNN–LLM Flight Diagnostic Generation
Offline report generation:
```bash
python src/bridge/run_gnn_llm.py --data-dir data/sample_graphs
```
Live UDP telemetry diagnostic server:
```bash
python src/bridge/gnn_server_live.py --data-dir data/sample_graphs --port 5005
```

## Hugging Face Hub Integration

Model checkpoints, multi-flight (`h1`–`h5` and fused pairs) graph CSVs, and diagnostic artifacts are hosted at:
- **Hugging Face Repository:** `https://huggingface.co/zarifbinakhtar/custom-GNN-LLM-VLM`

Example download using `huggingface_hub`:
```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='zarifbinakhtar/custom-GNN-LLM-VLM', filename='checkpoints/best.pt'))"

## Citation

If you use this codebase in your research, please cite it using the included `CITATION.cff` file.
