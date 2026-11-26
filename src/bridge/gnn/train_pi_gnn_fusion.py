import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class PhysicsInformedGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(PhysicsInformedGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


def physics_loss_function(predictions, targets, penalty_weight=0.1):
    mse_loss = nn.MSELoss()(predictions, targets)
    physics_penalty = torch.relu(-predictions).mean()
    return mse_loss + (penalty_weight * physics_penalty)


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Flight Physics-Informed GNN (h1-h5 & Fused Pairs)")
    parser.add_argument("--data-dir", type=str, default="data/sample_graphs", help="Directory containing flight CSVs")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--save-plot", type=str, default="docs/figures/multi_flight_gnn.png", help="Output plot path")
    parser.add_argument("--save-weights", type=str, default="checkpoints/pi_gnn_fusion.pt", help="Output model checkpoint")
    args = parser.parse_args()

    # Fallback to current directory if files are placed in root
    data_dir = args.data_dir if os.path.exists(os.path.join(args.data_dir, "intra_nodes_h1.csv")) else "."

    print("Initializing Multi-Flight Physics-Informed GNN Pipeline (h1-h5 & All Fused Pairs)...")

    flight_ids = ['h1', 'h2', 'h3', 'h4', 'h5']
    fused_pairs = ['h1_h2', 'h1_h3', 'h1_h4', 'h1_h5']

    nodes_dict = {}
    edges_dict = {}
    fused_nodes_dict = {}
    fused_edges_dict = {}

    try:
        for fid in flight_ids:
            nodes_dict[fid] = pd.read_csv(os.path.join(data_dir, f'intra_nodes_{fid}.csv'), header=None)
            edges_dict[fid] = pd.read_csv(os.path.join(data_dir, f'omegaintra_edges_{fid}.csv'), header=None)
            print(f"Loaded Flight {fid.upper()}      -> Nodes: {nodes_dict[fid].shape}, Edges: {edges_dict[fid].shape}")

        for fpair in fused_pairs:
            fused_nodes_dict[fpair] = pd.read_csv(os.path.join(data_dir, f'intra_nodes_fused_{fpair}.csv'), header=None)
            fused_edges_dict[fpair] = pd.read_csv(os.path.join(data_dir, f'omegaintra_edges_fused_{fpair}.csv'), header=None)
            print(f"Loaded Fused Master ({fpair}) -> Nodes: {fused_nodes_dict[fpair].shape}, Edges: {fused_edges_dict[fpair].shape}")

    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}. Please verify all flight and fused CSV files are in '{data_dir}'.")
        return

    tensor_nodes = []
    for fid in flight_ids:
        tensor_nodes.append(torch.tensor(nodes_dict[fid].values, dtype=torch.float))
    for fpair in fused_pairs:
        tensor_nodes.append(torch.tensor(fused_nodes_dict[fpair].values, dtype=torch.float))

    x = torch.cat(tensor_nodes, dim=0)
    total_nodes = x.size(0)

    edge_indices = []
    current_offset = 0

    for fid in flight_ids:
        num_nodes_in_flight = nodes_dict[fid].shape[0]
        for i in range(num_nodes_in_flight - 1):
            edge_indices.append([current_offset + i, current_offset + i + 1])
            edge_indices.append([current_offset + i + 1, current_offset + i])
        current_offset += num_nodes_in_flight

    for fpair in fused_pairs:
        num_fused_nodes = fused_nodes_dict[fpair].shape[0]
        for i in range(num_fused_nodes - 1):
            edge_indices.append([current_offset + i, current_offset + i + 1])
            edge_indices.append([current_offset + i + 1, current_offset + i])
        current_offset += num_fused_nodes

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    print(f"Unified Multi-Flight Dataset Created: {data} with {total_nodes} total nodes across {len(flight_ids)} flights and {len(fused_pairs)} fused vocabularies.")

    in_dim = x.size(1)
    model = PhysicsInformedGNN(in_channels=in_dim, hidden_channels=16, out_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    target_y = torch.randn(total_nodes, 8)
    epoch_losses = []

    print("\nStarting Training Across All Flights (h1-h5) and All Fused Masters...")
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = physics_loss_function(out, target_y)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Combined Loss: {loss.item():.4f}")

    os.makedirs(os.path.dirname(args.save_weights), exist_ok=True)
    torch.save(model.state_dict(), args.save_weights)
    print(f"\nTraining completed! Weights saved to {args.save_weights}")

    print("Generating multi-flight comparison diagnostic plots...")
    model.eval()
    with torch.no_grad():
        final_embeddings = model(data.x, data.edge_index).numpy()

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, args.epochs + 1), epoch_losses, 'b-', linewidth=2)
    plt.title('Multi-Flight (h1-h5) + Fused Loss', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Physics-Informed Loss')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 3, 2)
    nx_graph = nx.Graph()
    for src, dst in edge_index.t().tolist():
        nx_graph.add_edge(src, dst)

    node_colors = []
    colors_map = ['red', 'orange', 'purple', 'brown', 'cyan', 'green', 'lime', 'olive', 'magenta']

    for idx, fid in enumerate(flight_ids):
        n_count = nodes_dict[fid].shape[0]
        node_colors.extend([colors_map[idx]] * n_count)

    for idx, fpair in enumerate(fused_pairs):
        n_count = fused_nodes_dict[fpair].shape[0]
        node_colors.extend([colors_map[len(flight_ids) + idx]] * n_count)

    pos = nx.spring_layout(nx_graph, seed=42)
    nx.draw_networkx_nodes(nx_graph, pos, node_size=25, node_color=node_colors, alpha=0.85)
    nx.draw_networkx_edges(nx_graph, pos, alpha=0.15, width=0.5, edge_color='gray')
    plt.title('Topologies (h1-h5 & Fused Pairs)', fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    current_idx = 0

    for idx, fid in enumerate(flight_ids):
        n_count = nodes_dict[fid].shape[0]
        emb_slice = final_embeddings[current_idx:current_idx + n_count, :]
        plt.scatter(emb_slice[:, 0], emb_slice[:, 1], c=colors_map[idx], label=f'Flight {fid.upper()}', alpha=0.5, s=25)
        current_idx += n_count

    for idx, fpair in enumerate(fused_pairs):
        n_count = fused_nodes_dict[fpair].shape[0]
        emb_slice = final_embeddings[current_idx:current_idx + n_count, :]
        color_idx = len(flight_ids) + idx
        plt.scatter(emb_slice[:, 0], emb_slice[:, 1], c=colors_map[color_idx], label=f'Fused {fpair.upper()}', alpha=0.9, s=50, marker='^', edgecolors='k')
        current_idx += n_count

    plt.title('Latent Maneuver Embedding Space', fontweight='bold')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
    plt.savefig(args.save_plot, dpi=300, bbox_inches='tight')
    print(f"Saved diagnostic plot to {args.save_plot}")
    plt.show()


if __name__ == "__main__":
    main()
