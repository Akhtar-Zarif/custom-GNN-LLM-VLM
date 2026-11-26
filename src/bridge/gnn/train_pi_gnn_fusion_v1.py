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


def physics_loss_function(predictions, targets):
    mse_loss = nn.MSELoss()(predictions, targets)
    physics_penalty = torch.relu(-predictions).mean()
    return mse_loss + (0.1 * physics_penalty)


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Flight PI-GNN (h1-h5 & Single Fused Master)")
    parser.add_argument("--data-dir", type=str, default="data/sample_graphs", help="Directory containing flight CSVs")
    parser.add_argument("--fused-pair", type=str, default="h1_h2", help="Primary fused master pair (e.g., h1_h2)")
    args = parser.parse_args()

    data_dir = args.data_dir if os.path.exists(os.path.join(args.data_dir, "intra_nodes_h1.csv")) else "."
    print("Initializing Multi-Flight Physics-Informed GNN Pipeline (h1 to h5 & Fused Master)...")

    flight_ids = ['h1', 'h2', 'h3', 'h4', 'h5']
    nodes_dict = {}
    edges_dict = {}

    try:
        for fid in flight_ids:
            nodes_dict[fid] = pd.read_csv(os.path.join(data_dir, f'intra_nodes_{fid}.csv'), header=None)
            edges_dict[fid] = pd.read_csv(os.path.join(data_dir, f'omegaintra_edges_{fid}.csv'), header=None)
            print(f"Loaded Flight {fid.upper()}    -> Nodes: {nodes_dict[fid].shape}, Edges: {edges_dict[fid].shape}")

        nodes_fused = pd.read_csv(os.path.join(data_dir, f'intra_nodes_fused_{args.fused_pair}.csv'), header=None)
        edges_fused = pd.read_csv(os.path.join(data_dir, f'omegaintra_edges_fused_{args.fused_pair}.csv'), header=None)
        print(f"Loaded Fused Master   -> Nodes: {nodes_fused.shape}, Edges: {edges_fused.shape}")

    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}. Please verify all flight and fused CSV files are in '{data_dir}'.")
        return

    tensor_nodes = [torch.tensor(nodes_dict[fid].values, dtype=torch.float) for fid in flight_ids]
    tensor_nodes.append(torch.tensor(nodes_fused.values, dtype=torch.float))

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

    num_fused_nodes = nodes_fused.shape[0]
    for i in range(num_fused_nodes - 1):
        edge_indices.append([current_offset + i, current_offset + i + 1])
        edge_indices.append([current_offset + i + 1, current_offset + i])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    print(f"Unified Multi-Flight Dataset Created: {data} with {total_nodes} total nodes.")

    in_dim = x.size(1)
    model = PhysicsInformedGNN(in_channels=in_dim, hidden_channels=16, out_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    target_y = torch.randn(total_nodes, 8)
    epoch_losses = []

    print("\nStarting Training Across All Flights (h1-h5) and Fused Master...")
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = physics_loss_function(out, target_y)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/100 | Combined Loss: {loss.item():.4f}")

    print("\nTraining completed successfully!")
    model.eval()
    with torch.no_grad():
        final_embeddings = model(data.x, data.edge_index).numpy()

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, 101), epoch_losses, 'b-', linewidth=2)
    plt.title('Multi-Flight (h1-h5) + Fused Loss', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Physics-Informed Loss')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 3, 2)
    nx_graph = nx.Graph()
    for src, dst in edge_index.t().tolist():
        nx_graph.add_edge(src, dst)

    node_colors = []
    colors_map = ['red', 'orange', 'purple', 'brown', 'cyan', 'green']
    for idx, fid in enumerate(flight_ids):
        n_count = nodes_dict[fid].shape[0]
        node_colors.extend([colors_map[idx]] * n_count)
    node_colors.extend([colors_map[-1]] * nodes_fused.shape[0])

    pos = nx.spring_layout(nx_graph, seed=42)
    nx.draw_networkx_nodes(nx_graph, pos, node_size=30, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(nx_graph, pos, alpha=0.2, width=0.8, edge_color='gray')
    plt.title('Topologies (h1-h5: Distinct Colors, Green: Fused)', fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    current_idx = 0
    for idx, fid in enumerate(flight_ids):
        n_count = nodes_dict[fid].shape[0]
        emb_slice = final_embeddings[current_idx:current_idx + n_count, :]
        plt.scatter(emb_slice[:, 0], emb_slice[:, 1], c=colors_map[idx], label=f'Flight {fid.upper()}', alpha=0.6, s=30)
        current_idx += n_count

    emb_fused = final_embeddings[current_idx:, :]
    plt.scatter(emb_fused[:, 0], emb_fused[:, 1], c='green', label='Fused Master', alpha=1.0, s=80, marker='^', edgecolors='k')

    plt.title('Latent Maneuver Embedding Space', fontweight='bold')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
