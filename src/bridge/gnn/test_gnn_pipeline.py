import argparse
import socket
import json
import numpy as np
from collections import deque


def main():
    parser = argparse.ArgumentParser(description="Real-Time Sliding-Window Spatial GNN Telemetry Receiver")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="UDP bind IP")
    parser.add_argument("--port", type=int, default=5005, help="UDP bind port")
    parser.add_argument("--window-size", type=int, default=5, help="Temporal sliding window size")
    parser.add_argument("--adj-thresh", type=float, default=100.0, help="Spatial adjacency distance threshold")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.ip, args.port))

    print(f"[GNN Pipeline Test] Listening for OAK-1 UDP stream on {args.ip}:{args.port}...")
    node_buffer = deque(maxlen=args.window_size)

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            packet = json.loads(data.decode('utf-8'))

            node_id = packet["node_id"]
            pos_x = packet["pos_x"]
            pos_y = packet["pos_y"]
            vx = packet["vx"]
            vy = packet["vy"]
            conf = packet["confidence"]

            node_buffer.append([pos_x, pos_y, vx, vy, conf])

            if len(node_buffer) >= args.window_size:
                nodes = np.array(node_buffer)
                num_nodes = len(nodes)

                edges = np.zeros((num_nodes, num_nodes))
                omega_intra = np.zeros((num_nodes, num_nodes))

                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:
                            dist = np.linalg.norm(nodes[i, :2] - nodes[j, :2])
                            if dist < args.adj_thresh:
                                edges[i, j] = 1
                                omega_intra[i, j] = 1.0 / (dist + 1e-5)

                gnn_score = float(np.sum(omega_intra) * 0.1)
                print(f"[Frame {node_id:3d}] Window Nodes: {num_nodes} | Edges Active: {int(np.sum(edges))} | GNN Score: {gnn_score:.4f}")

    except KeyboardInterrupt:
        print("\n[GNN Pipeline Test] Stopped by user. Closing socket.")
        sock.close()


if __name__ == "__main__":
    main()
