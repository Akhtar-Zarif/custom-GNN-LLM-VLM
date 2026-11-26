import argparse
import os
import socket
import json
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


class MATLABGraphProjector(nn.Module):
    def __init__(self, input_dim=4, llm_dim=3584, num_tokens=2):
        super().__init__()
        self.num_tokens = num_tokens
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, num_tokens * llm_dim)
        )

    def forward(self, graph_features):
        z_g = torch.mean(graph_features, dim=0, keepdim=True)
        z_p = self.mlp(z_g)
        return z_p.view(1, self.num_tokens, -1)


def main():
    parser = argparse.ArgumentParser(description="Live UDP OAK-1 GNN-LLM Telemetry Server")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="UDP bind IP")
    parser.add_argument("--port", type=int, default=5005, help="UDP bind port")
    parser.add_argument("--data-dir", type=str, default="data/sample_graphs", help="Directory containing fallback graph CSVs")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face LLM ID")
    args = parser.parse_args()

    print("=" * 60)
    print("OAK-1 LIVE REAL-TIME GNN-LLM TELEMETRY SERVER")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Compute Device: {device}")

    print(f"Loading backbone model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float16, device_map=device
    )

    for param in llm.parameters():
        param.requires_grad = False

    llm_hidden_dim = llm.config.hidden_size
    projector = MATLABGraphProjector(input_dim=4, llm_dim=llm_hidden_dim, num_tokens=2).to(device, dtype=torch.float16)
    projector.eval()
    print("GNN Projector Bridge initialized successfully.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.ip, args.port))
    print(f"\nListening for live OAK-1 UDP telemetry stream on {args.ip}:{args.port}...")
    print("Waiting for incoming graph packet triggers (press Ctrl+C to exit)...\n")

    default_csv = os.path.join(args.data_dir, "intra_nodes_fused_h1_h2.csv")
    if not os.path.exists(default_csv):
        default_csv = "intra_nodes_fused_h1_h2.csv"

    try:
        while True:
            data_bytes, addr = sock.recvfrom(65535)
            print(f"Received telemetry packet from {addr} ({len(data_bytes)} bytes)")

            try:
                packet = json.loads(data_bytes.decode('utf-8'))
                csv_path = packet.get("node_file", default_csv)

                if os.path.exists(csv_path):
                    node_df = pd.read_csv(csv_path, header=None)
                    graph_tensor = torch.tensor(node_df.values, dtype=torch.float16, device=device)
                elif all(k in packet for k in ("pos_x", "pos_y", "vx", "vy")):
                    vals = [[float(packet["pos_x"]), float(packet["pos_y"]), float(packet["vx"]), float(packet["vy"])]]
                    graph_tensor = torch.tensor(vals, dtype=torch.float16, device=device)
                else:
                    print(f"Warning: File {csv_path} not found. Falling back to default fused master.")
                    node_df = pd.read_csv(default_csv, header=None)
                    graph_tensor = torch.tensor(node_df.values, dtype=torch.float16, device=device)

            except Exception as parse_err:
                print(f"Packet parse fallback triggered: {parse_err}")
                node_df = pd.read_csv(default_csv, header=None)
                graph_tensor = torch.tensor(node_df.values, dtype=torch.float16, device=device)

            print("Projecting live graph topology into soft LLM tokens...")
            graph_tokens = projector(graph_tensor)

            prompt = (
                "You are an autonomous UAV flight diagnostic core operating in real-time. "
                "Analyze the spatial-temporal graph topology and hierarchical cluster tokens "
                "extracted from the OAK-1 exocentric vision framework. "
                "Provide an immediate, concise operational status and structural stability assessment:"
            )

            messages = [
                {"role": "system", "content": "You are a specialized aerospace AI assistant for live UAV telemetry monitoring."},
                {"role": "user", "content": prompt}
            ]

            text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(text_prompt, return_tensors="pt").input_ids.to(device)
            text_embeds = llm.get_input_embeddings()(prompt_ids)

            combined_embeds = torch.cat([graph_tokens, text_embeds], dim=1)

            print("Generating real-time natural language telemetry report...")
            output_ids = llm.generate(
                inputs_embeds=combined_embeds,
                max_new_tokens=400,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print("\n" + "=" * 50)
            print("--- LIVE GNN-LLM TELEMETRY REPORT ---")
            print("=" * 50)
            print(response)
            print("=" * 50 + "\n")

    except KeyboardInterrupt:
        print("\nLive UDP server stopped by user.")


if __name__ == "__main__":
    main()
