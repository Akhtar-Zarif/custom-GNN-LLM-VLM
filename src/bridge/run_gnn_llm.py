import argparse
import os
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
    parser = argparse.ArgumentParser(description="Offline GNN-LLM Diagnostic Report Generator")
    parser.add_argument("--data-dir", type=str, default="data/sample_graphs", help="Directory containing graph CSVs")
    parser.add_argument("--fused-pair", type=str, default="h1_h2", help="Fused pair suffix")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face LLM ID")
    args = parser.parse_args()

    data_dir = args.data_dir if os.path.exists(os.path.join(args.data_dir, f"intra_nodes_fused_{args.fused_pair}.csv")) else "."

    print("Initializing GNN-LLM Bridge...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading backbone model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float16, device_map=device
    )

    llm_hidden_dim = llm.config.hidden_size
    projector = MATLABGraphProjector(
        input_dim=4, llm_dim=llm_hidden_dim, num_tokens=2
    ).to(device, dtype=torch.float16)
    projector.eval()

    node_file = os.path.join(data_dir, f"intra_nodes_fused_{args.fused_pair}.csv")
    edge_file = os.path.join(data_dir, f"omegaintra_edges_fused_{args.fused_pair}.csv")

    if not os.path.exists(node_file) or not os.path.exists(edge_file):
        raise FileNotFoundError(f"Could not find {node_file} or {edge_file}!")

    print(f"Loading graph topology from {node_file} and {edge_file}...")
    node_df = pd.read_csv(node_file, header=None)
    graph_tensor = torch.tensor(node_df.values, dtype=torch.float16, device=device)

    print("Projecting graph topology into soft LLM tokens...")
    graph_tokens = projector(graph_tensor)

    prompt = (
        "You are an autonomous UAV flight diagnostic core. "
        "Analyze the spatial-temporal graph topology and hierarchical cluster tokens "
        "extracted from the OAK-1 exocentric vision framework. "
        "Provide a concise technical report on the drone's flight state and structural stability:"
    )

    messages = [
        {"role": "system", "content": "You are a specialized aerospace AI assistant for graph-based UAV telemetry."},
        {"role": "user", "content": prompt}
    ]

    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(text_prompt, return_tensors="pt").input_ids.to(device)
    text_embeds = llm.get_input_embeddings()(prompt_ids)

    combined_embeds = torch.cat([graph_tokens, text_embeds], dim=1)

    print("Generating natural language flight reasoning...")
    output_ids = llm.generate(
        inputs_embeds=combined_embeds,
        max_new_tokens=800,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("\n" + "=" * 50)
    print("--- GNN-LLM FLIGHT DIAGNOSTIC REPORT ---")
    print("=" * 50)
    print(response)
    print("=" * 50)


if __name__ == "__main__":
    main()
