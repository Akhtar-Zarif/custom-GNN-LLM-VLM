import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .gnn_llm_projector import SpatialGraphProjector


class GNNLLMBridge:
    def __init__(self, gnn_model, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map=self.device
        )

        for param in self.llm.parameters():
            param.requires_grad = False

        self.gnn = gnn_model.to(self.device)
        llm_hidden_dim = self.llm.config.hidden_size
        self.projector = SpatialGraphProjector(gnn_dim=256, llm_dim=llm_hidden_dim).to(
            self.device, dtype=torch.float16
        )
        self.projector.eval()

    def generate_response(self, intra_nodes, omegaintra_edges, user_prompt):
        with torch.no_grad():
            intra_nodes = intra_nodes.to(self.device)
            omegaintra_edges = omegaintra_edges.to(self.device)

            z_g = self.gnn.extract_graph_embedding(intra_nodes, omegaintra_edges)
            graph_tokens = self.projector(z_g)

            prompt_ids = self.tokenizer(user_prompt, return_tensors="pt").input_ids.to(self.device)
            text_embeds = self.llm.get_input_embeddings()(prompt_ids)

            combined_embeds = torch.cat([graph_tokens, text_embeds], dim=1)

            output_ids = self.llm.generate(
                inputs_embeds=combined_embeds,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
