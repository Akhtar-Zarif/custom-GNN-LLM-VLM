import torch
import torch.nn as nn


class SpatialGraphProjector(nn.Module):
    def __init__(self, gnn_dim=256, llm_dim=4096, num_tokens=2):
        super().__init__()
        self.num_tokens = num_tokens
        self.mlp = nn.Sequential(
            nn.Linear(gnn_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, num_tokens * llm_dim),
        )

    def forward(self, z_g):
        batch_size = z_g.shape[0]
        z_p = self.mlp(z_g)
        return z_p.view(batch_size, self.num_tokens, -1)
