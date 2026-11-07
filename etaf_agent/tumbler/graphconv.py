from __future__ import annotations

import torch
from torch import nn


def _row_normalize(adj: torch.Tensor) -> torch.Tensor:
    deg = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adj / deg


class ReactomeGraphConv(nn.Module):
    def __init__(
        self,
        in_dim: int,
        graph_dim: int,
        seed: int = 0,
        *,
        membership: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
    ):
        super().__init__()
        if membership is None or adjacency is None:
            raise NotImplementedError(
                "Reactome pathway membership and graph edges must be supplied."
            )
        if (
            membership.shape != (in_dim, graph_dim)
            or adjacency.shape != (graph_dim, graph_dim)
        ):
            raise ValueError("Pathway graph dimensions do not match the model configuration.")
        self.register_buffer("membership", membership.float())
        self.register_buffer("adj_norm", _row_normalize(adjacency.float()))
        self.lin = nn.Linear(graph_dim, graph_dim)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        ranks = expression.argsort(dim=1).argsort(dim=1).float()
        ranks = ranks / (expression.shape[1] - 1 + 1e-6)
        membership: torch.Tensor = self.get_buffer("membership")
        adj_norm: torch.Tensor = self.get_buffer("adj_norm")
        pathway = ranks @ membership
        diffused = pathway @ adj_norm
        return torch.relu(self.lin(diffused))
