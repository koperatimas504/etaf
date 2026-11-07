from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from etaf_agent.strongbox import GenomicCfg
from etaf_agent.tumbler.backbone import ScprintTower
from etaf_agent.tumbler.graphconv import ReactomeGraphConv
from etaf_agent.tumbler.subtype import SubtypeHead, SubtypeReadout
from etaf_agent.vestibule.batch import Batch


@dataclass(frozen=True)
class GenomicState:
    embedding: torch.Tensor
    pathway: torch.Tensor
    subtype: SubtypeReadout


class GenomicReasoner(nn.Module):
    def __init__(self, d_e: int, d_g: int, cfg: GenomicCfg, seed: int = 0):
        super().__init__()
        self.tower = ScprintTower(d_g, cfg.embed_dim, cfg.n_heads, cfg.n_layers, cfg.dropout)
        self.graph = ReactomeGraphConv(d_e, cfg.graph_dim, seed=seed)
        self.embed_dim = self.tower.out_dim
        self.fuse = nn.Linear(self.embed_dim + cfg.graph_dim, self.embed_dim)
        self.subtype = SubtypeHead(self.embed_dim, cfg.n_subtypes)
        self.film = nn.Linear(cfg.n_subtypes + 1, 2 * self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, batch: Batch) -> GenomicState:
        tower = self.tower.encode(batch.features)
        pathway = self.graph(batch.expression)
        fused = torch.relu(self.fuse(torch.cat([tower, pathway], dim=1)))
        readout = self.subtype(fused)
        cond = torch.cat([readout.posterior, torch.sigmoid(readout.bap1_logit).unsqueeze(1)], dim=1)
        gamma_beta = self.film(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        conditioned = self.norm((1.0 + gamma) * fused + beta)
        return GenomicState(embedding=conditioned, pathway=pathway, subtype=readout)
