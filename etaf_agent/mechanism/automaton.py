from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from etaf_agent.bolt.integrate import integrate
from etaf_agent.bolt.ranking import ranking_order
from etaf_agent.quorum.backbones import backbone_scores
from etaf_agent.quorum.vote import ensemble_rank
from etaf_agent.seal.commitment import Commitment
from etaf_agent.seal.warden import FalsificationWarden
from etaf_agent.strongbox import Settings
from etaf_agent.tumbler.reasoner import GenomicReasoner, GenomicState
from etaf_agent.vestibule.batch import Batch
from etaf_agent.ward.curator import Curator


class EtafAgent(nn.Module):
    def __init__(self, d_e: int, d_g: int, mol_dim: int, settings: Settings):
        super().__init__()
        self.reasoner = GenomicReasoner(d_e, d_g, settings.genomic, seed=settings.seed)
        self.curator = Curator(mol_dim, self.reasoner.embed_dim)

    def encode(self, batch: Batch) -> GenomicState:
        return self.reasoner(batch)

    def drug_embeddings(self, mol_embed: torch.Tensor) -> torch.Tensor:
        return self.curator(mol_embed)

    def scores(
        self,
        batch: Batch,
        mol_embed: torch.Tensor,
        plausibility: np.ndarray,
        confidence: np.ndarray,
        settings: Settings,
    ) -> torch.Tensor:
        state = self.encode(batch)
        drug_embed = self.drug_embeddings(mol_embed)
        return integrate(state.embedding, drug_embed, plausibility, confidence, settings.score)


@dataclass(frozen=True)
class PipelineResult:
    agent_scores: np.ndarray
    pre_order: np.ndarray
    ensemble_order: np.ndarray
    disagreement: np.ndarray
    audit_ok: bool


def run_pipeline(
    agent: EtafAgent,
    batch: Batch,
    mol_embed: torch.Tensor,
    plausibility: np.ndarray,
    confidence: np.ndarray,
    settings: Settings,
) -> PipelineResult:
    agent.eval()
    with torch.no_grad():
        scores = agent.scores(batch, mol_embed, plausibility, confidence, settings).cpu().numpy()
    pre_order = ranking_order(scores)
    bs = backbone_scores(
        scores, settings.ensemble.error_rates, settings.ensemble.pair_corr, settings.seed + 104729
    )
    ensemble_order, flags = ensemble_rank(
        bs.scores, pre_order, settings.score.top_k, settings.ensemble.disagree_threshold
    )
    warden = FalsificationWarden(settings.warden.float_decimals, settings.warden.commit_k)
    top = ensemble_order[0, : settings.warden.commit_k]
    posterior = (
        agent.encode(batch).subtype.posterior[0].detach().cpu().numpy().astype(float).tolist()
    )
    commitment = Commitment(
        ranking=tuple(int(x) for x in top),
        subtype_posterior=tuple(posterior),
        llm_versions=bs.names,
        timestamp="1970-01-01T00:00:00Z",
        nonce=f"{settings.seed:016x}",
    )
    warden.predict(commitment)
    published = warden.commit()
    warden.query(lambda: published)
    audit = warden.verify(commitment)
    return PipelineResult(
        agent_scores=scores,
        pre_order=pre_order,
        ensemble_order=ensemble_order,
        disagreement=flags,
        audit_ok=audit.ok,
    )
