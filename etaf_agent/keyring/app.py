from __future__ import annotations

from pathlib import Path

import arguably
import numpy as np
import torch

from etaf_agent.gauge.ablation import interaction_ratio, run_ablation, t5d_retention
from etaf_agent.gauge.dossier import bap1_subgroup, fda_rediscovery, rediscovery_rate
from etaf_agent.gauge.report import _full_batch, evaluate_world
from etaf_agent.mechanism.automaton import EtafAgent, run_pipeline
from etaf_agent.mechanism.checkpoint import load_checkpoint
from etaf_agent.mechanism.trainer import train as train_loop
from etaf_agent.mechanism.trainer import trained_agent
from etaf_agent.mechanism.world import World, build_agent, build_world
from etaf_agent.strongbox import Settings, load_settings

DEFAULT_CONFIG = "keyplans/main.cfg"


def _prepare(config: str, steps: int, checkpoint: str | None) -> tuple[Settings, World, EtafAgent]:
    settings = load_settings(config)
    world = build_world(settings)
    if checkpoint is not None:
        agent = build_agent(world)
        load_checkpoint(checkpoint, agent, None)
    else:
        agent = trained_agent(settings, world, steps)
    return settings, world, agent


@arguably.command
def train(*, config: str = DEFAULT_CONFIG, steps: int | None = None, out: str = "runs") -> None:
    settings = load_settings(config)
    history = train_loop(settings, steps=steps, out_dir=out)
    first = history.losses[0] if history.losses else float("nan")
    last = history.losses[-1] if history.losses else float("nan")
    print(f"steps={history.steps} loss_first={first:.4f} loss_last={last:.4f} ckpt={out}/agent.pt")


@arguably.command
def evaluate(
    *, config: str = DEFAULT_CONFIG, steps: int = 80, checkpoint: str | None = None
) -> None:
    _settings, world, agent = _prepare(config, steps, checkpoint)
    report = evaluate_world(agent, world)
    layer = report.layers
    print(f"L1_top10={layer.l1_top10:.3f}")
    print(f"L2_map10={layer.l2_map10:.3f}")
    print(f"L3_spearman={layer.l3_spearman:.3f}")
    print(f"L4_kappa={layer.l4_kappa:.3f}")
    print(f"L5_tier_acc={layer.l5_tier:.3f}")
    print(f"overall={report.overall:.3f}")
    print(f"audit_ok={report.audit_ok} disagreement_rate={report.disagreement_rate:.3f}")


@arguably.command
def ablate(*, config: str = DEFAULT_CONFIG, steps: int = 80, checkpoint: str | None = None) -> None:
    _settings, world, agent = _prepare(config, steps, checkpoint)
    table = run_ablation(agent, world)
    for name in sorted(table.rows):
        print(f"{name}={table.rows[name]:.3f}")
    print(f"t5d_retention={t5d_retention(table):.3f}")
    print(f"interaction_ratio={interaction_ratio(table):.3f}")


@arguably.command
def rank(
    *,
    config: str = DEFAULT_CONFIG,
    patient: int = 0,
    steps: int = 80,
    checkpoint: str | None = None,
) -> None:
    _settings, world, agent = _prepare(config, steps, checkpoint)
    batch = _full_batch(world)
    result = run_pipeline(
        agent, batch, world.mol_embed, world.plausibility, world.confidence, world.settings
    )
    order = result.ensemble_order[patient, : world.settings.warden.commit_k]
    names = [world.catalog[i].name for i in order]
    for pos, name in enumerate(names, start=1):
        print(f"{pos:2d}. {name}")


@arguably.command
def dossier(
    *, config: str = DEFAULT_CONFIG, steps: int = 80, checkpoint: str | None = None
) -> None:
    _settings, world, agent = _prepare(config, steps, checkpoint)
    batch = _full_batch(world)
    result = run_pipeline(
        agent, batch, world.mol_embed, world.plausibility, world.confidence, world.settings
    )
    order = result.ensemble_order
    for row in fda_rediscovery(order, world):
        print(f"{row.name:14s} median_rank={row.median_rank:.1f} tier={row.tier}")
    rate = rediscovery_rate(order, world, top=world.settings.warden.commit_k)
    sub = bap1_subgroup(order, world)
    print(f"rediscovery_rate@{world.settings.warden.commit_k}={rate:.3f}")
    print(
        f"bap1_parp mut={sub.median_rank_mut:.1f} wt={sub.median_rank_wt:.1f} "
        f"gap={sub.rank_gap:.1f} p={sub.p_value:.4f}"
    )


@arguably.command
def audit(*, config: str = DEFAULT_CONFIG, steps: int = 0, checkpoint: str | None = None) -> None:
    _settings, world, agent = _prepare(config, steps, checkpoint)
    batch = _full_batch(world)
    result = run_pipeline(
        agent, batch, world.mol_embed, world.plausibility, world.confidence, world.settings
    )
    print(f"audit_ok={result.audit_ok}")
    print(f"commit_length={world.settings.warden.commit_k}")


@arguably.command
def export(
    *,
    config: str = DEFAULT_CONFIG,
    steps: int = 80,
    out: str = "runs/scores.npy",
    checkpoint: str | None = None,
) -> None:
    _settings, world, agent = _prepare(config, steps, checkpoint)
    batch = _full_batch(world)
    result = run_pipeline(
        agent, batch, world.mol_embed, world.plausibility, world.confidence, world.settings
    )
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, result.agent_scores.astype(np.float32))
    print(f"saved scores {result.agent_scores.shape} -> {out}")


def main() -> None:
    torch.set_num_threads(1)
    arguably.run()
