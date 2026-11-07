from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from etaf_agent.mechanism.automaton import EtafAgent
from etaf_agent.mechanism.checkpoint import save_checkpoint
from etaf_agent.mechanism.objective import composite_loss
from etaf_agent.mechanism.seed import set_seed
from etaf_agent.mechanism.world import World, build_agent, build_world
from etaf_agent.strongbox import Settings
from etaf_agent.vestibule.batch import iter_batches


@dataclass
class EMA:
    decay: float
    shadow: dict[str, torch.Tensor] = field(default_factory=dict)

    def register(self, model: EtafAgent) -> None:
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: EtafAgent) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()


@dataclass(frozen=True)
class TrainHistory:
    losses: tuple[float, ...]
    steps: int


def _warmup_lambda(warmup: int) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if warmup <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup)

    return fn


def train(
    settings: Settings,
    steps: int | None = None,
    out_dir: str | Path | None = None,
    world: World | None = None,
) -> TrainHistory:
    set_seed(settings.seed)
    if world is None:
        world = build_world(settings)
    agent = build_agent(world)
    agent.train()
    opt = torch.optim.AdamW(
        agent.parameters(), lr=settings.train.lr, weight_decay=settings.train.weight_decay
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _warmup_lambda(settings.train.warmup))
    ema = EMA(settings.train.ema_decay)
    ema.register(agent)

    relevance = torch.from_numpy(world.relevance)
    confidence = torch.from_numpy(world.confidence)
    mol_embed = world.mol_embed
    plaus = world.plausibility
    budget = steps if steps is not None else settings.train.max_steps
    losses: list[float] = []
    done = 0
    while done < budget:
        for batch in iter_batches(
            world.tensors, world.cohort.site, settings.train.batch_size, settings.seed + done, True
        ):
            if done >= budget:
                break
            conf_b = confidence[batch.index].numpy()
            scores = agent.scores(batch, mol_embed, plaus, conf_b, settings)
            loss = composite_loss(
                scores, relevance[batch.index], confidence[batch.index], settings.train.loss_weights
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if settings.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), settings.train.grad_clip)
            opt.step()
            sched.step()
            ema.update(agent)
            losses.append(float(loss.detach()))
            done += 1
    if out_dir is not None:
        save_checkpoint(Path(out_dir) / "agent.pt", agent, opt, done, settings.seed)
    return TrainHistory(losses=tuple(losses), steps=done)


def trained_agent(settings: Settings, world: World, steps: int) -> EtafAgent:
    set_seed(settings.seed)
    agent = build_agent(world)
    opt = torch.optim.AdamW(agent.parameters(), lr=settings.train.lr)
    relevance = torch.from_numpy(world.relevance)
    confidence = torch.from_numpy(world.confidence)
    done = 0
    while done < steps:
        for batch in iter_batches(
            world.tensors, world.cohort.site, settings.train.batch_size, settings.seed + done, True
        ):
            if done >= steps:
                break
            conf_b = confidence[batch.index].numpy()
            scores = agent.scores(batch, world.mol_embed, world.plausibility, conf_b, settings)
            loss = composite_loss(
                scores, relevance[batch.index], confidence[batch.index], settings.train.loss_weights
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            done += 1
    return agent
