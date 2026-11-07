from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from confection import Config


@dataclass(frozen=True)
class DataCfg:
    n_site_a: int
    n_site_b: int
    n_genes: int
    n_variants: int
    n_clinical: int
    catalog_size: int
    panel_overlap: float
    signal: float
    seed: int


@dataclass(frozen=True)
class GenomicCfg:
    embed_dim: int
    n_heads: int
    n_layers: int
    graph_dim: int
    n_subtypes: int
    dropout: float


@dataclass(frozen=True)
class CuratorCfg:
    tier_weights: tuple[float, ...]
    mol_dim: int


@dataclass(frozen=True)
class PathwayCfg:
    max_walk: int


@dataclass(frozen=True)
class ScoreCfg:
    alpha: float
    beta: float
    gamma: float
    top_k: int


@dataclass(frozen=True)
class EnsembleCfg:
    error_rates: tuple[float, ...]
    pair_corr: float
    disagree_threshold: float


@dataclass(frozen=True)
class WardenCfg:
    float_decimals: int
    commit_k: int


@dataclass(frozen=True)
class TrainCfg:
    lr: float
    weight_decay: float
    epochs: int
    batch_size: int
    grad_accum: int
    warmup: int
    max_steps: int
    amp: bool
    ema_decay: float
    grad_clip: float
    loss_weights: tuple[float, float, float]


@dataclass(frozen=True)
class EvalCfg:
    overall_weights: tuple[float, float, float, float, float]
    n_boot: int
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class Settings:
    seed: int
    data: DataCfg
    genomic: GenomicCfg
    curator: CuratorCfg
    pathway: PathwayCfg
    score: ScoreCfg
    ensemble: EnsembleCfg
    warden: WardenCfg
    train: TrainCfg
    eval: EvalCfg


def _floats(value: Any) -> tuple[float, ...]:
    return tuple(float(x) for x in value)


def _ints(value: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in value)


def from_mapping(raw: dict[str, Any]) -> Settings:
    top = raw["etaf"]
    d = raw["data"]
    g = raw["genomic"]
    c = raw["curator"]
    p = raw["pathway"]
    s = raw["score"]
    e = raw["ensemble"]
    w = raw["warden"]
    t = raw["train"]
    v = raw["eval"]
    lw = _floats(t["loss_weights"])
    ow = _floats(v["overall_weights"])
    return Settings(
        seed=int(top["seed"]),
        data=DataCfg(
            n_site_a=int(d["n_site_a"]),
            n_site_b=int(d["n_site_b"]),
            n_genes=int(d["n_genes"]),
            n_variants=int(d["n_variants"]),
            n_clinical=int(d["n_clinical"]),
            catalog_size=int(d["catalog_size"]),
            panel_overlap=float(d["panel_overlap"]),
            signal=float(d["signal"]),
            seed=int(d["seed"]),
        ),
        genomic=GenomicCfg(
            embed_dim=int(g["embed_dim"]),
            n_heads=int(g["n_heads"]),
            n_layers=int(g["n_layers"]),
            graph_dim=int(g["graph_dim"]),
            n_subtypes=int(g["n_subtypes"]),
            dropout=float(g["dropout"]),
        ),
        curator=CuratorCfg(tier_weights=_floats(c["tier_weights"]), mol_dim=int(c["mol_dim"])),
        pathway=PathwayCfg(max_walk=int(p["max_walk"])),
        score=ScoreCfg(
            alpha=float(s["alpha"]),
            beta=float(s["beta"]),
            gamma=float(s["gamma"]),
            top_k=int(s["top_k"]),
        ),
        ensemble=EnsembleCfg(
            error_rates=_floats(e["error_rates"]),
            pair_corr=float(e["pair_corr"]),
            disagree_threshold=float(e["disagree_threshold"]),
        ),
        warden=WardenCfg(float_decimals=int(w["float_decimals"]), commit_k=int(w["commit_k"])),
        train=TrainCfg(
            lr=float(t["lr"]),
            weight_decay=float(t["weight_decay"]),
            epochs=int(t["epochs"]),
            batch_size=int(t["batch_size"]),
            grad_accum=int(t["grad_accum"]),
            warmup=int(t["warmup"]),
            max_steps=int(t["max_steps"]),
            amp=bool(t["amp"]),
            ema_decay=float(t["ema_decay"]),
            grad_clip=float(t["grad_clip"]),
            loss_weights=(lw[0], lw[1], lw[2]),
        ),
        eval=EvalCfg(
            overall_weights=(ow[0], ow[1], ow[2], ow[3], ow[4]),
            n_boot=int(v["n_boot"]),
            seeds=_ints(v["seeds"]),
        ),
    )


def load_settings(path: str | Path) -> Settings:
    raw = Config().from_disk(str(path))
    return from_mapping(dict(raw))


def load_str(text: str) -> Settings:
    raw = Config().from_str(text)
    return from_mapping(dict(raw))
