from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Batch:
    index: torch.Tensor
    features: torch.Tensor
    expression: torch.Tensor
    variants: torch.Tensor
    clinical: torch.Tensor
    panel_mask: torch.Tensor
    subtype: torch.Tensor
    site: torch.Tensor

    def to(self, device: torch.device) -> Batch:
        return Batch(
            index=self.index.to(device),
            features=self.features.to(device),
            expression=self.expression.to(device),
            variants=self.variants.to(device),
            clinical=self.clinical.to(device),
            panel_mask=self.panel_mask.to(device),
            subtype=self.subtype.to(device),
            site=self.site.to(device),
        )

    def __len__(self) -> int:
        return int(self.index.shape[0])


def iter_batches(
    tensors: dict[str, torch.Tensor],
    site: np.ndarray,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> Iterator[Batch]:
    n = tensors["features"].shape[0]
    order = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    site_t = torch.from_numpy(site.astype(np.int64))
    for start in range(0, n, batch_size):
        sel = torch.from_numpy(order[start : start + batch_size].astype(np.int64))
        yield Batch(
            index=sel,
            features=tensors["features"][sel],
            expression=tensors["expression"][sel],
            variants=tensors["variants"][sel],
            clinical=tensors["clinical"][sel],
            panel_mask=tensors["panel_mask"][sel],
            subtype=tensors["subtype"][sel],
            site=site_t[sel],
        )
