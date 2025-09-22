"""Lightweight substitutes for the small subset of timm.layers used by BiRefNet."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def to_2tuple(val: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(val, tuple):
        return val
    return (val, val)


class DropPath(nn.Module):
    """Stochastic depth per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        output = x / keep_prob * random_tensor
        return output


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0,
                  a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """torch.nn.init.trunc_normal_ wrapper with fallback."""
    try:
        return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    except AttributeError:  # pragma: no cover - older torch
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp > a) & (tmp < b)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
        return tensor
