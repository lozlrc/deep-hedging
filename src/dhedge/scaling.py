"""Instrument rescaling so every hedge ratio a network must output is order one.

Multiplying an instrument's price by a constant c and dividing its position by
c leaves the hedging problem unchanged: gains delta . dS and proportional costs
eps |d delta| S are both invariant. It only changes the scale of the numbers the
network has to produce.

That matters for the 2019 Heston setup. The variance swap trades near 0.003, so
the model hedge holds a median of about 390 units of it (tail above 1000) while
holding under one unit of stock. A network initialized near zero and trained
with Adam at lr 0.005 moves each weight by about lr per step and cannot reach
those outputs in a realistic budget: measured, it stalled at CVaR 2.42 against
1.95 for the model hedge, having only partly used the variance swap.

The scale is chosen model-free, from the simulated paths alone: each instrument
is rescaled so its per-step price change has the same standard deviation as the
first instrument's (the spot). No model hedge or pricing formula is consulted.
"""
from __future__ import annotations

from typing import Callable

import torch

from .spec import MarketPaths


def instrument_scales(paths: MarketPaths) -> torch.Tensor:
    """c_i = std(dS_1) / std(dS_i), so c_1 = 1 and every column moves alike."""
    dS = paths.instruments[:, 1:, :] - paths.instruments[:, :-1, :]
    sd = dS.reshape(-1, dS.shape[-1]).std(dim=0)
    if torch.any(sd <= 0):
        raise ValueError("an instrument never moves, so it cannot be scaled")
    return (sd[0] / sd).to(paths.instruments.dtype)


def rescale(paths: MarketPaths, scales: torch.Tensor) -> MarketPaths:
    """Return paths whose instrument i is priced in units of 1 / scales[i]."""
    s = scales.to(paths.instruments.device, paths.instruments.dtype)
    return MarketPaths(times=paths.times, spot=paths.spot,
                       instruments=paths.instruments * s, var=paths.var)


def scaled_sampler(sample: Callable[[int, int], MarketPaths],
                   scales: torch.Tensor) -> Callable[[int, int], MarketPaths]:
    return lambda n, seed: rescale(sample(n, seed), scales)


def to_original_units(positions: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Convert positions held in rescaled instruments back to original units."""
    return positions * scales.to(positions.device, positions.dtype)
