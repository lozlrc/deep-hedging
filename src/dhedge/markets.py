"""Simple reference market: Black-Scholes geometric Brownian motion.

Heston lives in heston.py. GBM is here because it is the market where the right
answer is known (discrete delta hedging), which makes it the test bed for every
learning method before it is pointed at Heston.
"""
from __future__ import annotations

import math

import torch

from .spec import MarketPaths


def gbm_paths(n_paths: int, s0: float, sigma: float, T: float, n_steps: int,
              seed: int = 0, dtype: torch.dtype = torch.float64,
              device: str = "cpu") -> MarketPaths:
    """Exact lognormal sampling with zero drift, so S is a martingale."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    dt = T / n_steps
    z = torch.randn(n_paths, n_steps, generator=g, dtype=dtype)
    incr = -0.5 * sigma * sigma * dt + sigma * math.sqrt(dt) * z
    logs = torch.cat([torch.zeros(n_paths, 1, dtype=dtype), torch.cumsum(incr, dim=1)], dim=1)
    spot = s0 * torch.exp(logs)
    times = torch.linspace(0.0, T, n_steps + 1, dtype=dtype)
    var = torch.full_like(spot, sigma * sigma)
    return MarketPaths(times=times, spot=spot, instruments=spot.unsqueeze(-1), var=var).to(device)
