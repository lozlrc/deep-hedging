"""Shared contracts: model parameters from the two papers and the path container
every hedging method consumes.

Units: time in years, zero rates and zero drift (both papers price under a
risk-neutral measure with no carry), prices in currency units of the spot.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

T_30D = 30.0 / 365.0  # both papers: 30 trading days, one rebalance per day
N_STEPS = 30


@dataclass(frozen=True)
class HestonParams:
    """dS = sqrt(V) S dB,  dV = kappa (theta - V) dt + xi sqrt(V) dW,  d<B,W> = rho dt.

    Buehler et al. (2019) write the same model as alpha (b - V) dt + sigma sqrt(V) dW,
    so alpha = kappa, b = theta, sigma = xi.
    """

    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float | None
    s0: float

    def require_v0(self) -> float:
        if self.v0 is None:
            raise ValueError("v0 is not given by the source paper; calibrate it first")
        return float(self.v0)


# Buehler, Gonon, Teichmann, Wood (2019), Section 5.2. Fully specified.
BUEHLER_2019 = HestonParams(kappa=1.0, theta=0.04, xi=2.0, rho=-0.7, v0=0.04, s0=100.0)

# Murray, Wood, Buehler, Wiese, Pakkanen (2022), Section 4. The paper gives
# kappa, theta, xi, rho, mu = 0 and S0 = 1 but never states v0, so it is left
# as None and must be calibrated before use.
MURRAY_2022 = HestonParams(kappa=8.0, theta=0.00625, xi=1.0, rho=-0.7, v0=None, s0=1.0)


@dataclass
class MarketPaths:
    """Simulated trajectories on the trading grid t_0 = 0 < ... < t_n = T.

    instruments[..., 0] is always the spot. Buehler (2019) adds a variance swap
    as instruments[..., 1]; the 2022 setup hedges with the spot only.
    """

    times: torch.Tensor            # [n + 1]
    spot: torch.Tensor             # [N, n + 1]
    instruments: torch.Tensor      # [N, n + 1, d]
    var: torch.Tensor | None = None  # [N, n + 1] instantaneous variance, when the model has one

    @property
    def n_paths(self) -> int:
        return int(self.spot.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.spot.shape[1] - 1)

    @property
    def n_instruments(self) -> int:
        return int(self.instruments.shape[-1])

    def to(self, device: torch.device | str) -> "MarketPaths":
        return MarketPaths(
            times=self.times.to(device),
            spot=self.spot.to(device),
            instruments=self.instruments.to(device),
            var=None if self.var is None else self.var.to(device),
        )
