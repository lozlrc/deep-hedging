"""Analytic hedging baselines: Black-Scholes delta and the Whalley-Wilmott band.

Both return positions of shape [N, n, 1] in the spot, chosen at t_0..t_{n-1},
so they plug straight into pnl.hedge_pnl next to any learned policy.
"""
from __future__ import annotations

import torch

from .bs import call_delta, greeks
from .spec import MarketPaths


def _vol_and_tau(paths: MarketPaths, T: float, vol):
    """Per-date vol [N, n] and time to expiry [n] at the rebalancing dates."""
    n = paths.n_steps
    tau = (T - paths.times[:n]).to(paths.spot.dtype)
    if vol is None:
        if paths.var is None:
            raise ValueError("paths carry no variance process; pass a fixed vol")
        sig = torch.sqrt(torch.clamp(paths.var[:, :n], min=0.0))
    else:
        sig = torch.full_like(paths.spot[:, :n], float(vol))
    return sig, tau


def bs_delta_positions(paths: MarketPaths, strike: float, T: float, notional: float = 1.0,
                       vol: float | None = None) -> torch.Tensor:
    """Black-Scholes delta of `notional` calls at each rebalancing date.

    With vol=None the implied vol is sqrt(V_t), which is the 2022 paper's
    baseline ("the square root of the variance process as the implied
    volatility"). A fixed vol gives the textbook constant-vol delta.
    """
    n = paths.n_steps
    sig, tau = _vol_and_tau(paths, T, vol)
    delta = call_delta(paths.spot[:, :n], strike, tau[None, :], sig)
    return (notional * delta).unsqueeze(-1)


def whalley_wilmott_half_width(spot, gamma, eps: float, risk_aversion: float) -> torch.Tensor:
    """H = (3 eps S Gamma^2 / (2 gamma_ra))^(1/3), the leading-order band half-width.

    Whalley and Wilmott (1997), "An asymptotic analysis of an optimal hedging
    model for option pricing with transaction costs", Mathematical Finance 7(3),
    eq. (17) and Section 4: the band is y* +- (3 eps S delta(T,t) (y*_S)^2 / (2 gamma))^(1/3)
    for exponential utility U(x) = 1 - exp(-gamma x), where trading N shares
    costs eps N S and delta(T,t) = e^{-r(T-t)}. Here r = 0, so delta(T,t) = 1,
    and with the drift equal to r the frictionless holding y* is the
    Black-Scholes delta, so y*_S is the portfolio gamma.
    """
    return (1.5 * eps * spot * gamma * gamma / risk_aversion) ** (1.0 / 3.0)


def whalley_wilmott_positions(paths: MarketPaths, strike: float, T: float, eps: float,
                              risk_aversion: float, notional: float = 1.0,
                              vol: float | None = None) -> torch.Tensor:
    """No-trade band around the Black-Scholes delta for proportional costs eps.

    The band is notional * Delta +- H with H computed from the portfolio gamma
    notional * Gamma. A position inside the band is left alone; outside, it is
    moved only to the nearest edge, which is the optimal policy for costs linear
    in the traded amount. The book starts flat (delta_{-1} = 0, matching
    hedge_pnl), so the first trade goes to the near edge of the first band.
    Path dependent, so it loops over dates; each step is vectorised over paths.
    """
    if risk_aversion <= 0.0:
        raise ValueError("risk_aversion must be positive")
    n = paths.n_steps
    sig, tau = _vol_and_tau(paths, T, vol)
    g = greeks(paths.spot[:, :n], strike, tau[None, :], sig)
    centre = notional * g["delta"]
    half = whalley_wilmott_half_width(paths.spot[:, :n], notional * g["gamma"], eps, risk_aversion)
    pos = torch.empty_like(centre)
    prev = torch.zeros_like(centre[:, 0])
    for k in range(n):
        prev = torch.minimum(torch.maximum(prev, centre[:, k] - half[:, k]), centre[:, k] + half[:, k])
        pos[:, k] = prev
    return pos.unsqueeze(-1)
