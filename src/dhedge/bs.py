"""Black-Scholes call price and Greeks at zero rates.

Used three ways: the delta-hedge baseline, the book value B(s) in the
actor-critic value function, and the Greeks that make up its portfolio features.
Everything is torch so the same code serves training and evaluation, and is
differentiable where a caller needs that.

Sign conventions: theta and charm are derivatives with respect to calendar time
t (time passing), not time to maturity, so a long call has negative theta.
"""
from __future__ import annotations

import math

import torch

_TAU_FLOOR = 1e-10
_VOL_FLOOR = 1e-8
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) / _SQRT_2PI


def _cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(-x / math.sqrt(2.0))


def _d1_d2(spot, strike, tau, vol):
    tau = torch.clamp(torch.as_tensor(tau, dtype=spot.dtype, device=spot.device), min=_TAU_FLOOR)
    vol = torch.clamp(torch.as_tensor(vol, dtype=spot.dtype, device=spot.device), min=_VOL_FLOOR)
    strike = torch.as_tensor(strike, dtype=spot.dtype, device=spot.device)
    sd = vol * torch.sqrt(tau)
    d1 = (torch.log(spot / strike) + 0.5 * sd * sd) / sd
    return d1, d1 - sd, tau, vol, strike, sd


def call_price(spot, strike, tau, vol) -> torch.Tensor:
    """Undiscounted call price, exactly intrinsic at or past maturity.

    The explicit expiry branch matters for the actor-critic, whose terminal
    book value must equal the payoff rather than carry a floor-induced sliver
    of time value.
    """
    tau_raw = torch.as_tensor(tau, dtype=spot.dtype, device=spot.device)
    d1, d2, _, _, strike, _ = _d1_d2(spot, strike, tau, vol)
    price = spot * _cdf(d1) - strike * _cdf(d2)
    return torch.where(tau_raw <= _TAU_FLOOR, torch.clamp(spot - strike, min=0.0), price)


def greeks(spot, strike, tau, vol) -> dict[str, torch.Tensor]:
    """Price plus the seven Greeks the 2022 paper feeds its networks."""
    d1, d2, tau, vol, strike, sd = _d1_d2(spot, strike, tau, vol)
    n1 = _pdf(d1)
    sqrt_tau = torch.sqrt(tau)
    price = spot * _cdf(d1) - strike * _cdf(d2)
    delta = _cdf(d1)
    gamma = n1 / (spot * sd)
    vega = spot * n1 * sqrt_tau
    theta = -spot * n1 * vol / (2.0 * sqrt_tau)
    vanna = -n1 * d2 / vol
    charm = n1 * d2 / (2.0 * tau)
    vomma = vega * d1 * d2 / vol
    return {
        "price": price, "delta": delta, "gamma": gamma, "vega": vega,
        "theta": theta, "vanna": vanna, "charm": charm, "vomma": vomma,
    }


def call_delta(spot, strike, tau, vol) -> torch.Tensor:
    d1, *_ = _d1_d2(spot, strike, tau, vol)
    return _cdf(d1)
