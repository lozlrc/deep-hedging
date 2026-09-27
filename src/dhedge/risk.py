"""Convex risk measures on a sample of terminal P&L, Buehler et al. (2019), Section 3.

Every function takes the P&L X (a gain, so losses are -X) as a 1-D tensor of
equally weighted scenarios and returns a scalar in currency units. All of them
are cash invariant, rho(X + c) = rho(X) - c, which is what makes the minimal
risk of the hedged short position an indifference price.

Everything is torch and differentiable in X, because pathwise deep hedging
backpropagates the risk of the simulated P&L into the hedging networks.
"""
from __future__ import annotations

import math

import torch


def _check(X: torch.Tensor) -> torch.Tensor:
    if X.dim() != 1 or X.numel() == 0:
        raise ValueError("X must be a non-empty 1-D tensor of scenario P&L")
    return X


def entropic_risk(X: torch.Tensor, lam: float) -> torch.Tensor:
    """(1/lam) log E[exp(-lam X)].

    logsumexp keeps it finite for large lam, where exp(-lam X) of the worst
    scenario would overflow and the measure approaches -min X.
    """
    X = _check(X)
    if lam <= 0.0:
        raise ValueError("lam must be positive")
    return (torch.logsumexp(-lam * X, dim=0) - math.log(X.numel())) / lam


def exp_utility(X: torch.Tensor, lam: float) -> torch.Tensor:
    """Certainty equivalent of exponential utility, the negative of entropic risk."""
    return -entropic_risk(X, lam)


def cvar(X: torch.Tensor, alpha: float) -> torch.Tensor:
    """Exact empirical expected shortfall of the loss -X at level alpha.

    Paper eq. (5.1): rho(X) = 1/(1 - alpha) * integral_0^{1-alpha} VaR_gamma(X) dgamma.
    On N equally weighted scenarios the tail holds m = (1 - alpha) N scenarios,
    generally not an integer: the floor(m) largest losses enter with weight one
    and the next one with the fractional weight m - floor(m). Dropping that
    fraction would bias the estimate and make it discontinuous in alpha.
    alpha = 0 gives -E[X]; alpha = 1 gives the worst loss -min X.
    """
    X = _check(X)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    losses = torch.sort(-X, descending=True).values
    if alpha >= 1.0:
        return losses[0]
    m = (1.0 - alpha) * X.numel()
    idx = torch.arange(X.numel(), dtype=X.dtype, device=X.device)
    weights = torch.clamp(m - idx, min=0.0, max=1.0)
    return (weights * losses).sum() / m


def cvar_oce(X: torch.Tensor, w: torch.Tensor | float, alpha: float) -> torch.Tensor:
    """Rockafellar-Uryasev form w + E[(-X - w)^+] / (1 - alpha).

    Its minimum over w is cvar(X, alpha), attained at the alpha-VaR of the
    loss. Training minimises it jointly over w and the hedge, which avoids
    differentiating through a sort and yields VaR as a by-product.
    """
    X = _check(X)
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must lie in [0, 1) for the OCE form")
    return w + torch.clamp(-X - w, min=0.0).mean() / (1.0 - alpha)


def risk_measure(X: torch.Tensor, risk: str, alpha: float | None = None,
                 lam: float | None = None) -> torch.Tensor:
    """Dispatch by name, so callers can select the measure from configuration."""
    if risk == "cvar":
        if alpha is None:
            raise ValueError("cvar needs alpha")
        return cvar(X, alpha)
    if risk == "entropic":
        if lam is None:
            raise ValueError("entropic risk needs lam")
        return entropic_risk(X, lam)
    raise ValueError(f"unknown risk measure {risk!r}")
