"""The one hedging P&L definition every method in this repo is scored with.

Buehler et al. (2019), eq. (2.1) and (5.7):

    PL = p0 - Z + sum_k delta_k . (S_{k+1} - S_k) - sum_{k=0}^{n} c_k(delta_k - delta_{k-1})
    c_k(n) = sum_i eps * |n^i| * S^i_k,   delta_{-1} = 0

with delta_n = 0 when the book is liquidated at maturity. The 2022 paper values
the terminal portfolio at book value instead, so liquidation cost is optional.
Keeping this in one place is what makes the methods comparable: a baseline and
a learned policy can differ only in the positions they choose.
"""
from __future__ import annotations

import torch


def hedge_pnl(instruments: torch.Tensor, positions: torch.Tensor, payoff: torch.Tensor,
              eps: float = 0.0, premium: float | torch.Tensor = 0.0,
              liquidate: bool = True) -> torch.Tensor:
    """Terminal P&L of a short claim hedged with `positions`.

    instruments: [N, n+1, d] prices on the trading grid.
    positions:   [N, n, d] holdings chosen at t_0..t_{n-1}, held over [t_k, t_{k+1}].
    payoff:      [N] the claim's payoff Z, paid by the hedger at T.
    eps:         proportional cost rate, charged on every rebalance.
    """
    if positions.dim() == 2:
        positions = positions.unsqueeze(-1)
    if instruments.shape[1] != positions.shape[1] + 1:
        raise ValueError("positions must cover exactly the n rebalancing dates")
    if instruments.shape[-1] != positions.shape[-1]:
        raise ValueError("positions and instruments disagree on the number of instruments")

    ds = instruments[:, 1:, :] - instruments[:, :-1, :]
    gains = (positions * ds).sum(dim=(1, 2))

    costs = torch.zeros_like(gains)
    if eps > 0.0:
        zeros = torch.zeros_like(positions[:, :1, :])
        prev = torch.cat([zeros, positions[:, :-1, :]], dim=1)
        trades = positions - prev                                   # trades at t_0..t_{n-1}
        costs = eps * (trades.abs() * instruments[:, :-1, :]).sum(dim=(1, 2))
        if liquidate:
            costs = costs + eps * (positions[:, -1, :].abs() * instruments[:, -1, :]).sum(dim=1)

    return premium - payoff + gains - costs


def call_payoff(spot: torch.Tensor, strike: float | torch.Tensor) -> torch.Tensor:
    """(S_T - K)^+ from a [N, n+1] spot array."""
    return torch.clamp(spot[:, -1] - strike, min=0.0)
