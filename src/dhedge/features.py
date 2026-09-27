"""State features for the 2022 actor-critic.

Murray et al. (2022) describe the portfolio to the networks through its
Black-Scholes Greeks rather than through raw positions, so that one network can
in principle hedge any book. Here the book is a short call of size `notional`
plus the current spot holding h_{t-1}, valued at the implied vol sqrt(V_t)
(variance is treated as observable). The spot holding only contributes to delta.

Feature vector, in order (FEATURE_NAMES):
    portfolio delta, gamma, vega, theta, vanna, charm, vomma,
    strike, time to maturity, log spot, variance, calendar time, log lambda.

Risk aversion enters as log lambda because it is sampled log-uniformly over
four decades; a linear lambda would give the network almost no resolution
between 1e-4 and 1e-2.

Normalisation follows the paper: roll out the Black-Scholes delta policy and
standardise every feature with its mean and std under that policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .bs import call_price, greeks

FEATURE_NAMES = ("delta", "gamma", "vega", "theta", "vanna", "charm", "vomma",
                 "strike", "tau", "log_spot", "var", "time", "log_lam")
GREEK_NAMES = FEATURE_NAMES[:7]
N_FEATURES = len(FEATURE_NAMES)
DELTA_INDEX = 0


def implied_vol(var: torch.Tensor, vol_floor: float) -> torch.Tensor:
    """sqrt(V) with a floor. Heston variance under the 2022 parameters violates
    Feller and touches zero, where BS gamma and charm blow up; the floor keeps
    the features finite without affecting any realistic vol level."""
    return torch.sqrt(torch.clamp(var, min=vol_floor * vol_floor))


@dataclass
class HedgeState:
    """One decision point s_t for a batch of N episodes. All tensors are [N]."""

    spot: torch.Tensor
    var: torch.Tensor
    time: torch.Tensor      # calendar time t
    maturity: float         # T, so tau = T - t
    h_prev: torch.Tensor    # spot holding carried into t, h_{t-1}
    strike: torch.Tensor
    lam: torch.Tensor

    @property
    def tau(self) -> torch.Tensor:
        return torch.clamp(self.maturity - self.time, min=0.0)


def market_block(spot: torch.Tensor, var: torch.Tensor, time: torch.Tensor, maturity: float,
                 strike: torch.Tensor, lam: torch.Tensor, notional: float,
                 vol_floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Features with the holding set to zero, plus the call delta.

    The holding enters the feature vector only through the delta slot, and
    linearly (portfolio delta = h - notional * call delta). Splitting it out lets
    training precompute every Greek once in float64 and then add h cheaply,
    and keeps the gradient path from an action to the next state's features
    exact and trivial.

    Works on any matching shapes; returns ([..., N_FEATURES], [...]).
    """
    tau = torch.clamp(maturity - time, min=0.0)
    g = greeks(spot, strike, tau, implied_vol(var, vol_floor))
    cols = [-notional * g["delta"]]
    cols += [-notional * g[name] for name in GREEK_NAMES[1:]]
    cols += [strike * torch.ones_like(spot), tau * torch.ones_like(spot), torch.log(spot),
             var, time * torch.ones_like(spot), torch.log(lam) * torch.ones_like(spot)]
    return torch.stack(cols, dim=-1), g["delta"]


def raw_features(state: HedgeState, notional: float, vol_floor: float) -> torch.Tensor:
    """[N, N_FEATURES] unnormalised features of a batch of states."""
    block, _ = market_block(state.spot, state.var, state.time, state.maturity, state.strike,
                            state.lam, notional, vol_floor)
    block = block.clone()
    block[:, DELTA_INDEX] = block[:, DELTA_INDEX] + state.h_prev
    return block


def portfolio_delta(state: HedgeState, notional: float, vol_floor: float) -> torch.Tensor:
    """h_{t-1} - notional * BS call delta: the delta the actor's skip connection cancels."""
    _, cdelta = market_block(state.spot, state.var, state.time, state.maturity, state.strike,
                             state.lam, notional, vol_floor)
    return state.h_prev - notional * cdelta


def book_value(h_prev: torch.Tensor, spot: torch.Tensor, var: torch.Tensor, tau: torch.Tensor,
               strike: torch.Tensor, notional: float, vol_floor: float) -> torch.Tensor:
    """B(s_t) = h_{t-1} S_t - notional * C(S_t, K, tau_t, sqrt(V_t)).

    At tau = 0 bs.call_price is exactly intrinsic, so B(s_T) is the terminal
    reward r_T with no residual time value.
    """
    return h_prev * spot - notional * call_price(spot, strike, tau, implied_vol(var, vol_floor))


@dataclass
class FeatureNormalizer:
    mean: torch.Tensor   # [N_FEATURES]
    std: torch.Tensor    # [N_FEATURES]
    clip: float = 10.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Clipping guards against a rare state far outside the BS-delta rollout
        # (a near-zero Heston variance a day before expiry) saturating every
        # sigmoid in the first layer.
        # dtype first, then device: a combined cast between float64 and MPS
        # silently yields zeros in torch 2.14.
        mean = self.mean.to(x.dtype).to(x.device)
        std = self.std.to(x.dtype).to(x.device)
        return torch.clamp((x - mean) / std, -self.clip, self.clip)

    def to(self, device, dtype) -> "FeatureNormalizer":
        return FeatureNormalizer(self.mean.to(dtype).to(device), self.std.to(dtype).to(device),
                                 self.clip)


def fit_normalizer(spot: torch.Tensor, var: torch.Tensor, times: torch.Tensor, strike: torch.Tensor,
                   lam: torch.Tensor, notional: float, vol_floor: float,
                   clip: float = 10.0) -> FeatureNormalizer:
    """Mean and std of every feature along a Black-Scholes delta rollout.

    spot, var: [N, n+1]; strike, lam: [N]. Uses the decision dates t_0..t_{n-1},
    where the carried holding is h_{t-1} = notional * delta_{t-1}, h_{-1} = 0.
    A feature that is constant under the rollout (a single strike) gets std 1,
    so it is centred to zero rather than divided by zero.
    """
    n = spot.shape[1] - 1
    maturity = float(times[-1])
    t_grid = times[:n].to(spot.dtype).expand(spot.shape[0], n)
    block, cdelta = market_block(spot[:, :n], var[:, :n], t_grid, maturity,
                                 strike[:, None], lam[:, None], notional, vol_floor)
    h_prev = torch.cat([torch.zeros_like(cdelta[:, :1]), notional * cdelta[:, :-1]], dim=1)
    block = block.clone()
    block[..., DELTA_INDEX] = block[..., DELTA_INDEX] + h_prev
    flat = block.reshape(-1, N_FEATURES)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0)
    scale = torch.clamp(mean.abs(), min=1.0)
    std = torch.where(std < 1e-9 * scale, torch.ones_like(std), std)
    return FeatureNormalizer(mean, std, clip)
