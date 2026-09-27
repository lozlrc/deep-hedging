"""Risk-averse actor-critic hedging, after Murray, Wood, Buehler, Wiese, Pakkanen (2022),
"Deep Hedging: Continuous Reinforcement Learning for Hedging of General
Portfolios across Multiple Risk Aversions".

Problem. Hedge a short European call of size `notional` with the spot only,
trading once a day, paying c(a) = eps * |a| * S per trade. The objective is the
exponential utility U_lam(X) = -(1/lam) log E[exp(-lam X)], and lam is part of
the state: every episode draws its own lam log-uniformly on [1e-4, 1], so one
pair of networks covers the whole range of risk aversions.

Cash accounting (paper eq. 14). The action a_t is a trade, h_t = h_{t-1} + a_t,
h_{-1} = 0, and

    r_t = -a_t S_t - c(a_t)                        t = 0..n-1
    r_T = h_{n-1} S_T - notional * (S_T - K)^+     at maturity, book value, no liquidation

so sum_t r_t is exactly dhedge.pnl.hedge_pnl(..., liquidate=False).

Skip connections. The actor is a_t = -Delta_portfolio(s_t) + pi_theta(s_t): with
pi_theta = 0 the trade flattens portfolio delta, which is the BS delta hedge.
The critic is V(s_t) = B(s_t) + V_phi(s_t) with book value
B(s_t) = h_{t-1} S_t - notional * C_BS(S_t, K, tau_t, sqrt(V_t)). Both residual
networks start at exactly zero, so training starts from the delta hedge and
its book value and only has to learn corrections.

Transitions. From s_t the action produces r_t and s_{t+1}; the terminal state
s_n has V(s_n) = B(s_n) = r_T exactly (no learned residual), so the reward r_T
enters through the last transition's target and the Bellman recursion needs no
special case.

Precision. Paths, Greeks and book values are computed in float64 and cast once
per minibatch to the training dtype (float32 by default, and MPS has no
float64). Evaluation rollouts accumulate holdings in float64.
"""
from __future__ import annotations

import copy
import math
import time as _time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .bs import call_price
from .features import (DELTA_INDEX, N_FEATURES, FeatureNormalizer, HedgeState, fit_normalizer,
                       implied_vol, market_block, portfolio_delta, raw_features)
from .spec import MarketPaths

# Critic loss is extended linearly (in lam * u) beyond this exponent. The exact
# loss is kept for |lam u| <= 30, i.e. whenever the one-step residual is within
# 30 / lam of the target, which covers every state the policy actually visits.
_CRITIC_EXP_CAP = 30.0
# Actor loss exponents are shifted down per batch only when they exceed this.
_ACTOR_EXP_CAP = 60.0

# Default training budget. The paper trains 100,000 iterations of 2048
# episodes, days of laptop compute. Measured on an M2 Pro with the paper's
# network sizes: 0.105 s per 2048-episode iteration on MPS (19.5k episodes/s),
# 0.334 s on CPU (6.1k episodes/s). 15,000 iterations (30.7M episodes) took
# 1675 s (28 minutes, 0.112 s/iter including GBM path sampling) on MPS; the
# wall-clock cap stops a CPU run at 30 minutes (roughly 5,400 iterations).
DEFAULT_ITERS = 15_000
DEFAULT_MAX_SECONDS = 1800.0


def default_device() -> str:
    """MPS when available (about 3x the CPU throughput here), else CPU."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------- objectives

def exp_utility(pnl: torch.Tensor, lam: float) -> float:
    """U_lam(X) = -(1/lam) log E[exp(-lam X)], via logsumexp so lam = 1 cannot overflow."""
    x = pnl.detach().to(torch.float64).reshape(-1)
    return float(-(torch.logsumexp(-lam * x, dim=0) - math.log(x.numel())) / lam)


def _exp_excess(w: torch.Tensor) -> torch.Tensor:
    """g(w) = exp(-w) - 1 + w, accurate for tiny w and linear past w = -cap.

    At lam = 1e-4 the two exponential terms cancel to roughly 1e-9 of their
    size, which float32 cannot resolve, so small |w| uses the Taylor series
    w^2/2 - w^3/6 + w^4/24 (truncation error below 1e-17 for |w| < 1e-3).
    """
    small = w.abs() < 1e-3
    w_s = torch.where(small, w, torch.zeros_like(w))
    series = w_s * w_s * (0.5 - w_s / 6.0 + w_s * w_s / 24.0)
    w_c = torch.clamp(w, min=-_CRITIC_EXP_CAP)
    exact = torch.expm1(-w_c) + w_c
    # Linear continuation g(-c) + g'(-c) (w + c) keeps the loss convex and C1
    # and bounds its gradient by exp(cap).
    slope = 1.0 - math.exp(_CRITIC_EXP_CAP)
    linear = exact + slope * (w - w_c)
    big = torch.where(w < -_CRITIC_EXP_CAP, linear, exact)
    return torch.where(small, series, big)


def critic_loss(u: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """L = E[(exp(-lam u) + lam u - 1) / lam^2],  u = y - V(s),  y = r + Vbar(s').

    Why this loss. The risk-averse Bellman equation (paper eq. 9) is
    exp(-lam V(s)) = E[exp(-lam y) | s]. Write g(w) = exp(-w) - 1 + w, which is
    convex, non-negative and zero only at w = 0, so L(V) = E[g(lam (Y - V))] / lam^2.
    Differentiating in the scalar V:

        dL/dV = E[-lam g'(lam (Y - V))] / lam^2 = -E[1 - exp(-lam (Y - V))] / lam
              = (E[exp(-lam (Y - V))] - 1) / lam,

    which vanishes iff exp(lam V) E[exp(-lam Y)] = 1, i.e. iff
    V = -(1/lam) log E[exp(-lam Y)] = U_lam(Y). The second derivative is
    E[exp(-lam (Y - V))] > 0, so this is the unique minimiser: the population
    minimiser of L solves eq. 9 exactly. Since lam is part of the state, the
    same holds conditionally on each (s, lam) in a mixed-lam minibatch. As
    lam -> 0, g(lam u) / lam^2 -> u^2 / 2, the ordinary risk-neutral TD loss.

    The only departure from the exact loss is the linear tail of _exp_excess
    for lam u < -30, which cannot move the minimiser of any bounded target
    whose support lies within 30 / lam of it.
    """
    return (_exp_excess(lam * u) / (lam * lam)).mean()


def actor_loss(x: torch.Tensor, baseline: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """E[(1/lam) exp(-lam X)],  X = r_t + V(s_{t+1}), minimised over the action.

    For fixed (s, lam), U_lam(X) = -(1/lam) log(lam E[(1/lam) exp(-lam X)]) is
    strictly decreasing in the loss, so minimising it maximises the utility
    of acting at s and then following V. Gradients reach the actor through
    r_t and through the next holding h_t in V(s_{t+1}).

    Numerical guard, two positive rescalings that leave every per-state
    minimiser and every per-state gradient direction unchanged:
      1. Each sample is multiplied by exp(lam V(s_t)) with V(s_t) detached, i.e.
         the exponent uses X - V(s_t), the one-step residual, which is O(1) at
         the Bellman fixed point instead of O(notional). The factor does not
         depend on the action, so it only reweights states.
      2. If the largest remaining exponent exceeds 60, the whole batch is
         multiplied by exp(-m) with m a detached scalar: the batch gradient keeps
         its exact direction and Adam is invariant to its overall scale.
    """
    z = -lam * (x - baseline.detach())
    m = torch.clamp(z.detach().max() - _ACTOR_EXP_CAP, min=0.0)
    return (torch.exp(z - m) / lam).mean()


def step_reward(trade: torch.Tensor, spot: torch.Tensor, eps: float) -> torch.Tensor:
    """r_t = -a_t S_t - eps |a_t| S_t for t < T."""
    return -trade * spot - eps * trade.abs() * spot


def episode_rewards(spot: torch.Tensor, holdings: torch.Tensor, strike, notional: float,
                    eps: float) -> torch.Tensor:
    """[N, n+1] rewards r_0..r_{n-1}, r_T of an episode with holdings [N, n] or [N, n, 1]."""
    if holdings.dim() == 3:
        holdings = holdings[..., 0]
    prev = torch.cat([torch.zeros_like(holdings[:, :1]), holdings[:, :-1]], dim=1)
    trades = holdings - prev
    r = step_reward(trades, spot[:, :-1], eps)
    strike = torch.as_tensor(strike, dtype=spot.dtype, device=spot.device)
    terminal = holdings[:, -1] * spot[:, -1] - notional * torch.clamp(spot[:, -1] - strike, min=0.0)
    return torch.cat([r, terminal[:, None]], dim=1)


# ---------------------------------------------------------------- networks

def _mlp(n_in: int, hidden: int, depth: int, dtype: torch.dtype) -> nn.Sequential:
    """depth hidden layers of `hidden` sigmoid units and a linear head that
    starts at exactly zero, so the skip connections define the initial policy
    and value."""
    layers: list[nn.Module] = []
    width = n_in
    for _ in range(depth):
        layers += [nn.Linear(width, hidden, dtype=dtype), nn.Sigmoid()]
        width = hidden
    head = nn.Linear(width, 1, dtype=dtype)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """theta_bar <- (1 - tau) theta_bar + tau theta."""
    for p_bar, p in zip(target.parameters(), online.parameters()):
        p_bar.lerp_(p, tau)


# ---------------------------------------------------------------- batches

@dataclass
class _Batch:
    """Everything about a minibatch of episodes that does not depend on the policy."""

    spot: torch.Tensor      # [N, n+1]
    claim: torch.Tensor     # [N, n+1] notional * C_BS, exactly the payoff at T
    hstar: torch.Tensor     # [N, n]   notional * BS call delta, the delta-neutral holding
    xs: torch.Tensor        # [N, n, F] normalised features with h = 0 (unclipped)
    lam: torch.Tensor       # [N]
    noise: torch.Tensor     # [N, n]   exploration noise, zero when deterministic

    def to(self, device, dtype) -> "_Batch":
        return _Batch(*(_to_net(getattr(self, f), device, dtype) for f in
                        ("spot", "claim", "hstar", "xs", "lam", "noise")))

    @property
    def n_steps(self) -> int:
        return int(self.hstar.shape[1])


def _to_net(x: torch.Tensor, device, dtype) -> torch.Tensor:
    """Cast on the CPU first, then move. A combined .to(device, dtype) across
    MPS and float64 silently returns zeros in torch 2.14, so the two steps are
    kept separate in both directions."""
    return x.to(dtype).to(device)


def _as_strikes(strikes, n: int, gen: torch.Generator) -> torch.Tensor:
    """A scalar strike, or one drawn uniformly per episode from a set of strikes."""
    k = torch.as_tensor(strikes, dtype=torch.float64).reshape(-1)
    if k.numel() == 1:
        return k.expand(n).clone()
    return k[torch.randint(0, k.numel(), (n,), generator=gen)]


def _as_lam(lam, n: int) -> torch.Tensor:
    return torch.as_tensor(lam, dtype=torch.float64).reshape(-1).expand(n).clone()


class ActorCritic:
    """Risk-averse actor-critic with lam in the state.

    Defaults are the paper's: 3 sigmoid layers of 256 units per network, Adam
    at 1e-3 (actor) and 1e-4 (critic), target critic with soft update 1e-3,
    minibatches of 2048 episodes, one update of each network at every time step.
    """

    def __init__(self, notional: float = 100.0, eps: float = 0.002,
                 lam_range: tuple[float, float] = (1e-4, 1.0), hidden: int = 256, depth: int = 3,
                 lr_actor: float = 1e-3, lr_critic: float = 1e-4, tau: float = 1e-3,
                 explore: float = 0.0, vol_floor: float = 1e-3, clip: float = 10.0,
                 value_scale: float = 1.0, device: str = "cpu",
                 dtype: torch.dtype = torch.float32, seed: int = 0):
        self.notional = float(notional)
        self.eps = float(eps)
        self.lam_range = (float(lam_range[0]), float(lam_range[1]))
        self.tau = float(tau)
        # Exploration std, as a fraction of notional, added to executed trades
        # during training only. It widens the holdings the critic sees around
        # the on-policy path, which is where the actor reads dV/dh.
        self.explore = float(explore)
        self.vol_floor = float(vol_floor)
        self.clip = float(clip)
        # The actor head outputs a fraction of notional; the critic head
        # outputs currency units times value_scale.
        self.action_scale = self.notional
        self.value_scale = float(value_scale)
        self.device = torch.device(device)
        self.dtype = dtype
        self.seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.actor = _mlp(N_FEATURES, hidden, depth, dtype).to(self.device)
            self.critic = _mlp(N_FEATURES, hidden, depth, dtype).to(self.device)
        self.target = copy.deepcopy(self.critic)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.normalizer: FeatureNormalizer | None = None
        self.maturity: float | None = None
        self.episodes_seen = 0

    # ------------------------------------------------------------ features

    def fit_normalizer(self, paths: MarketPaths, strikes, seed: int = 0) -> FeatureNormalizer:
        """Standardise features under the BS delta policy (the paper's recipe)."""
        gen = torch.Generator().manual_seed(seed)
        n = paths.n_paths
        spot, var, times = self._paths64(paths)
        self.normalizer = fit_normalizer(spot, var, times, _as_strikes(strikes, n, gen),
                                         self._sample_lam(n, gen), self.notional,
                                         self.vol_floor, self.clip)
        self.maturity = float(times[-1])
        return self.normalizer

    @staticmethod
    def _paths64(paths: MarketPaths):
        spot = paths.spot.detach().cpu().to(torch.float64)
        if paths.var is None:
            raise ValueError("the actor-critic needs the variance path: its features use sqrt(V_t)")
        var = paths.var.detach().cpu().to(torch.float64)
        return spot, var, paths.times.detach().cpu().to(torch.float64)

    def _sample_lam(self, n: int, gen: torch.Generator) -> torch.Tensor:
        lo, hi = math.log(self.lam_range[0]), math.log(self.lam_range[1])
        return torch.exp(lo + (hi - lo) * torch.rand(n, generator=gen, dtype=torch.float64))

    def _build_batch(self, paths: MarketPaths, strike: torch.Tensor, lam: torch.Tensor,
                     noise: torch.Tensor | None = None) -> _Batch:
        """float64 CPU precompute of Greeks, book values and normalised features."""
        assert self.normalizer is not None
        spot, var, times = self._paths64(paths)
        N, n = spot.shape[0], spot.shape[1] - 1
        if self.maturity is not None and abs(float(times[-1]) - self.maturity) > 1e-12:
            raise ValueError("paths have a different maturity from the ones the normaliser saw")
        maturity = float(times[-1])
        tau = torch.clamp(maturity - times, min=0.0).expand(N, n + 1)
        claim = self.notional * call_price(spot, strike[:, None], tau,
                                           implied_vol(var, self.vol_floor))
        t_grid = times[:n].expand(N, n)
        block, cdelta = market_block(spot[:, :n], var[:, :n], t_grid, maturity,
                                     strike[:, None], lam[:, None], self.notional, self.vol_floor)
        norm = self.normalizer
        xs = (block - norm.mean.to(torch.float64)) / norm.std.to(torch.float64)
        if noise is None:
            noise = torch.zeros(N, n, dtype=torch.float64)
        return _Batch(spot, claim, self.notional * cdelta, xs, lam, noise)

    def _x(self, b: _Batch, t: int, h: torch.Tensor) -> torch.Tensor:
        """Normalised, clipped features of s_t given the carried holding h = h_{t-1}.
        h only enters the portfolio-delta slot, linearly."""
        x = b.xs[:, t, :]
        std0 = self.normalizer.std[DELTA_INDEX].item()
        delta = x[:, DELTA_INDEX] + h / std0
        x = torch.cat([delta[:, None], x[:, DELTA_INDEX + 1:]], dim=1)
        return torch.clamp(x, -self.clip, self.clip)

    def _value(self, net: nn.Module, b: _Batch, t: int, h: torch.Tensor) -> torch.Tensor:
        """V(s_t) = B(s_t) + V_phi(s_t); at t = n it is exactly the terminal reward."""
        book = h * b.spot[:, t] - b.claim[:, t]
        if t == b.n_steps:
            return book
        return book + self.value_scale * net(self._x(b, t, h)).squeeze(-1)

    def _trade(self, b: _Batch, t: int, h: torch.Tensor) -> torch.Tensor:
        """a_t = -(h_{t-1} - notional * delta_t) + pi_theta(s_t)."""
        return b.hstar[:, t] - h + self.action_scale * self.actor(self._x(b, t, h)).squeeze(-1)

    # ------------------------------------------------------------ training

    def train(self, sample_paths: Callable[[int, int], MarketPaths], strikes=1.0,
              n_iters: int = DEFAULT_ITERS, batch: int = 2048, seed: int = 0,
              max_seconds: float | None = DEFAULT_MAX_SECONDS, n_norm: int = 20000,
              log_every: int = 0) -> list[dict]:
        """On-policy training on fresh minibatches of `batch` episodes.

        Each iteration draws `batch` paths from sample_paths(n, seed), one lam
        per episode, and walks the episodes forward in time, updating the critic,
        then the actor, then the target critic at every step t. No replay buffer.
        Returns one history row per iteration.
        """
        if self.normalizer is None:
            self.fit_normalizer(sample_paths(n_norm, 10_000_019 + seed), strikes, seed=seed)
        gen = torch.Generator().manual_seed(1_000_003 * (seed + 1))
        history: list[dict] = []
        start = _time.perf_counter()
        for it in range(n_iters):
            paths = sample_paths(batch, 7919 * seed + it + 1)
            strike = _as_strikes(strikes, batch, gen)
            lam = self._sample_lam(batch, gen)
            noise = None
            if self.explore > 0.0:
                noise = self.explore * self.notional * torch.randn(
                    batch, paths.n_steps, generator=gen, dtype=torch.float64)
            b = self._build_batch(paths, strike, lam, noise).to(self.device, self.dtype)
            lc, la = self._train_episodes(b)
            self.episodes_seen += batch
            row = {"iter": it, "episodes": self.episodes_seen, "critic_loss": lc, "actor_loss": la,
                   "seconds": _time.perf_counter() - start}
            history.append(row)
            if log_every and (it % log_every == 0 or it == n_iters - 1):
                print(f"iter {it:5d}  episodes {self.episodes_seen:9d}  critic {lc:.4e}  "
                      f"actor {la:.4e}  {row['seconds']:.1f}s", flush=True)
            if max_seconds is not None and row["seconds"] >= max_seconds:
                break
        return history

    def _train_episodes(self, b: _Batch) -> tuple[float, float]:
        n = b.n_steps
        lam = b.lam
        h = torch.zeros_like(b.spot[:, 0])
        critic_params = list(self.critic.parameters())
        lc_sum = torch.zeros((), device=self.device, dtype=self.dtype)
        la_sum = torch.zeros((), device=self.device, dtype=self.dtype)
        for t in range(n):
            s_t = b.spot[:, t]
            a = self._trade(b, t, h)

            # Critic: y_t = r_t + Vbar(s_{t+1}) along the executed trade.
            with torch.no_grad():
                a_exec = a + b.noise[:, t]
                h_next = h + a_exec
                y = step_reward(a_exec, s_t, self.eps) + self._value(self.target, b, t + 1, h_next)
            v = self._value(self.critic, b, t, h)
            lc = critic_loss(y - v, lam)
            self.opt_critic.zero_grad(set_to_none=True)
            lc.backward()
            self.opt_critic.step()

            # Actor: differentiate r_t + V(s_{t+1}) through the trade and the next
            # holding. The critic is frozen here so no weight gradients are
            # computed for it.
            for p in critic_params:
                p.requires_grad_(False)
            x = step_reward(a, s_t, self.eps) + self._value(self.critic, b, t + 1, h + a)
            la = actor_loss(x, v.detach(), lam)
            self.opt_actor.zero_grad(set_to_none=True)
            la.backward()
            self.opt_actor.step()
            for p in critic_params:
                p.requires_grad_(True)

            soft_update(self.target, self.critic, self.tau)
            h = h_next.detach()
            lc_sum = lc_sum + lc.detach()
            la_sum = la_sum + la.detach()
        return float(lc_sum) / n, float(la_sum) / n

    # ------------------------------------------------------------ policy use

    @torch.no_grad()
    def act(self, state: HedgeState, deterministic: bool = True,
            generator: torch.Generator | None = None) -> torch.Tensor:
        """Trade a_t (units of spot bought) at a batch of states, in float64."""
        if self.normalizer is None:
            raise RuntimeError("fit the normaliser (or train) before acting")
        x = self.normalizer(raw_features(state, self.notional, self.vol_floor).to(torch.float64))
        out = self.actor(_to_net(x, self.device, self.dtype)).squeeze(-1).cpu().to(torch.float64)
        trade = -portfolio_delta(state, self.notional, self.vol_floor) + self.action_scale * out
        if not deterministic and self.explore > 0.0:
            trade = trade + self.explore * self.notional * torch.randn(
                trade.shape, generator=generator, dtype=torch.float64)
        return trade

    @torch.no_grad()
    def value(self, state: HedgeState) -> torch.Tensor:
        """V(s) = B(s) + V_phi(s), the certainty equivalent of the remaining rewards."""
        if self.normalizer is None:
            raise RuntimeError("fit the normaliser (or train) before valuing states")
        x = self.normalizer(raw_features(state, self.notional, self.vol_floor).to(torch.float64))
        resid = self.critic(_to_net(x, self.device, self.dtype)).squeeze(-1).cpu().to(torch.float64)
        vol = implied_vol(state.var, self.vol_floor)
        book = state.h_prev * state.spot - self.notional * call_price(
            state.spot, state.strike, state.tau, vol)
        # At maturity V(s_T) = r_T exactly, as in training (_value); the network
        # was never fitted there and its Greek inputs are degenerate at tau = 0.
        resid = torch.where(state.tau > 0.0, resid, torch.zeros_like(resid))
        return book + self.value_scale * resid

    @torch.no_grad()
    def rollout_holdings(self, paths: MarketPaths, strikes=1.0, lam=0.1,
                         chunk: int = 65536) -> torch.Tensor:
        """Deterministic holdings h_0..h_{n-1}, [N, n, 1] float64 on CPU, ready for
        dhedge.pnl.hedge_pnl. `lam` is a scalar or one value per path; `strikes`
        a scalar or one value per path."""
        if self.normalizer is None:
            raise RuntimeError("fit the normaliser (or train) before rolling out")
        N = paths.n_paths
        strike = torch.as_tensor(strikes, dtype=torch.float64).reshape(-1).expand(N).clone()
        lam_t = _as_lam(lam, N)
        out = []
        for lo in range(0, N, chunk):
            sl = slice(lo, min(N, lo + chunk))
            sub = MarketPaths(paths.times, paths.spot[sl], paths.instruments[sl],
                              None if paths.var is None else paths.var[sl])
            b = self._build_batch(sub, strike[sl], lam_t[sl])
            h = torch.zeros(b.spot.shape[0], dtype=torch.float64)
            hs = []
            for t in range(b.n_steps):
                pi = self.actor(_to_net(self._x(b, t, h), self.device, self.dtype))
                pi = pi.squeeze(-1).cpu().to(torch.float64)
                h = h + (b.hstar[:, t] - h + self.action_scale * pi)
                hs.append(h)
            out.append(torch.stack(hs, dim=1))
        return torch.cat(out, dim=0).unsqueeze(-1)
