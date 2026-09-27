"""Pathwise deep hedging, Buehler, Gonon, Teichmann, Wood (2019), Sections 3 to 5.

The hedger is short a claim Z and holds delta_k of the instruments over
[t_k, t_{k+1}], chosen by neural networks from the information I_k. The loss is
a convex risk measure of

    PL = -Z + (delta . S)_T - C_T(delta)

computed by dhedge.pnl.hedge_pnl with zero premium, and its minimum over
strategies is the indifference price p0. Gradients flow straight through the
simulated P&L into every date's network, so this is direct policy
optimisation over whole paths, not dynamic programming.

A policy is any torch.nn.Module with forward(features [N, F], prev [N, d], k)
returning the holdings [N, d] for rebalancing date k. rollout_positions turns
one into the [N, n, d] position array hedge_pnl consumes, so learned policies
and closed-form baselines are scored by the same code.
"""
from __future__ import annotations

import time
from typing import Callable

import torch
from torch import nn

from .pnl import hedge_pnl
from .risk import cvar_oce, entropic_risk, risk_measure
from .spec import N_STEPS, MarketPaths

FeaturesFn = Callable[[MarketPaths, int], torch.Tensor]
PayoffFn = Callable[[MarketPaths], torch.Tensor]


# ----------------------------------------------------------------- features

def heston_features(paths: MarketPaths, k: int) -> torch.Tensor:
    """I_k = (log S1_k, V_k), the information set of the paper's Heston experiments."""
    if paths.var is None:
        raise ValueError("heston_features needs the variance path")
    return torch.stack([torch.log(paths.spot[:, k]), paths.var[:, k]], dim=-1)


def bs_features(paths: MarketPaths, k: int) -> torch.Tensor:
    """I_k = (log S_k,). In Black-Scholes the spot is a sufficient state."""
    return torch.log(paths.spot[:, k]).unsqueeze(-1)


# ----------------------------------------------------------------- policies

class PerStepNet(nn.Module):
    """The 2019 architecture: an independent network F_k for every date k.

    Each F_k has two hidden layers of width d + 15 with BatchNorm right before
    each ReLU and a linear output of size d. The semi-recurrent variant feeds
    the previous holdings delta_{k-1} alongside I_k, which the policy needs
    once trading costs make the optimal position path dependent. With
    semi_recurrent=False it is the "simpler" network of Figures 4 and 5.
    """

    def __init__(self, n_steps: int, feat_dim: int, d: int, semi_recurrent: bool = True,
                 deterministic_start: bool = True):
        super().__init__()
        self.n_steps, self.feat_dim, self.d = n_steps, feat_dim, d
        self.semi_recurrent = semi_recurrent
        n_in = feat_dim + d if semi_recurrent else feat_dim
        width = d + 15

        def norm(k: int) -> nn.Module:
            # When every path starts from the same state, I_0 is one point and
            # BatchNorm at date 0 sees zero batch variance. In training it then
            # outputs its shift beta whatever the weights, so it adds nothing,
            # but in eval mode it divides float rounding by sqrt(1e-5) and the
            # error is amplified again by every later BatchNorm through
            # delta_0. Date 0 still learns a constant either way.
            if k == 0 and deterministic_start:
                return nn.Identity()
            return nn.BatchNorm1d(width)

        self.nets = nn.ModuleList(
            nn.Sequential(
                nn.Linear(n_in, width), norm(k), nn.ReLU(),
                nn.Linear(width, width), norm(k), nn.ReLU(),
                nn.Linear(width, d),
            )
            for k in range(n_steps)
        )
        self.register_buffer("feat_mean", torch.zeros(n_steps, feat_dim))
        self.register_buffer("feat_std", torch.ones(n_steps, feat_dim))

    @torch.no_grad()
    def fit_normalization(self, paths: MarketPaths, features_fn: "FeaturesFn") -> None:
        """Fix a per-date affine rescaling of I_k to zero mean and unit variance.

        Daily log S moves by about 1 percent while delta_{k-1} spans tens of
        percent, so in raw units the semi-recurrent input is dominated by the
        previous holding and Adam needs thousands of steps just to grow the
        spot weight. The rescaling is fixed, not learned, so train and eval
        see the same map.
        """
        for k in range(self.n_steps):
            f = features_fn(paths, k).cpu().to(torch.float64)
            std = f.std(dim=0)
            self.feat_mean[k] = f.mean(dim=0).to(self.feat_mean.dtype)
            # a deterministic feature (every I_0) is only centred
            self.feat_std[k] = torch.where(std > 1e-12, std, torch.ones_like(std)).to(self.feat_std.dtype)

    def forward(self, features: torch.Tensor, prev: torch.Tensor, k: int) -> torch.Tensor:
        z = (features - self.feat_mean[k].to(features.dtype)) / self.feat_std[k].to(features.dtype)
        x = torch.cat([z, prev], dim=-1) if self.semi_recurrent else z
        return self.nets[k](x)


_ACTIVATIONS = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU, "tanh": nn.Tanh}


class SharedNet(nn.Module):
    """One network across all dates, as in Murray et al. (2022), Table 1.

    That comparison used a three layer feedforward network with 256 units per
    layer and sigmoid activations. Sharing weights needs the date as an input,
    supplied as the fraction of the horizon left, 1 - k / n_steps, which is
    time to maturity on the uniform grid both papers use. There is no
    BatchNorm here, so raw features such as log(100) are standardised with
    feat_mean and feat_std (given, or set by fit_normalization) to keep the
    sigmoids out of saturation.
    """

    def __init__(self, feat_dim: int, d: int, hidden: tuple[int, ...] = (256, 256, 256),
                 activation: str = "sigmoid", time_feature: bool = True,
                 semi_recurrent: bool = False, n_steps: int = N_STEPS,
                 feat_mean: torch.Tensor | None = None, feat_std: torch.Tensor | None = None):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}")
        self.feat_dim, self.d, self.n_steps = feat_dim, d, n_steps
        self.time_feature, self.semi_recurrent = time_feature, semi_recurrent
        mean = torch.zeros(feat_dim) if feat_mean is None else torch.as_tensor(feat_mean).reshape(feat_dim)
        std = torch.ones(feat_dim) if feat_std is None else torch.as_tensor(feat_std).reshape(feat_dim)
        self.register_buffer("feat_mean", mean.float().clone())
        self.register_buffer("feat_std", std.float().clone())
        n_in = feat_dim + (1 if time_feature else 0) + (d if semi_recurrent else 0)
        layers: list[nn.Module] = []
        for width in hidden:
            layers += [nn.Linear(n_in, width), _ACTIVATIONS[activation]()]
            n_in = width
        layers.append(nn.Linear(n_in, d))
        self.net = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_normalization(self, paths: MarketPaths, features_fn: "FeaturesFn") -> None:
        """Standardise features with moments pooled over all dates, one map for the shared net."""
        f = torch.cat([features_fn(paths, k) for k in range(paths.n_steps)], dim=0).cpu().to(torch.float64)
        std = f.std(dim=0)
        self.feat_mean.copy_(f.mean(dim=0).to(self.feat_mean.dtype))
        self.feat_std.copy_(torch.where(std > 1e-12, std, torch.ones_like(std)).to(self.feat_std.dtype))

    def forward(self, features: torch.Tensor, prev: torch.Tensor, k: int) -> torch.Tensor:
        parts = [(features - self.feat_mean.to(features.dtype)) / self.feat_std.to(features.dtype)]
        if self.time_feature:
            ttm = 1.0 - k / self.n_steps
            parts.append(torch.full_like(features[:, :1], ttm))
        if self.semi_recurrent:
            parts.append(prev)
        return self.net(torch.cat(parts, dim=-1))


# ----------------------------------------------------------------- rollout

def _param_dtype_device(policy: nn.Module, paths: MarketPaths) -> tuple[torch.dtype, torch.device]:
    for t in policy.parameters():
        return t.dtype, t.device
    for t in policy.buffers():
        if t.is_floating_point():
            return t.dtype, t.device
    # closed-form baselines have no weights; they run at the paths' precision
    return paths.spot.dtype, paths.spot.device


def rollout_positions(policy: nn.Module, paths: MarketPaths, features_fn: FeaturesFn) -> torch.Tensor:
    """Holdings [N, n, d] from running the policy date by date, delta_{-1} = 0.

    Runs in whatever mode the policy is in: training keeps the graph, and
    callers scoring a trained policy should put it in eval mode first.
    """
    dtype, device = _param_dtype_device(policy, paths)
    n, d = paths.n_steps, paths.n_instruments
    prev = torch.zeros(paths.n_paths, d, dtype=dtype, device=device)
    out = []
    for k in range(n):
        feats = features_fn(paths, k).to(dtype=dtype).to(device)   # MPS has no float64
        prev = policy(feats, prev, k)
        out.append(prev)
    return torch.stack(out, dim=1)


def _subset(paths: MarketPaths, idx: torch.Tensor) -> MarketPaths:
    return MarketPaths(times=paths.times, spot=paths.spot[idx], instruments=paths.instruments[idx],
                       var=None if paths.var is None else paths.var[idx])


def _cast(paths: MarketPaths, dtype: torch.dtype, device: torch.device | str) -> MarketPaths:
    def c(t):
        return None if t is None else t.to(dtype=dtype).to(device)
    return MarketPaths(times=c(paths.times), spot=c(paths.spot),
                       instruments=c(paths.instruments), var=c(paths.var))


def _payoff_tensor(payoff: torch.Tensor | PayoffFn, paths: MarketPaths) -> torch.Tensor:
    return payoff(paths) if callable(payoff) else torch.as_tensor(payoff)


@torch.no_grad()
def recalibrate_batchnorm(policy: nn.Module, paths: MarketPaths, features_fn: FeaturesFn,
                          chunk: int = 8192) -> None:
    """Replace BatchNorm running statistics with exact averages under the final weights.

    The running averages accumulated during training lag the weights, and
    eval-mode BatchNorm divides that lag by the running standard deviation,
    which is tiny wherever the batch barely varies (early dates, where every
    path is still near S0). The error then compounds through delta_{k-1}.
    Averaging batch statistics under frozen weights removes the lag.
    """
    bns = [m for m in policy.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
    was_training = policy.training
    policy.train()
    seen = 0
    for start in range(0, paths.n_paths, chunk):
        idx = torch.arange(start, min(start + chunk, paths.n_paths), device=paths.spot.device)
        if idx.numel() < 2:
            continue
        # Weight each chunk by its size. momentum=None would average chunks
        # equally, letting a short trailing chunk move the variances by percents.
        for m in bns:
            m.momentum = idx.numel() / (seen + idx.numel())
        seen += idx.numel()
        rollout_positions(policy, _subset(paths, idx), features_fn)
    for m, mom in zip(bns, momenta):
        m.momentum = mom
    policy.train(was_training)


# ----------------------------------------------------------------- training

def train_pathwise(policy: nn.Module, sample_paths: Callable[[int, int], MarketPaths],
                   payoff_fn: PayoffFn, features_fn: FeaturesFn, risk: str = "cvar",
                   alpha: float | None = None, lam: float | None = None, eps: float = 0.0,
                   steps: int = 10_000, batch: int = 256, lr: float = 0.005,
                   liquidate: bool = True, seed: int = 0, log_every: int = 100,
                   n_pool: int = 100_000, device: str | torch.device = "cpu",
                   dtype: torch.dtype = torch.float32, recalibrate_bn: bool = True,
                   fit_normalization: bool = True) -> dict:
    """Minimise rho(hedge_pnl) over the policy with Adam on minibatches.

    Batches are drawn with replacement from a pool of n_pool paths generated
    once with sample_paths(n_pool, seed), the finite probability space of the
    paper. For CVaR the Rockafellar-Uryasev objective is trained with a
    learnable threshold w, initialised at the empirical alpha-VaR of the
    untrained policy's loss so the hedge does not wait for w to crawl there.
    Policies exposing fit_normalization get their fixed input rescaling set
    from the pool before training; pass fit_normalization=False to keep a
    rescaling the caller has already chosen.

    Returns a history dict with the logged step, mean loss since the previous
    log, w (CVaR only) and elapsed seconds.
    """
    if risk not in ("cvar", "entropic"):
        raise ValueError(f"unknown risk measure {risk!r}")
    if risk == "cvar" and alpha is None:
        raise ValueError("cvar needs alpha")
    if risk == "entropic" and lam is None:
        raise ValueError("entropic risk needs lam")

    pool64 = sample_paths(n_pool, seed)
    pool = _cast(pool64, dtype, device)
    payoff_pool = _payoff_tensor(payoff_fn, pool64).to(dtype=dtype).to(device)
    if fit_normalization and hasattr(policy, "fit_normalization"):
        policy.fit_normalization(pool64, features_fn)
    policy.to(device=device, dtype=dtype)
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 1)

    def draw() -> tuple[MarketPaths, torch.Tensor]:
        idx = torch.randint(0, pool.n_paths, (batch,), generator=gen).to(device)
        return _subset(pool, idx), payoff_pool[idx]

    def pnl_of(paths: MarketPaths, z: torch.Tensor) -> torch.Tensor:
        pos = rollout_positions(policy, paths, features_fn)
        return hedge_pnl(paths.instruments, pos, z, eps=eps, premium=0.0, liquidate=liquidate)

    params = list(policy.parameters())
    w = None
    if risk == "cvar":
        policy.train()
        with torch.no_grad():
            init_loss = -pnl_of(*draw())
            w0 = torch.quantile(init_loss.detach().cpu().double(), alpha)   # no MPS quantile
        w = torch.nn.Parameter(w0.to(dtype=dtype, device=device).reshape(()))
        params.append(w)
    opt = torch.optim.Adam(params, lr=lr)

    history: dict[str, list] = {"step": [], "loss": [], "w": [], "seconds": []}
    running, count, t0 = 0.0, 0, time.perf_counter()
    policy.train()
    for step in range(1, steps + 1):
        paths, z = draw()
        pl = pnl_of(paths, z)
        loss = cvar_oce(pl, w, alpha) if risk == "cvar" else entropic_risk(pl, lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running += float(loss.detach())
        count += 1
        if step % log_every == 0 or step == steps:
            history["step"].append(step)
            history["loss"].append(running / count)
            history["w"].append(None if w is None else float(w.detach()))
            history["seconds"].append(time.perf_counter() - t0)
            running, count = 0.0, 0

    if recalibrate_bn:
        recalibrate_batchnorm(policy, pool, features_fn)
    policy.eval()
    return history


# ----------------------------------------------------------------- evaluation

@torch.no_grad()
def indifference_price(policy: nn.Module, test_paths: MarketPaths, payoff: torch.Tensor | PayoffFn,
                       features_fn: FeaturesFn, risk: str = "cvar", alpha: float | None = None,
                       lam: float | None = None, eps: float = 0.0,
                       liquidate: bool = True) -> tuple[float, torch.Tensor]:
    """p0 = rho(-Z + (delta . S)_T - C_T(delta)) on held-out paths.

    CVaR is evaluated with the exact empirical estimator, never the learned w,
    so an under-trained threshold cannot flatter the result. The policy runs in
    eval mode and the P&L is computed in float64 whatever the training dtype.
    Returns (p0, per-path P&L [N] in float64).
    """
    was_training = policy.training
    policy.eval()
    pos = rollout_positions(policy, test_paths, features_fn)
    policy.train(was_training)
    inst = test_paths.instruments.cpu().to(torch.float64)
    z = _payoff_tensor(payoff, test_paths).cpu().to(torch.float64)
    pnl = hedge_pnl(inst, pos.cpu().to(torch.float64), z, eps=eps, premium=0.0, liquidate=liquidate)
    return float(risk_measure(pnl, risk, alpha=alpha, lam=lam)), pnl
