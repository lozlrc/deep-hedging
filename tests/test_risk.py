"""Risk measures: the axioms that make the minimal risk an indifference price,
the known limits, and the exactness of the empirical CVaR estimator."""
from __future__ import annotations

import pytest
import torch

from dhedge.risk import cvar, cvar_oce, entropic_risk, exp_utility, risk_measure


def _sample(n: int = 2001, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # skewed, heavy left tail: the shape of a short option's hedging P&L
    return torch.randn(n, generator=g, dtype=torch.float64) - 0.5 * torch.rand(n, generator=g, dtype=torch.float64) ** 4 * 10


MEASURES = [
    ("cvar_0", lambda X: cvar(X, 0.0)),
    ("cvar_0.5", lambda X: cvar(X, 0.5)),
    ("cvar_0.99", lambda X: cvar(X, 0.99)),
    ("cvar_0.3337", lambda X: cvar(X, 0.3337)),       # fractional tail count
    ("entropic_0.5", lambda X: entropic_risk(X, 0.5)),
    ("entropic_5", lambda X: entropic_risk(X, 5.0)),
]


@pytest.mark.parametrize("name,rho", MEASURES)
def test_cash_invariance(name, rho):
    X = _sample()
    for c in (-3.0, 0.7, 12.5):
        assert float(rho(X + c)) == pytest.approx(float(rho(X)) - c, abs=1e-10)


@pytest.mark.parametrize("name,rho", MEASURES)
def test_monotonicity(name, rho):
    X = _sample()
    g = torch.Generator().manual_seed(1)
    Y = X + torch.rand(X.shape, generator=g, dtype=torch.float64)      # Y >= X pathwise
    assert float(rho(Y)) <= float(rho(X)) + 1e-12


@pytest.mark.parametrize("name,rho", MEASURES)
def test_convexity_on_random_mixtures(name, rho):
    g = torch.Generator().manual_seed(2)
    for trial in range(20):
        X, Y = _sample(seed=10 + trial), 2.0 * _sample(seed=100 + trial) + 0.3
        t = float(torch.rand(1, generator=g))
        lhs = float(rho(t * X + (1 - t) * Y))
        rhs = t * float(rho(X)) + (1 - t) * float(rho(Y))
        assert lhs <= rhs + 1e-10


def test_cvar_at_zero_is_minus_mean():
    X = _sample()
    assert float(cvar(X, 0.0)) == pytest.approx(-float(X.mean()), abs=1e-12)


def test_cvar_at_one_is_worst_loss():
    X = _sample()
    assert float(cvar(X, 1.0)) == pytest.approx(-float(X.min()), abs=1e-12)
    assert float(cvar(X, 1.0 - 1e-9)) == pytest.approx(-float(X.min()), abs=1e-9)


def test_cvar_matches_quantile_integral_with_fractional_tail():
    """Paper eq. (5.1) integrated numerically against the estimator.

    With N = 7 and alpha = 0.5 the tail holds 3.5 scenarios, so the fourth
    largest loss must enter with weight one half.
    """
    X = torch.tensor([3.0, -1.0, 0.5, -4.0, 2.0, -2.5, 1.0], dtype=torch.float64)
    losses = torch.sort(-X, descending=True).values
    N, alpha = X.numel(), 0.5
    expected = (losses[:3].sum() + 0.5 * losses[3]) / 3.5
    assert float(cvar(X, alpha)) == pytest.approx(float(expected), abs=1e-14)

    # VaR_gamma is the loss exceeded with probability gamma: on the empirical
    # measure it is the ceil(gamma N)-th largest loss.
    grid = (torch.arange(200_000, dtype=torch.float64) + 0.5) / 200_000 * (1 - alpha)
    var_gamma = losses[torch.ceil(grid * N).long() - 1]
    assert float(cvar(X, alpha)) == pytest.approx(float(var_gamma.mean()), abs=1e-5)


def test_cvar_is_continuous_in_alpha():
    X = _sample(n=101)
    a = torch.linspace(0.0, 0.999, 2000)
    vals = torch.stack([cvar(X, float(x)) for x in a])
    assert float((vals[1:] - vals[:-1]).abs().max()) < 0.05
    assert bool((vals[1:] >= vals[:-1] - 1e-12).all())          # more tail, more risk


@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.9, 0.3337, 0.99])
def test_cvar_oce_minimum_equals_cvar(alpha):
    X = _sample()
    exact = float(cvar(X, alpha))
    # piecewise linear and convex in w with kinks at the sample losses, so the
    # minimum over the sample points is exact
    at_samples = torch.stack([cvar_oce(X, w, alpha) for w in (-X)])
    assert float(at_samples.min()) == pytest.approx(exact, abs=1e-10)
    lo, hi = float((-X).min()), float((-X).max())
    grid = torch.linspace(lo, hi, 20_001, dtype=torch.float64)
    on_grid = torch.stack([cvar_oce(X, w, alpha) for w in grid])
    assert float(on_grid.min()) == pytest.approx(exact, abs=5e-3)
    assert float(on_grid.min()) >= exact - 1e-10               # never below the infimum


def test_entropic_limits():
    X = _sample()
    assert float(entropic_risk(X, 1e-6)) == pytest.approx(-float(X.mean()), abs=1e-4)
    assert float(entropic_risk(X, 1e4)) == pytest.approx(-float(X.min()), abs=1e-3)
    lams = [0.01, 0.1, 1.0, 10.0]
    vals = [float(entropic_risk(X, lam)) for lam in lams]
    assert vals == sorted(vals)                                   # more aversion, more risk
    assert torch.isfinite(entropic_risk(X * 1e3, 1e3))            # logsumexp, no overflow


def test_exp_utility_is_negative_entropic():
    X = _sample()
    assert float(exp_utility(X, 2.0)) == pytest.approx(-float(entropic_risk(X, 2.0)), abs=1e-14)


def test_gradients_flow():
    X = _sample().requires_grad_(True)
    for val in (cvar(X, 0.5), entropic_risk(X, 1.0), cvar_oce(X, torch.tensor(0.3), 0.5)):
        (g,) = torch.autograd.grad(val, X)
        assert torch.isfinite(g).all() and float(g.abs().sum()) > 0
    # CVaR's gradient is minus the tail weights over the tail mass
    (g,) = torch.autograd.grad(cvar(X, 0.5), X)
    assert float(g.sum()) == pytest.approx(-1.0, abs=1e-12)


def test_dispatch_and_validation():
    X = _sample()
    assert float(risk_measure(X, "cvar", alpha=0.5)) == float(cvar(X, 0.5))
    assert float(risk_measure(X, "entropic", lam=1.0)) == float(entropic_risk(X, 1.0))
    with pytest.raises(ValueError):
        risk_measure(X, "var", alpha=0.5)
    with pytest.raises(ValueError):
        cvar(X, 1.5)
    with pytest.raises(ValueError):
        entropic_risk(X, 0.0)
