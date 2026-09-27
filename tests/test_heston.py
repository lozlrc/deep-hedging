"""Heston path simulation: exact CIR law, martingale property, MC vs analytic price.

The scheme freezes V inside each step, so S is a martingale only up to a
discretisation defect. The tests pin it at substeps=20 and report its size at
substeps=1 from the closed-form one-step drift, since Monte Carlo cannot
resolve a defect that small.
"""
from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from dhedge.heston import (cir_moments, heston_paths, spot_drift_defect,
                           variance_swap_drift_defect, variance_swap_fair_leg)
from dhedge.heston_pricing import heston_call
from dhedge.spec import BUEHLER_2019, MURRAY_2022, N_STEPS, T_30D

MURRAY_V0 = dataclasses.replace(MURRAY_2022, v0=MURRAY_2022.theta)  # stand-in until calibrated


def _mean_se(x: torch.Tensor) -> tuple[float, float]:
    return float(x.mean()), float(x.std()) / math.sqrt(x.numel())


@pytest.fixture(scope="module")
def fine_paths():
    return heston_paths(BUEHLER_2019, 200_000, substeps=20, with_variance_swap=True, seed=11)


@pytest.fixture(scope="module")
def coarse_paths():
    return heston_paths(BUEHLER_2019, 1_000_000, substeps=1, with_variance_swap=True, seed=12)


def test_shapes_grid_and_initial_values():
    p = heston_paths(BUEHLER_2019, 64, with_variance_swap=True, seed=0)
    assert p.spot.shape == (64, N_STEPS + 1) and p.instruments.shape == (64, N_STEPS + 1, 2)
    assert p.var.shape == (64, N_STEPS + 1) and p.spot.dtype == torch.float64
    assert torch.allclose(p.times, torch.linspace(0.0, T_30D, N_STEPS + 1, dtype=torch.float64))
    assert torch.all(p.spot[:, 0] == 100.0) and torch.all(p.var[:, 0] == 0.04)
    assert torch.equal(p.instruments[..., 0], p.spot)
    s2_0 = variance_swap_fair_leg(BUEHLER_2019, T_30D, 0.04)
    assert float(p.instruments[0, 0, 1]) == pytest.approx(s2_0, rel=1e-14)
    assert float(s2_0) == pytest.approx(0.04 * T_30D, rel=1e-14)       # v0 = theta
    # S2 at expiry is the realised integrated variance, the swap's payoff.
    assert torch.all(p.var >= 0.0) and torch.all(p.instruments[:, -1, 1] >= 0.0)
    one = heston_paths(BUEHLER_2019, 8, seed=0)
    assert one.instruments.shape == (8, N_STEPS + 1, 1) and one.var is not None


def test_seed_reproducible_and_dtype():
    a = heston_paths(BUEHLER_2019, 100, seed=5)
    b = heston_paths(BUEHLER_2019, 100, seed=5, dtype=torch.float32)
    assert b.spot.dtype == torch.float32
    assert torch.allclose(a.spot.float(), b.spot)


def test_substeps_record_the_same_fine_path():
    """n_steps * substeps fixed with one seed gives the same fine path, sampled coarser."""
    a = heston_paths(BUEHLER_2019, 500, n_steps=60, substeps=2, with_variance_swap=True, seed=9)
    b = heston_paths(BUEHLER_2019, 500, n_steps=30, substeps=4, with_variance_swap=True, seed=9)
    assert torch.allclose(a.instruments[:, ::2], b.instruments, rtol=1e-13, atol=1e-15)
    assert torch.allclose(a.var[:, ::2], b.var, rtol=1e-13, atol=0.0)


def test_v0_required_for_murray():
    with pytest.raises(ValueError):
        heston_paths(MURRAY_2022, 10)


@pytest.mark.parametrize("params,substeps", [(BUEHLER_2019, 1), (BUEHLER_2019, 5), (MURRAY_V0, 1)])
def test_cir_moments_exact(params, substeps):
    """E[V_t] and Var[V_t] at several dates against the closed forms, within 4 SE."""
    n = 400_000
    p = heston_paths(params, n, substeps=substeps, seed=21)
    for k in (1, 10, N_STEPS):
        v = p.var[:, k]
        mean, var = cir_moments(params, params.v0, float(p.times[k]))
        m, se_m = _mean_se(v)
        c = v - v.mean()
        s2 = float((c * c).mean())
        se_var = math.sqrt(float((c ** 4).mean()) - s2 * s2) / math.sqrt(n)
        assert abs(m - mean) < 4.0 * se_m, (k, m, mean, se_m)
        assert abs(s2 - var) < 4.0 * se_var, (k, s2, var, se_var)


def test_spot_and_swap_are_martingales_at_substeps_20(fine_paths):
    p = fine_paths
    m, se = _mean_se(p.spot[:, -1])
    assert abs(m - 100.0) < 3.0 * se, (m, se)
    for k in (10, 20):
        mk, sek = _mean_se(p.spot[:, k])
        assert abs(mk - 100.0) < 3.0 * sek
    d = p.instruments[:, -1, 1] - p.instruments[:, 0, 1]
    m2, se2 = _mean_se(d)
    assert abs(m2) < 3.0 * se2, (m2, se2)


def test_drift_defect_formula_matches_one_big_step():
    """The closed-form one-step defect is checked where it is large enough to see."""
    p = dataclasses.replace(BUEHLER_2019, v0=1.0)
    h, n = 0.25, 2_000_000
    # The defect formula describes the paper's continuous fair leg; the default
    # discrete leg is an exact martingale and has no defect to measure.
    paths = heston_paths(p, n, T=2 * h, n_steps=2, seed=3, with_variance_swap=True,
                         variance_swap="continuous")
    r, se = _mean_se(paths.spot[:, 1] / paths.spot[:, 0])
    d = float(spot_drift_defect(p, 1.0, h))
    assert d < -0.03                                 # a genuine, visible bias at this step size
    assert abs((r - 1.0) - d) < 4.0 * se, (r - 1.0, d, se)
    inc, se2 = _mean_se(paths.instruments[:, 1, 1] - paths.instruments[:, 0, 1])
    d2 = float(variance_swap_drift_defect(p, 1.0, h))
    assert abs(inc - d2) < 4.0 * se2, (inc, d2, se2)


def test_martingale_bias_at_substeps_1_is_reported(coarse_paths, fine_paths, record_property):
    """Accumulated E[S_T]/S0 - 1 implied by the exact one-step defect along the V paths.

    Reported, not hidden: the number is printed and recorded. It must stay far
    below the Monte Carlo noise of any experiment in this repo, and shrink by
    roughly the refinement factor at substeps=20 (the defect is O(h^2) per step).
    """
    out = {}
    for name, p, sub in (("substeps=1", coarse_paths, 1), ("substeps=20", fine_paths, 20)):
        h = T_30D / (N_STEPS * sub)
        v = p.var[:, :-1].numpy()                  # V at the recorded dates only
        # With substeps the defect accrues on every fine step; the recorded
        # dates are an unbiased sample of the fine V path for this estimate.
        acc = float(spot_drift_defect(BUEHLER_2019, v, h).sum(axis=1).mean()) * sub
        m, se = _mean_se(p.spot[:, -1])
        out[name] = (acc, (m - 100.0) / 100.0, se / 100.0)
        record_property(f"martingale_defect_{name}", acc)
        print(f"{name}: implied E[S_T]/S0-1 = {acc:.3e}; MC {(m - 100) / 100:.3e} +- {se / 100:.1e}")
    acc1, mc1, se1 = out["substeps=1"]
    acc20 = out["substeps=20"][0]
    assert abs(acc1) < 1e-4 and abs(acc1) < se1           # invisible at 1e6 paths
    assert abs(mc1 - acc1) < 4.0 * se1
    assert abs(acc20) < abs(acc1) / 10.0


def test_mc_call_matches_analytic_at_substeps_20(fine_paths, coarse_paths, record_property):
    q = heston_call(100.0, 0.04, T_30D, 100.0, BUEHLER_2019)
    pay = torch.clamp(fine_paths.spot[:, -1] - 100.0, min=0.0)
    m, se = _mean_se(pay)
    print(f"substeps=20: MC {m:.5f} +- {se:.5f}, analytic {q:.5f}, z = {(m - q) / se:.2f}")
    record_property("mc_call_substeps_20", (m, se, q))
    assert abs(m - q) < 3.0 * se

    # substeps=1 with the spot as a control variate, for a tighter bias estimate.
    s = coarse_paths.spot[:, -1]
    pay1 = torch.clamp(s - 100.0, min=0.0)
    cov = float(((pay1 - pay1.mean()) * (s - s.mean())).mean())
    beta = cov / float(s.var())
    cv = pay1 - beta * (s - 100.0)
    m1, se1 = _mean_se(cv)
    print(f"substeps=1: MC (control variate) {m1:.5f} +- {se1:.5f}, bias {m1 - q:+.5f}, z = {(m1 - q) / se1:.2f}")
    record_property("mc_call_substeps_1", (m1, se1, q))
    assert abs(m1 - q) < 0.02 * q           # reported above; bounded loosely, not hidden


# ------------------------------------------------ variance-swap arbitrage fix

def _step_increments(scheme, n=200_000, seed=5):
    from dhedge.heston import heston_paths
    from dhedge.spec import BUEHLER_2019
    p = heston_paths(BUEHLER_2019, n, seed=seed, with_variance_swap=True, variance_swap=scheme)
    dS2 = p.instruments[:, 1:, 1] - p.instruments[:, :-1, 1]
    return p, dS2


def test_continuous_fair_leg_has_a_deterministic_last_step():
    """Documents the arbitrage: with the paper's continuous L the final increment
    is a known function of V_{n-1}, so it carries no risk at all."""
    import math
    from dhedge.spec import BUEHLER_2019 as P
    p, dS2 = _step_increments("continuous", n=20_000)
    h = (30 / 365) / 30
    predicted = (p.var[:, -2] - P.theta) * (h - (1 - math.exp(-P.kappa * h)) / P.kappa)
    assert torch.allclose(dS2[:, -1], predicted, rtol=1e-6, atol=1e-15)


def test_discrete_fair_leg_last_step_is_not_predictable():
    """With the discrete fair leg the last increment has zero conditional mean:
    regressing it on V_{n-1} gives no slope beyond Monte Carlo noise."""
    p, dS2 = _step_increments("discrete")
    x = p.var[:, -2] - p.var[:, -2].mean()
    y = dS2[:, -1]
    slope = float((x * (y - y.mean())).sum() / (x * x).sum())
    resid = y - y.mean() - slope * x
    se = float(resid.std() / (x.std() * math.sqrt(len(x))))
    assert abs(slope / se) < 4.0
    assert float(y.std()) > 0.0


def test_discrete_variance_swap_is_a_martingale_every_step():
    """E[dS2_k | V_k] = 0 at every date, the property that rules out the arbitrage."""
    p, dS2 = _step_increments("discrete")
    for k in range(dS2.shape[1]):
        x = p.var[:, k] - p.var[:, k].mean()
        y = dS2[:, k]
        if float(x.std()) == 0.0:
            assert abs(float(y.mean())) < 4 * float(y.std()) / math.sqrt(len(y))
            continue
        slope = float((x * (y - y.mean())).sum() / (x * x).sum())
        resid = y - y.mean() - slope * x
        se = float(resid.std() / (x.std() * math.sqrt(len(x))))
        assert abs(slope / se) < 4.5, f"date {k}: t = {slope / se:.1f}"


def test_discrete_fair_leg_matches_continuous_as_h_shrinks():
    import numpy as np
    from dhedge.heston import variance_swap_fair_leg, variance_swap_fair_leg_discrete
    from dhedge.spec import BUEHLER_2019 as P
    tau, v = 0.05, 0.07
    cont = float(variance_swap_fair_leg(P, tau, v))
    for m in (10, 100, 10_000):
        disc = float(variance_swap_fair_leg_discrete(P, m, tau / m, v))
        assert abs(disc - cont) < 2.0 * P.kappa * (tau / m) * abs(cont) + 1e-12
    assert float(variance_swap_fair_leg_discrete(P, 0, 0.01, v)) == 0.0
