"""Heston pricing, sensitivities and the 2019 model hedge.

The reference is Lewis's formula on its original real-line contour, integrated
with scipy.integrate.quad and a separate numpy characteristic function, so it
shares neither the residue step nor the deformed contour with the code under
test.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad

import dhedge.heston_pricing as hp
from dhedge.bs import call_price
from dhedge.heston import heston_paths
from dhedge.heston_pricing import (heston_call, heston_delta, heston_dv, heston_greeks,
                                   model_hedge_positions, variance_swap_dldv)
from dhedge.pnl import call_payoff, hedge_pnl
from dhedge.spec import BUEHLER_2019, MURRAY_2022, T_30D

MURRAY_V0 = dataclasses.replace(MURRAY_2022, v0=MURRAY_2022.theta)

F64 = torch.float64


def _char_np(w, v, tau, p):
    """Little-trap characteristic function written out independently in numpy."""
    beta = p.kappa - p.rho * p.xi * 1j * w
    d = np.sqrt(beta * beta + p.xi ** 2 * (1j * w + w * w))
    g = (beta - d) / (beta + d)
    e = np.exp(-d * tau)
    C = p.kappa * p.theta / p.xi ** 2 * ((beta - d) * tau - 2.0 * np.log((1 - g * e) / (1 - g)))
    D = (beta - d) / p.xi ** 2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v)


def lewis_quad(S, v, tau, K, p):
    """C = S - sqrt(SK)/pi int_0^inf Re[e^{iuX} phi(u - i/2)] / (u^2 + 1/4) du."""
    X = math.log(S / K)

    def f(u):
        return (np.exp(1j * u * X) * _char_np(u - 0.5j, v, tau, p)).real / (u * u + 0.25)

    total, lo = 0.0, 0.0
    for hi in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6):
        val, _ = quad(f, lo, hi, limit=2000, epsabs=1e-14, epsrel=1e-13)
        total, lo = total + val, hi
    return S - math.sqrt(S * K) / math.pi * total


def test_buehler_2019_risk_neutral_price(record_property):
    """Buehler et al. (2019), Section 5.2: q = 1.69 for the 30-day ATM call."""
    q = heston_call(100.0, 0.04, T_30D, 100.0, BUEHLER_2019)
    print(f"q = {q:.4f}")
    record_property("q", q)
    assert round(q, 2) == 1.69
    assert q == pytest.approx(lewis_quad(100.0, 0.04, T_30D, 100.0, BUEHLER_2019), rel=1e-10)


GRID = [(BUEHLER_2019, 100.0, (0.01, 0.04, 0.16)), (MURRAY_V0, 1.0, (0.002, 0.00625, 0.025))]


# quad warns on the far tail pieces of a few short-dated, low-variance points;
# the reference still has to agree with the contour to 1e-6 there.
@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize("params,s0,vs", GRID, ids=["buehler2019", "murray2022"])
def test_price_matches_quad_reference(params, s0, vs, record_property):
    """Relative error below 1e-6 wherever the price exceeds 1e-6 S.

    Below that level the real-line reference itself cancels S against an
    integral of size S, so it only carries about 1e-13 S of absolute accuracy;
    those points are held to that absolute bound instead.
    """
    worst = 0.0
    n_rel = 0
    for tau in (1 / 365, 7 / 365, T_30D, 1.0):
        for v in vs:
            for m in (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25):
                K = s0 * m
                ref = lewis_quad(s0, v, tau, K, params)
                got = heston_call(s0, v, tau, K, params)
                if ref > 1e-6 * s0:
                    rel = abs(got - ref) / ref
                    worst, n_rel = max(worst, rel), n_rel + 1
                    assert rel < 1e-6, (tau, v, m, got, ref)
                else:
                    assert abs(got - ref) < 1e-11 * s0, (tau, v, m, got, ref)
    record_property("worst_rel_error", worst)
    print(f"worst relative error over {n_rel} points: {worst:.2e}")
    assert n_rel >= 60


def test_vectorised_matches_scalar_and_types():
    S = torch.tensor([90.0, 100.0, 110.0], dtype=F64)
    v = torch.tensor([0.01, 0.04, 0.09], dtype=F64)
    tau = torch.tensor([T_30D, 0.5, 1 / 365], dtype=F64)
    vec = heston_call(S, v, tau, 100.0, BUEHLER_2019)
    assert isinstance(vec, torch.Tensor) and vec.dtype == torch.float64
    for i in range(3):
        one = heston_call(float(S[i]), float(v[i]), float(tau[i]), 100.0, BUEHLER_2019)
        assert isinstance(one, float)
        assert float(vec[i]) == pytest.approx(one, rel=1e-13)
    grid = heston_call(np.array([[90.0], [110.0]]), 0.04, np.array([T_30D, 1.0]), 100.0, BUEHLER_2019)
    assert isinstance(grid, np.ndarray) and grid.shape == (2, 2)


def test_expiry_and_bounds():
    S = torch.tensor([80.0, 100.0, 120.0], dtype=F64)
    assert torch.equal(heston_call(S, 0.04, 0.0, 100.0, BUEHLER_2019), torch.tensor([0.0, 0.0, 20.0], dtype=F64))
    assert torch.equal(heston_delta(S, 0.04, 0.0, 100.0, BUEHLER_2019), torch.tensor([0.0, 0.0, 1.0], dtype=F64))
    S = torch.linspace(50.0, 200.0, 61, dtype=F64)
    c = heston_call(S, 0.04, T_30D, 100.0, BUEHLER_2019)
    assert torch.all(c >= torch.clamp(S - 100.0, min=0.0) - 1e-12) and torch.all(c <= S)


def test_black_scholes_limit():
    """xi -> 0 with v = theta: V stays at theta, so the price is Black-Scholes at sqrt(theta)."""
    for base, s0 in ((BUEHLER_2019, 100.0), (MURRAY_V0, 1.0)):
        p = dataclasses.replace(base, xi=1e-7)
        th = p.theta
        S = torch.tensor([0.9, 1.0, 1.1], dtype=torch.float64) * s0
        for tau in (1 / 365, T_30D, 1.0):
            h = heston_call(S, th, tau, s0, p)
            bs = call_price(S, s0, tau, math.sqrt(th))
            assert torch.allclose(h, bs, rtol=1e-6, atol=1e-12 * s0), (tau, h, bs)


def test_price_monotone_in_s_k_tau_v():
    p = BUEHLER_2019
    S = torch.linspace(80.0, 120.0, 81, dtype=F64)
    assert torch.all(torch.diff(heston_call(S, 0.04, T_30D, 100.0, p)) > 0)
    K = torch.linspace(80.0, 120.0, 81, dtype=F64)
    assert torch.all(torch.diff(heston_call(100.0, 0.04, T_30D, K, p)) < 0)
    tau = torch.linspace(1 / 365, 1.0, 60, dtype=F64)
    assert torch.all(torch.diff(heston_call(100.0, 0.04, tau, 100.0, p)) > 0)
    v = torch.linspace(0.0, 0.5, 60, dtype=F64)
    assert torch.all(torch.diff(heston_call(100.0, v, T_30D, 100.0, p)) > 0)
    # The two contours (X >= 0 and X < 0) must join continuously at the money.
    s = torch.tensor([100.0 * (1 - 1e-9), 100.0, 100.0 * (1 + 1e-9)], dtype=F64)
    c = heston_call(s, 0.04, T_30D, 100.0, p)
    d = float(heston_delta(100.0, 0.04, T_30D, 100.0, p))
    assert float(c[1] - c[0]) == pytest.approx(d * 1e-7, rel=1e-4)
    assert float(c[2] - c[1]) == pytest.approx(d * 1e-7, rel=1e-4)


CASES = [(BUEHLER_2019, 100.0, 0.04, T_30D, 100.0), (BUEHLER_2019, 95.0, 0.01, 7 / 365, 100.0),
         (BUEHLER_2019, 108.0, 0.2, 0.5, 100.0), (BUEHLER_2019, 100.3, 1e-4, 1 / 365, 100.0),
         (MURRAY_V0, 1.0, 0.00625, T_30D, 1.0), (MURRAY_V0, 0.97, 0.02, 0.25, 1.0)]


@pytest.mark.parametrize("p,S,v,tau,K", CASES)
def test_delta_and_dv_match_finite_differences(p, S, v, tau, K):
    g = heston_greeks(S, v, tau, K, p)

    def richardson(f, x, h):
        # Central differences at h and h/2 combined to cancel the h^2 term;
        # plain central differences are too coarse where xi = 2 bends the smile.
        d1 = (f(x + h) - f(x - h)) / (2 * h)
        d2 = (f(x + h / 2) - f(x - h / 2)) / h
        return (4 * d2 - d1) / 3

    # Steps scaled to the width of the return distribution, which is only a
    # few basis points one day out at low variance.
    width = math.sqrt((v + p.theta) * tau)
    fd_s = richardson(lambda s: heston_call(s, v, tau, K, p), S, 1e-2 * width * S)
    fd_v = richardson(lambda w: heston_call(S, w, tau, K, p), v, 1e-2 * v)
    assert g["delta"] == pytest.approx(fd_s, rel=1e-7, abs=1e-10)
    assert g["dv"] == pytest.approx(fd_v, rel=1e-6, abs=1e-10 * S)
    assert g["price"] == pytest.approx(heston_call(S, v, tau, K, p), rel=1e-14)
    assert heston_delta(S, v, tau, K, p) == pytest.approx(g["delta"], rel=1e-14)
    assert heston_dv(S, v, tau, K, p) == pytest.approx(g["dv"], rel=1e-14)


def test_contour_converged_in_the_hard_corner(monkeypatch):
    """v = 0 six hours before expiry, where the real-line integral is hopeless.

    Halving the step and lengthening the contour must not move anything.
    """
    S = torch.tensor([90.0, 99.0, 99.9, 100.0, 100.1, 101.0, 110.0], dtype=F64)
    args = (S, 0.0, 0.25 / 365, 100.0, BUEHLER_2019)
    base = heston_greeks(*args)
    monkeypatch.setattr(hp, "_ZETA", 0.025)
    monkeypatch.setattr(hp, "_Y_MAX", 24.0)
    monkeypatch.setattr(hp, "_TAIL_TOL", 1e-30)
    fine = heston_greeks(*args)
    for k in ("price", "delta", "dv"):
        assert torch.allclose(base[k], fine[k], rtol=0.0, atol=1e-12), k
    assert torch.all(base["delta"] >= -1e-12) and torch.all(base["delta"] <= 1 + 1e-12)


def test_model_hedge_positions_match_pointwise_greeks():
    p = heston_paths(BUEHLER_2019, 300, with_variance_swap=True, seed=4)
    pos = model_hedge_positions(p, BUEHLER_2019, 100.0, T_30D)
    assert pos.shape == (300, 30, 2) and pos.dtype == torch.float64
    for k in (0, 13, 29):
        tau = T_30D - float(p.times[k])
        g = heston_greeks(p.spot[:, k], p.var[:, k], tau, 100.0, BUEHLER_2019)
        assert torch.allclose(pos[:, k, 0], g["delta"], rtol=0, atol=1e-10)
        dv = g["dv"] / variance_swap_dldv(BUEHLER_2019, tau)
        assert torch.allclose(pos[:, k, 1], dv, rtol=1e-8, atol=1e-8)


def test_model_hedge_error_shrinks_with_rebalancing(record_property):
    """Zero costs, Buehler 2019 market: the model hedge replicates as n grows.

    One 240-step fine path set is recorded at 15, 30, 60 and 120 dates, so the
    four hedges see the same underlying paths.
    """
    q = heston_call(100.0, 0.04, T_30D, 100.0, BUEHLER_2019)
    stds = []
    for n in (15, 30, 60, 120):
        p = heston_paths(BUEHLER_2019, 4000, n_steps=n, substeps=240 // n,
                         with_variance_swap=True, seed=17)
        pos = model_hedge_positions(p, BUEHLER_2019, 100.0, T_30D)
        pnl = hedge_pnl(p.instruments, pos, call_payoff(p.spot, 100.0), premium=q)
        stds.append(float(pnl.std()))
        print(f"n = {n:4d}: hedge error mean {float(pnl.mean()):+.4f}, std {stds[-1]:.4f}")
    record_property("model_hedge_std", stds)
    assert all(a > b for a, b in zip(stds, stds[1:])), stds
    assert stds[-1] < 0.6 * stds[0]


def test_long_maturity_past_the_moment_explosion():
    """The 2019 parameters make E[S^-1/2] explode at tau = 2.05; prices must not notice."""
    t_star = hp.moment_explosion_time(-0.5, BUEHLER_2019)
    assert t_star == pytest.approx(2.045, abs=1e-3)
    assert math.isinf(hp.moment_explosion_time(1.5, BUEHLER_2019))
    # D(w) at w = i/2 is the exponent's coefficient for E[S^-1/2]; it blows up at T*.
    w = torch.tensor([0.5j], dtype=torch.complex128)
    _, d_near = hp.char_exponents(w, t_star * (1 - 1e-7), BUEHLER_2019)
    _, d_half = hp.char_exponents(w, 0.5 * t_star, BUEHLER_2019)
    assert abs(complex(d_near[0])) > 1e5 and abs(complex(d_half[0])) < 10.0
    for tau in (1.5, 2.5, 5.0):
        for m in (0.8, 1.0, 1.25):
            ref = lewis_quad(100.0, 0.04, tau, 100.0 * m, BUEHLER_2019)
            got = heston_call(100.0, 0.04, tau, 100.0 * m, BUEHLER_2019)
            assert got == pytest.approx(ref, rel=1e-9), (tau, m)


def _gil_pelaez(j, S, v, tau, K, p):
    """P_j of Heston (1993), in his own (u_j, b_j) parametrisation, by quad.

    Shares no formula with the Lewis contour: at zero rates C = S P1 - K P2 and
    delta = P1 exactly, which checks delta without finite differences.
    """
    uj, bj = (0.5, p.kappa - p.rho * p.xi) if j == 1 else (-0.5, p.kappa)
    x = math.log(S / K)

    def f(u):
        a = bj - p.rho * p.xi * 1j * u
        d = np.sqrt(a * a - p.xi ** 2 * (2 * uj * 1j * u - u * u))
        g = (a - d) / (a + d)
        e = np.exp(-d * tau)
        C = p.kappa * p.theta / p.xi ** 2 * ((a - d) * tau - 2 * np.log((1 - g * e) / (1 - g)))
        D = (a - d) / p.xi ** 2 * (1 - e) / (1 - g * e)
        return (np.exp(1j * u * x + C + D * v) / (1j * u)).real

    total, lo = 0.0, 1e-12
    for hi in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6):
        total, lo = total + quad(f, lo, hi, limit=4000, epsabs=1e-15, epsrel=1e-13)[0], hi
    return 0.5 + total / math.pi


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize("p,S,v,tau,K", [(BUEHLER_2019, 100.0, 0.04, T_30D, 100.0),
                                         (BUEHLER_2019, 100.0, 0.005, 3 / 365, 97.0),
                                         (BUEHLER_2019, 100.0, 0.3, 0.5, 120.0),
                                         (MURRAY_V0, 1.0, 0.001, T_30D, 1.03)])
def test_price_and_delta_match_heston_1993_probabilities(p, S, v, tau, K):
    P1, P2 = _gil_pelaez(1, S, v, tau, K, p), _gil_pelaez(2, S, v, tau, K, p)
    assert heston_call(S, v, tau, K, p) == pytest.approx(S * P1 - K * P2, rel=1e-8)
    assert heston_delta(S, v, tau, K, p) == pytest.approx(P1, abs=1e-10)


# A fixed 0.4 opening angle left the decay cone |arg w| < arccos|rho| for
# |rho| > 0.92: these points gave NaN (rho = -0.95 OTM, rho = +0.95 ATM) or
# were off by 1.8e-5 in price and 1.4e-5 in delta (rho = -0.9).
@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize("p,S,v,tau,K", [
    (dataclasses.replace(BUEHLER_2019, kappa=3.0, theta=0.1, xi=0.1, rho=-0.95), 100.0, 0.05, 0.1, 104.0),
    (dataclasses.replace(BUEHLER_2019, xi=0.5, rho=0.95), 100.0, 0.3, 0.75, 100.0),
    (dataclasses.replace(BUEHLER_2019, kappa=0.5, theta=0.2, xi=0.3, rho=-0.9),
     100.0, 0.10960713295319056, 0.8080975553211144, 102.66548055969),
    (dataclasses.replace(BUEHLER_2019, xi=1.0, rho=-0.99), 100.0, 0.05, 0.1, 104.0)])
def test_strong_correlation_stays_inside_the_decay_cone(p, S, v, tau, K):
    P1, P2 = _gil_pelaez(1, S, v, tau, K, p), _gil_pelaez(2, S, v, tau, K, p)
    assert heston_call(S, v, tau, K, p) == pytest.approx(S * P1 - K * P2, rel=1e-9)
    assert heston_delta(S, v, tau, K, p) == pytest.approx(P1, abs=1e-10)
    assert hp._omega(BUEHLER_2019) == pytest.approx(0.5 * math.acos(0.7))
    with pytest.raises(ValueError):
        heston_call(S, v, tau, K, dataclasses.replace(p, rho=-1.0))
