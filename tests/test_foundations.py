"""Tests for the shared foundations. Every method is scored through these
functions, so an error here would corrupt every comparison in the repo."""
from __future__ import annotations

import math

import pytest
import torch

from dhedge.bs import call_delta, call_price, greeks
from dhedge.markets import gbm_paths
from dhedge.pnl import call_payoff, hedge_pnl
from dhedge.spec import BUEHLER_2019, MURRAY_2022, T_30D

torch.set_default_dtype(torch.float64)


def test_call_price_matches_textbook_value():
    # S = K = 100, sigma = 0.2, tau = 1, r = 0: the standard value 7.9656.
    p = call_price(torch.tensor(100.0), 100.0, 1.0, 0.2)
    assert float(p) == pytest.approx(7.965567, abs=1e-5)


def test_call_price_limits():
    s = torch.tensor([80.0, 100.0, 120.0])
    near_expiry = call_price(s, 100.0, 1e-12, 0.2)
    assert torch.allclose(near_expiry, torch.clamp(s - 100.0, min=0.0), atol=1e-6)
    deep_itm = call_price(torch.tensor(1e6), 100.0, 0.5, 0.2)
    assert float(deep_itm) == pytest.approx(1e6 - 100.0, rel=1e-9)


@pytest.mark.parametrize("s,k,tau,vol", [(100.0, 100.0, 0.25, 0.2), (95.0, 105.0, T_30D, 0.3),
                                         (1.0, 0.9, T_30D, 0.25), (1.2, 1.0, 0.5, 0.15)])
def test_greeks_match_autograd(s, k, tau, vol):
    """Every closed-form Greek against automatic differentiation of the price."""
    S = torch.tensor(s, requires_grad=True)
    sig = torch.tensor(vol, requires_grad=True)
    t = torch.tensor(0.0, requires_grad=True)          # calendar time; tau shrinks as t grows
    price = call_price(S, k, tau - t, sig)
    dS, dsig, dt = torch.autograd.grad(price, (S, sig, t), create_graph=True)
    gamma_ad = torch.autograd.grad(dS, S, retain_graph=True)[0]
    vanna_ad = torch.autograd.grad(dS, sig, retain_graph=True)[0]
    charm_ad = torch.autograd.grad(dS, t, retain_graph=True)[0]
    vomma_ad = torch.autograd.grad(dsig, sig, retain_graph=True)[0]
    dS, dsig, dt, gamma_ad, vanna_ad, charm_ad, vomma_ad = (
        x.detach() for x in (dS, dsig, dt, gamma_ad, vanna_ad, charm_ad, vomma_ad))

    g = greeks(torch.tensor(s), k, tau, vol)
    assert float(g["delta"]) == pytest.approx(float(dS), rel=1e-8, abs=1e-12)
    assert float(g["vega"]) == pytest.approx(float(dsig), rel=1e-8, abs=1e-12)
    assert float(g["theta"]) == pytest.approx(float(dt), rel=1e-8, abs=1e-12)
    assert float(g["gamma"]) == pytest.approx(float(gamma_ad), rel=1e-7, abs=1e-12)
    assert float(g["vanna"]) == pytest.approx(float(vanna_ad), rel=1e-7, abs=1e-12)
    assert float(g["charm"]) == pytest.approx(float(charm_ad), rel=1e-7, abs=1e-12)
    assert float(g["vomma"]) == pytest.approx(float(vomma_ad), rel=1e-7, abs=1e-12)
    assert float(call_delta(torch.tensor(s), k, tau, vol)) == pytest.approx(float(g["delta"]))


def test_paper_parameters_transcribed():
    assert (BUEHLER_2019.kappa, BUEHLER_2019.theta, BUEHLER_2019.xi) == (1.0, 0.04, 2.0)
    assert (BUEHLER_2019.rho, BUEHLER_2019.v0, BUEHLER_2019.s0) == (-0.7, 0.04, 100.0)
    assert (MURRAY_2022.kappa, MURRAY_2022.theta, MURRAY_2022.xi, MURRAY_2022.rho) == (8.0, 0.00625, 1.0, -0.7)
    with pytest.raises(ValueError):
        MURRAY_2022.require_v0()


def test_gbm_is_a_martingale_with_correct_variance():
    p = gbm_paths(400_000, 100.0, 0.2, 1.0, 10, seed=1)
    st = p.spot[:, -1]
    assert float(st.mean()) == pytest.approx(100.0, abs=3 * float(st.std()) / math.sqrt(len(st)))
    logret = torch.log(st / 100.0)
    assert float(logret.var()) == pytest.approx(0.04, rel=0.01)
    assert p.instruments.shape == (400_000, 11, 1)


def test_pnl_hand_computed_case():
    """Two paths, two steps, one instrument, costs on, liquidation on."""
    inst = torch.tensor([[[100.0], [110.0], [105.0]],
                         [[100.0], [90.0], [95.0]]])
    pos = torch.tensor([[[0.5], [1.0]],
                        [[0.5], [0.0]]])
    payoff = torch.tensor([5.0, 0.0])
    eps = 0.01
    # path 0: gains 0.5*10 + 1.0*(-5) = 0; trades 0.5@100, 0.5@110, liquidate 1.0@105
    #         costs = 0.01*(50 + 55 + 105) = 2.10;  pnl = 3 - 5 + 0 - 2.10 = -4.10
    # path 1: gains 0.5*(-10) + 0 = -5; trades 0.5@100, 0.5@90, liquidate 0
    #         costs = 0.01*(50 + 45) = 0.95;  pnl = 3 - 0 - 5 - 0.95 = -2.95
    out = hedge_pnl(inst, pos, payoff, eps=eps, premium=3.0, liquidate=True)
    assert torch.allclose(out, torch.tensor([-4.10, -2.95]))
    no_liq = hedge_pnl(inst, pos, payoff, eps=eps, premium=3.0, liquidate=False)
    assert torch.allclose(no_liq, torch.tensor([-4.10 + 1.05, -2.95]))


def test_pnl_zero_cost_equals_discrete_stochastic_integral():
    p = gbm_paths(1000, 100.0, 0.2, T_30D, 30, seed=3)
    pos = torch.rand(1000, 30, 1, dtype=torch.float64)
    manual = (pos[..., 0] * (p.spot[:, 1:] - p.spot[:, :-1])).sum(dim=1)
    assert torch.allclose(hedge_pnl(p.instruments, pos, torch.zeros(1000)), manual)


def test_pnl_is_differentiable_in_positions():
    p = gbm_paths(64, 100.0, 0.2, T_30D, 30, seed=4)
    pos = torch.full((64, 30, 1), 0.5, requires_grad=True)
    hedge_pnl(p.instruments, pos, call_payoff(p.spot, 100.0), eps=0.01).mean().backward()
    assert pos.grad is not None and torch.isfinite(pos.grad).all()


def test_delta_hedging_error_shrinks_with_rebalancing():
    """Discrete delta hedging in Black-Scholes has error of order 1/sqrt(n).
    Quadrupling the rebalancing frequency must roughly halve the error std,
    which checks price, delta, paths, and P&L accounting together."""
    def hedge_error_std(n):
        p = gbm_paths(100_000, 100.0, 0.2, T_30D, n, seed=7)
        tau = T_30D - p.times[:-1]
        pos = call_delta(p.spot[:, :-1], 100.0, tau, 0.2).unsqueeze(-1)
        q = float(call_price(torch.tensor(100.0), 100.0, T_30D, 0.2))
        return float(hedge_pnl(p.instruments, pos, call_payoff(p.spot, 100.0), premium=q).std())
    coarse, fine = hedge_error_std(30), hedge_error_std(120)
    assert 1.7 < coarse / fine < 2.3
