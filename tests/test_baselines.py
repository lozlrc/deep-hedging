"""Analytic baselines: Black-Scholes delta and the Whalley-Wilmott no-trade band."""
from __future__ import annotations

import math

import pytest
import torch

from dhedge.baselines import bs_delta_positions, whalley_wilmott_half_width, whalley_wilmott_positions
from dhedge.bs import call_delta, greeks
from dhedge.heston import heston_paths
from dhedge.markets import gbm_paths
from dhedge.pnl import call_payoff, hedge_pnl
from dhedge.spec import BUEHLER_2019, T_30D

F64 = torch.float64


def _turnover(pos: torch.Tensor) -> torch.Tensor:
    """Shares traded per path, opening trade and final liquidation included."""
    p = pos[..., 0]
    z = torch.zeros_like(p[:, :1])
    return torch.cat([p, z], dim=1).diff(dim=1, prepend=z).abs().sum(dim=1)


@pytest.fixture(scope="module")
def heston():
    return heston_paths(BUEHLER_2019, 5000, with_variance_swap=True, seed=2)


def test_bs_delta_uses_sqrt_variance_as_implied_vol(heston):
    pos = bs_delta_positions(heston, 100.0, T_30D, notional=2.0)
    assert pos.shape == (5000, 30, 1)
    for k in (0, 17, 29):
        tau = T_30D - float(heston.times[k])
        ref = 2.0 * call_delta(heston.spot[:, k], 100.0, tau, torch.sqrt(heston.var[:, k]))
        assert torch.allclose(pos[:, k, 0], ref, rtol=0, atol=1e-15)
    atm = call_delta(torch.tensor(100.0, dtype=F64), 100.0, T_30D, 0.2)
    assert float(pos[0, 0, 0]) == pytest.approx(2.0 * float(atm))


def test_bs_delta_fixed_vol_replicates_under_gbm():
    """Sanity of the baseline itself: finer rebalancing, smaller error in its own model."""
    stds = []
    for n in (10, 40, 160):
        p = gbm_paths(20000, 100.0, 0.2, T_30D, n, seed=3)
        pos = bs_delta_positions(p, 100.0, T_30D, vol=0.2)
        prem = float(greeks(torch.tensor(100.0, dtype=F64), 100.0, T_30D, 0.2)["price"])
        stds.append(float(hedge_pnl(p.instruments, pos, call_payoff(p.spot, 100.0), premium=prem).std()))
    assert stds[0] > 1.8 * stds[1] > 1.8 ** 2 * stds[2]      # about sqrt(4) per step


def test_half_width_formula():
    """Whalley and Wilmott (1997) eq. (17) at r = 0: H = (3 eps S Gamma^2 / (2 gamma))^(1/3)."""
    S, gamma_bs, eps, ra = 100.0, 0.07, 0.002, 1.5
    expected = (3.0 * eps * S * gamma_bs ** 2 / (2.0 * ra)) ** (1.0 / 3.0)
    got = whalley_wilmott_half_width(torch.tensor(S, dtype=torch.float64),
                                     torch.tensor(gamma_bs, dtype=torch.float64), eps, ra)
    assert float(got) == pytest.approx(expected, rel=1e-14)
    assert float(got) == pytest.approx(0.0993288, rel=1e-6)


def test_band_holds_and_trades_less(heston):
    eps, ra, notional = 0.002, 1.0, 1.0
    bs = bs_delta_positions(heston, 100.0, T_30D, notional=notional)
    ww = whalley_wilmott_positions(heston, 100.0, T_30D, eps, ra, notional=notional)
    assert ww.shape == bs.shape

    n = heston.n_steps
    tau = T_30D - heston.times[:n]
    g = greeks(heston.spot[:, :n], 100.0, tau[None, :], torch.sqrt(heston.var[:, :n]))
    half = whalley_wilmott_half_width(heston.spot[:, :n], notional * g["gamma"], eps, ra)
    assert torch.all((ww[..., 0] - bs[..., 0]).abs() <= half * (1 + 1e-12) + 1e-15)

    t_ww, t_bs = _turnover(ww), _turnover(bs)
    assert torch.all(t_ww <= t_bs + 1e-12)
    assert float(t_ww.mean()) < float(t_bs.mean())
    print(f"mean turnover: Whalley-Wilmott {float(t_ww.mean()):.3f}, BS delta {float(t_bs.mean()):.3f}")


def test_band_saves_costs_under_gbm():
    """Where it is derived (BS model, proportional costs) the band should pay less in costs."""
    p = gbm_paths(20000, 100.0, 0.2, T_30D, 30, seed=8)
    eps = 0.005
    bs = bs_delta_positions(p, 100.0, T_30D, vol=0.2)
    ww = whalley_wilmott_positions(p, 100.0, T_30D, eps, 1.0, vol=0.2)
    pay = call_payoff(p.spot, 100.0)
    cost_bs = hedge_pnl(p.instruments, bs, pay) - hedge_pnl(p.instruments, bs, pay, eps=eps)
    cost_ww = hedge_pnl(p.instruments, ww, pay) - hedge_pnl(p.instruments, ww, pay, eps=eps)
    assert float(cost_ww.mean()) < float(cost_bs.mean())


def test_reduces_to_bs_delta_as_costs_vanish(heston):
    """The band closes like eps^(1/3), so the positions converge to the BS delta.

    On Heston paths the mean gap is used: where V_t is essentially zero the
    BS gamma at the money is enormous and the band stays wide for any eps
    that is not absurdly small, which is the formula working as written.
    """
    bs = bs_delta_positions(heston, 100.0, T_30D)
    gaps = []
    for eps in (1e-3, 1e-6, 1e-9, 1e-12):
        ww = whalley_wilmott_positions(heston, 100.0, T_30D, eps, 1.0)
        gaps.append(float((ww - bs).abs().mean()))
    assert all(a > 3.0 * b for a, b in zip(gaps, gaps[1:])), gaps
    ww0 = whalley_wilmott_positions(heston, 100.0, T_30D, 0.0, 1.0)
    assert torch.equal(ww0, bs)

    p = gbm_paths(5000, 100.0, 0.2, T_30D, 30, seed=4)
    bs_g = bs_delta_positions(p, 100.0, T_30D, vol=0.2)
    worst = [float((whalley_wilmott_positions(p, 100.0, T_30D, e, 1.0, vol=0.2) - bs_g).abs().max())
             for e in (1e-3, 1e-6, 1e-9, 1e-12)]
    assert all(a > b for a, b in zip(worst, worst[1:])) and worst[-1] < 1e-3, worst


def test_notional_scales_band_and_centre():
    p = gbm_paths(1000, 100.0, 0.2, T_30D, 30, seed=1)
    one = whalley_wilmott_positions(p, 100.0, T_30D, 1e-3, 1.0, vol=0.2)
    ten = whalley_wilmott_positions(p, 100.0, T_30D, 1e-3, 1.0, notional=10.0, vol=0.2)
    # Portfolio gamma scales the half-width by 10^(2/3), not 10: per unit of
    # notional the band is the single-option band at a tenth of the cost rate.
    assert not torch.allclose(ten, 10.0 * one)
    per_unit = whalley_wilmott_positions(p, 100.0, T_30D, 1e-4, 1.0, vol=0.2)
    assert torch.allclose(ten / 10.0, per_unit, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError):
        whalley_wilmott_positions(p, 100.0, T_30D, 1e-3, 0.0, vol=0.2)
    assert math.isfinite(float(ten.sum()))
