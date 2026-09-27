"""The rescaling is only legitimate if the hedging problem is unchanged by it.
These tests prove that exactly, costs included."""
from __future__ import annotations

import torch

from dhedge.heston import heston_paths
from dhedge.pnl import call_payoff, hedge_pnl
from dhedge.scaling import instrument_scales, rescale, to_original_units
from dhedge.spec import BUEHLER_2019


def _paths(n=4000, seed=3):
    return heston_paths(BUEHLER_2019, n, seed=seed, with_variance_swap=True)


def test_scales_equalize_per_step_volatility():
    p = _paths()
    c = instrument_scales(p)
    assert float(c[0]) == 1.0
    q = rescale(p, c)
    d = q.instruments[:, 1:, :] - q.instruments[:, :-1, :]
    sd = d.reshape(-1, 2).std(dim=0)
    assert torch.allclose(sd[0], sd[1], rtol=1e-10)


def test_pnl_is_invariant_under_rescaling_with_costs():
    """delta . dS and eps |d delta| S are both unchanged by (S*c, delta/c)."""
    p = _paths()
    c = instrument_scales(p)
    q = rescale(p, c)
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(p.n_paths, p.n_steps, 2, generator=g, dtype=p.instruments.dtype)
    pos[..., 1] *= 400.0                                  # original-unit variance-swap sizes
    Z = call_payoff(p.spot, 100.0)
    for eps in (0.0, 0.01):
        for liq in (True, False):
            a = hedge_pnl(p.instruments, pos, Z, eps=eps, liquidate=liq)
            b = hedge_pnl(q.instruments, pos / c, Z, eps=eps, liquidate=liq)
            assert torch.allclose(a, b, rtol=1e-11, atol=1e-9)


def test_round_trip_positions():
    p = _paths(500)
    c = instrument_scales(p)
    pos = torch.rand(500, p.n_steps, 2, dtype=p.instruments.dtype)
    assert torch.allclose(to_original_units(pos / c, c), pos)


def test_spot_column_untouched():
    p = _paths(500)
    q = rescale(p, instrument_scales(p))
    assert torch.equal(q.instruments[..., 0], p.instruments[..., 0])
    assert torch.equal(q.spot, p.spot)
