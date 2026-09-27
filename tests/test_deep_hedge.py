"""Pathwise deep hedging on Black-Scholes, where the right answer is known.

Zero costs: the discrete delta hedge is the benchmark and the learned hedge
should match its risk. Proportional costs: delta hedging overtrades, so the
learned hedge must beat it. Training runs are shared through module fixtures
because they dominate the runtime.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from dhedge.bs import call_delta, call_price
from dhedge.deep_hedge import (PerStepNet, SharedNet, bs_features, heston_features,
                               indifference_price, recalibrate_batchnorm, rollout_positions,
                               train_pathwise)
from dhedge.markets import gbm_paths
from dhedge.pnl import call_payoff, hedge_pnl
from dhedge.risk import cvar
from dhedge.spec import N_STEPS, T_30D, MarketPaths

S0, SIGMA, K, ALPHA = 100.0, 0.2, 100.0, 0.5
TRAIN_STEPS = 2000
N_TEST = 50_000


def sample_paths(n: int, seed: int) -> MarketPaths:
    return gbm_paths(n, S0, SIGMA, T_30D, N_STEPS, seed=seed)


def payoff(paths: MarketPaths) -> torch.Tensor:
    return call_payoff(paths.spot, K)


class BSDelta(nn.Module):
    """Closed-form delta hedge behind the policy interface, the benchmark."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.float64), requires_grad=False)

    def forward(self, features, prev, k):
        tau = T_30D * (1.0 - k / N_STEPS)
        return call_delta(torch.exp(features[:, :1]), K, tau, SIGMA)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    # The per-date nets are tiny, so thread handoff costs more than it saves.
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def test_paths() -> MarketPaths:
    return sample_paths(N_TEST, seed=987_654)          # disjoint from every training seed


@pytest.fixture(scope="module")
def bs_hedge(test_paths):
    pos = rollout_positions(BSDelta(), test_paths, bs_features)
    return {eps: hedge_pnl(test_paths.instruments, pos, payoff(test_paths), eps=eps)
            for eps in (0.0, 0.01)}


def _train(semi_recurrent: bool, eps: float, steps: int = TRAIN_STEPS, seed: int = 0):
    torch.manual_seed(seed)
    policy = PerStepNet(N_STEPS, 1, 1, semi_recurrent=semi_recurrent)
    hist = train_pathwise(policy, sample_paths, payoff, bs_features, risk="cvar", alpha=ALPHA,
                          eps=eps, steps=steps, batch=256, lr=0.005, seed=seed,
                          log_every=100, n_pool=50_000)
    return policy, hist


@pytest.fixture(scope="module")
def zero_cost_semi():
    return _train(semi_recurrent=True, eps=0.0)


@pytest.fixture(scope="module")
def zero_cost_simple():
    return _train(semi_recurrent=False, eps=0.0, steps=1500)


@pytest.fixture(scope="module")
def costly_semi():
    return _train(semi_recurrent=True, eps=0.01)


# ----------------------------------------------------------------- architecture

def test_per_step_net_is_the_paper_architecture():
    net = PerStepNet(N_STEPS, feat_dim=2, d=2, semi_recurrent=True)
    assert len(net.nets) == N_STEPS
    assert net.nets[0] is not net.nets[1]
    lin = [m for m in net.nets[5] if isinstance(m, nn.Linear)]
    bns = [m for m in net.nets[5] if isinstance(m, nn.BatchNorm1d)]
    assert [(m.in_features, m.out_features) for m in lin] == [(4, 17), (17, 17), (17, 2)]  # N0 = 2d
    assert len(bns) == 2
    seq = list(net.nets[5])
    for i, m in enumerate(seq):
        if isinstance(m, nn.ReLU):
            assert isinstance(seq[i - 1], nn.BatchNorm1d)
    simple = PerStepNet(N_STEPS, feat_dim=2, d=2, semi_recurrent=False)
    assert simple.nets[3][0].in_features == 2
    out = net(torch.randn(8, 2), torch.randn(8, 2), 7)
    assert out.shape == (8, 2)


def test_shared_net_matches_table_1_architecture():
    net = SharedNet(feat_dim=2, d=1)
    lin = [m for m in net.net if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in lin] == [(3, 256), (256, 256), (256, 256), (256, 1)]
    assert sum(isinstance(m, nn.Sigmoid) for m in net.net) == 3
    assert net(torch.randn(5, 2), torch.zeros(5, 1), 4).shape == (5, 1)
    # time to maturity is an input: same features, different dates, different holdings
    x = torch.randn(5, 2)
    assert not torch.allclose(net(x, torch.zeros(5, 1), 0), net(x, torch.zeros(5, 1), 29))


def test_feature_builders():
    p = sample_paths(10, seed=3)
    assert torch.allclose(bs_features(p, 4)[:, 0], torch.log(p.spot[:, 4]))
    h = heston_features(p, 4)
    assert h.shape == (10, 2)
    assert torch.allclose(h[:, 1], p.var[:, 4])
    with pytest.raises(ValueError):
        heston_features(MarketPaths(p.times, p.spot, p.instruments, None), 0)


# ----------------------------------------------------------------- scoring

def test_indifference_price_scores_with_the_shared_pnl(test_paths):
    """Any policy is scored by rollout + hedge_pnl; the BS delta hedge makes it checkable."""
    pos = rollout_positions(BSDelta(), test_paths, bs_features)
    tau = T_30D - test_paths.times[:-1]
    direct = call_delta(test_paths.spot[:, :-1], K, tau, SIGMA).unsqueeze(-1)
    assert pos.shape == (N_TEST, N_STEPS, 1)
    assert torch.allclose(pos, direct, atol=1e-12)
    for eps in (0.0, 0.01):
        p0, pnl = indifference_price(BSDelta(), test_paths, payoff, bs_features, "cvar",
                                     alpha=ALPHA, eps=eps)
        ref = hedge_pnl(test_paths.instruments, pos, payoff(test_paths), eps=eps)
        assert torch.equal(pnl, ref)
        closed_form = hedge_pnl(test_paths.instruments, direct, payoff(test_paths), eps=eps)
        assert torch.allclose(pnl, closed_form, atol=1e-10)
        assert p0 == pytest.approx(float(cvar(ref, ALPHA)), abs=1e-12)


class ParamFreeBSDelta(nn.Module):
    """Same benchmark with no weights at all, as a closed-form baseline is written."""

    def forward(self, features, prev, k):
        tau = T_30D * (1.0 - k / N_STEPS)
        return call_delta(torch.exp(features[:, :1]), K, tau, SIGMA)


def test_parameter_free_policy_can_be_scored(test_paths):
    """Regression: rollout took its dtype from next(policy.parameters()) and
    raised StopIteration for a weightless policy, so closed-form baselines
    could not be scored without a dummy parameter."""
    pos = rollout_positions(ParamFreeBSDelta(), test_paths, bs_features)
    assert pos.dtype == test_paths.spot.dtype
    assert torch.equal(pos, rollout_positions(BSDelta(), test_paths, bs_features))
    p0, pnl = indifference_price(ParamFreeBSDelta(), test_paths, payoff, bs_features, "cvar", alpha=ALPHA)
    ref = hedge_pnl(test_paths.instruments, pos, payoff(test_paths))
    assert torch.equal(pnl, ref)
    assert p0 == pytest.approx(float(cvar(ref, ALPHA)), abs=1e-12)


# ----------------------------------------------------------------- learning

def test_zero_cost_price_matches_bs(zero_cost_semi, test_paths, bs_hedge):
    """CVaR_0.5 price of the learned hedge against Black-Scholes.

    p0 = BS price + CVaR of the daily hedging error, and that error is
    irreducible with 30 rebalances, so the learned price is compared with the
    BS hedge scored by the same risk measure, and the risk-neutral part
    -E[PL] with the BS price itself.
    """
    policy, _ = zero_cost_semi
    p0, pnl = indifference_price(policy, test_paths, payoff, bs_features, "cvar", alpha=ALPHA)
    bs_price = float(call_price(torch.tensor(S0, dtype=torch.float64), K, T_30D, SIGMA))
    bs_p0 = float(cvar(bs_hedge[0.0], ALPHA))
    print(f"\nzero cost: deep p0 {p0:.4f}  BS-hedge CVaR {bs_p0:.4f}  BS price {bs_price:.4f}"
          f"  deep std {float(pnl.std()):.4f}  BS std {float(bs_hedge[0.0].std()):.4f}")
    assert abs(p0 - bs_p0) / bs_p0 < 0.05
    assert abs(-float(pnl.mean()) - bs_price) / bs_price < 0.05
    assert p0 >= bs_price * 0.99          # CVaR dominates -E[PL], which is the BS price


def test_zero_cost_pnl_spread_matches_delta_hedge(zero_cost_semi, test_paths, bs_hedge):
    policy, _ = zero_cost_semi
    _, pnl = indifference_price(policy, test_paths, payoff, bs_features, "cvar", alpha=ALPHA)
    ratio = float(pnl.std()) / float(bs_hedge[0.0].std())
    assert abs(ratio - 1.0) < 0.25


def test_zero_cost_learns_the_delta(zero_cost_semi, test_paths):
    """Mid-life holdings should sit close to the Black-Scholes delta."""
    policy, _ = zero_cost_semi
    policy.eval()
    with torch.no_grad():
        pos = rollout_positions(policy, test_paths, bs_features).double()
    tau = T_30D - test_paths.times[:-1]
    delta = call_delta(test_paths.spot[:, :-1], K, tau, SIGMA).unsqueeze(-1)
    err = (pos - delta)[:, 5:25].abs().mean()
    print(f"\nmean |deep - BS delta| over dates 5..24: {float(err):.4f}")
    assert float(err) < 0.08


def test_costs_learned_hedge_beats_delta_hedge(costly_semi, test_paths, bs_hedge):
    policy, _ = costly_semi
    p0, _ = indifference_price(policy, test_paths, payoff, bs_features, "cvar", alpha=ALPHA, eps=0.01)
    bs_p0 = float(cvar(bs_hedge[0.01], ALPHA))
    print(f"\neps 0.01: deep p0 {p0:.4f}  BS-hedge CVaR {bs_p0:.4f}")
    assert p0 < bs_p0


def test_both_variants_train(zero_cost_semi, zero_cost_simple, test_paths, bs_hedge):
    bs_p0 = float(cvar(bs_hedge[0.0], ALPHA))
    for name, (policy, hist) in (("semi", zero_cost_semi), ("simple", zero_cost_simple)):
        assert hist["loss"][-1] < 0.9 * hist["loss"][0]
        p0, _ = indifference_price(policy, test_paths, payoff, bs_features, "cvar", alpha=ALPHA)
        print(f"\n{name}: first logged loss {hist['loss'][0]:.4f}  last {hist['loss'][-1]:.4f}  p0 {p0:.4f}")
        assert abs(p0 - bs_p0) / bs_p0 < 0.05
        # the learned threshold w is the 0.5-VaR of the loss, i.e. near the median
        assert hist["w"][-1] == pytest.approx(-float(torch.median(bs_hedge[0.0])), abs=0.3)


def test_entropic_and_shared_net_train(test_paths):
    torch.manual_seed(1)
    policy = SharedNet(1, 1, hidden=(64, 64))
    hist = train_pathwise(policy, sample_paths, payoff, bs_features, risk="entropic", lam=1.0,
                          steps=400, lr=0.005, seed=1, log_every=50, n_pool=20_000)
    assert hist["loss"][-1] < hist["loss"][0]
    assert all(w is None for w in hist["w"])
    p0, _ = indifference_price(policy, test_paths, payoff, bs_features, "entropic", lam=1.0)
    unhedged = float(torch.logsumexp(payoff(test_paths), 0) - torch.log(torch.tensor(float(N_TEST))))
    assert p0 < 0.5 * unhedged


def test_determinism_under_fixed_seed():
    runs = []
    for _ in range(2):
        torch.manual_seed(5)
        policy = PerStepNet(N_STEPS, 1, 1)
        hist = train_pathwise(policy, sample_paths, payoff, bs_features, risk="cvar", alpha=ALPHA,
                              eps=0.01, steps=40, seed=5, log_every=10, n_pool=2000)
        runs.append((policy.state_dict(), hist))
    (sd_a, h_a), (sd_b, h_b) = runs
    assert h_a["loss"] == h_b["loss"] and h_a["w"] == h_b["w"]
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), key


def test_batchnorm_recalibration_weights_chunks_by_size():
    """Regression: recalibration used momentum=None, which averages chunk
    statistics equally, so a short trailing chunk carried as much weight as a
    full one. With 9000 paths in chunks of 8000 the 1000-path remainder got
    weight 1/2 instead of 1/9 and running variances were off by percents. The
    chunked result must match one pass over the whole set."""
    torch.manual_seed(11)
    policy = PerStepNet(N_STEPS, 1, 1).double()
    paths = sample_paths(9000, seed=21)
    policy.fit_normalization(paths, bs_features)

    def running_stats(chunk):
        recalibrate_batchnorm(policy, paths, bs_features, chunk=chunk)
        return [(m.running_mean.clone(), m.running_var.clone())
                for m in policy.modules() if isinstance(m, nn.BatchNorm1d)]

    # Train-mode chunks normalise with their own batch moments, so later dates
    # see slightly different inputs and agreement is not exact. Measured on
    # seeds 11 to 13: equal chunk weights miss by 6 to 8 percent in variance
    # and 0.03 to 0.04 sd in mean, size weights by at most 1.1 and 0.004.
    whole, chunked = running_stats(9000), running_stats(8000)
    for (mw, vw), (mc, vc) in zip(whole, chunked):
        assert float(((mc - mw) / vw.sqrt()).abs().max()) < 0.015
        assert float((vc / vw - 1).abs().max()) < 0.03


def test_validation():
    policy = PerStepNet(N_STEPS, 1, 1)
    with pytest.raises(ValueError):
        train_pathwise(policy, sample_paths, payoff, bs_features, risk="cvar", steps=1)
    with pytest.raises(ValueError):
        train_pathwise(policy, sample_paths, payoff, bs_features, risk="var", alpha=0.5, steps=1)
