"""Tests for the 2022 risk-averse actor-critic.

Everything runs on GBM, where the BS delta hedge is the known reference and
the implied vol sqrt(V_t) equals the true vol.
"""
from __future__ import annotations

import math

import pytest
import torch

from dhedge.actor_critic import (ActorCritic, actor_loss, critic_loss, episode_rewards,
                                 exp_utility, soft_update, step_reward)
from dhedge.bs import call_delta, call_price
from dhedge.features import HedgeState, raw_features
from dhedge.markets import gbm_paths
from dhedge.pnl import call_payoff, hedge_pnl
from dhedge.spec import N_STEPS, T_30D

SIGMA = 0.2
NOTIONAL = 100.0
EPS = 0.002
K = 1.0
F64 = torch.float64


def sample_paths(n: int, seed: int):
    return gbm_paths(n, 1.0, SIGMA, T_30D, N_STEPS, seed=seed)


def bs_holdings(paths) -> torch.Tensor:
    tau = T_30D - paths.times[:-1]
    return (NOTIONAL * call_delta(paths.spot[:, :-1], K, tau, SIGMA)).unsqueeze(-1)


def scored_pnl(paths, holdings) -> torch.Tensor:
    return hedge_pnl(paths.instruments, holdings, NOTIONAL * call_payoff(paths.spot, K),
                     eps=EPS, liquidate=False)


def _randomise_head(ac: ActorCritic, seed: int = 0, scale: float = 0.05) -> None:
    g = torch.Generator().manual_seed(seed)
    head = ac.actor[-1]
    with torch.no_grad():
        head.weight.copy_(scale * torch.randn(head.weight.shape, generator=g, dtype=head.weight.dtype))
        head.bias.fill_(0.01)


# ---------------------------------------------------------------- accounting

def test_rewards_telescope_to_hedge_pnl():
    """Cash accounting of eq. 14: sum_t r_t is the book-value P&L, no liquidation."""
    paths = sample_paths(4000, seed=1)
    g = torch.Generator().manual_seed(2)
    holdings = NOTIONAL * torch.rand(paths.n_paths, paths.n_steps, generator=g, dtype=F64) - 20.0
    r = episode_rewards(paths.spot, holdings, K, NOTIONAL, EPS)
    assert r.shape == (paths.n_paths, paths.n_steps + 1)
    ref = scored_pnl(paths, holdings.unsqueeze(-1))
    assert torch.allclose(r.sum(dim=1), ref, rtol=0.0, atol=1e-10)
    # Liquidating would charge eps |h_{n-1}| S_T more, which the paper does not.
    liq = hedge_pnl(paths.instruments, holdings.unsqueeze(-1), NOTIONAL * call_payoff(paths.spot, K),
                    eps=EPS, liquidate=True)
    assert not torch.allclose(r.sum(dim=1), liq)


def test_training_transitions_telescope_to_hedge_pnl():
    """The rewards and terminal value the trainer actually uses add up to hedge_pnl."""
    paths = sample_paths(3000, seed=3)
    ac = ActorCritic(notional=NOTIONAL, eps=EPS)
    ac.fit_normalizer(sample_paths(5000, seed=4), K)
    _randomise_head(ac)
    holdings = ac.rollout_holdings(paths, K, lam=0.3)
    n = paths.n_paths
    b = ac._build_batch(paths, torch.full((n,), K, dtype=F64), torch.full((n,), 0.3, dtype=F64))
    h = torch.zeros(n, dtype=F64)
    total = torch.zeros(n, dtype=F64)
    for t in range(paths.n_steps):
        a = holdings[:, t, 0] - h
        total = total + step_reward(a, b.spot[:, t], EPS)
        h = holdings[:, t, 0]
    total = total + ac._value(ac.target, b, paths.n_steps, h)    # V(s_T) = r_T exactly
    assert torch.allclose(total, scored_pnl(paths, holdings), rtol=0.0, atol=1e-9)


# ---------------------------------------------------------------- losses

@pytest.mark.parametrize("lam", [1e-4, 1e-2, 0.1, 1.0])
def test_critic_loss_minimiser_is_exponential_utility(lam):
    """One-step problem y in {-3, 0.5, 2} w.p. {0.2, 0.5, 0.3}: argmin_V of the
    critic loss is U_lam(y) = -(1/lam) log E[exp(-lam y)], the solution of eq. 9."""
    y = torch.tensor([-3.0] * 2 + [0.5] * 5 + [2.0] * 3, dtype=F64)
    lam_t = torch.full_like(y, lam)
    target = -math.log(torch.exp(-lam * y).mean().item()) / lam

    v = torch.zeros((), dtype=F64, requires_grad=True)
    for _ in range(50):                                  # Newton: the loss is strictly convex in V
        loss = critic_loss(y - v, lam_t)
        (g,) = torch.autograd.grad(loss, v, create_graph=True)
        (hess,) = torch.autograd.grad(g, v)
        with torch.no_grad():
            v -= g / hess
    assert float(v.detach()) == pytest.approx(target, abs=1e-9)

    # The derivative is (E[exp(-lam (Y - V))] - 1) / lam, the docstring's identity.
    v0 = torch.tensor(0.7, dtype=F64, requires_grad=True)
    (g0,) = torch.autograd.grad(critic_loss(y - v0, lam_t), v0)
    expected = (torch.exp(-lam * (y - 0.7)).mean().item() - 1.0) / lam
    assert float(g0) == pytest.approx(expected, rel=1e-8, abs=1e-12)
    # Risk aversion prices the downside: U_lam sits below the mean for lam > 0.
    assert target < float(y.mean())


def test_critic_loss_is_accurate_in_float32_at_small_lam():
    """At lam = 1e-4 the naive formula cancels to noise in float32; the series
    branch must still put the minimiser on U_lam."""
    y = torch.tensor([-3.0, 0.5, 0.5, 2.0], dtype=torch.float32)
    lam = torch.full_like(y, 1e-4)
    target = -math.log(torch.exp(-1e-4 * y.double()).mean().item()) / 1e-4
    v = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([v], lr=1.0, max_iter=100, tolerance_grad=1e-12, tolerance_change=1e-14)

    def closure():
        opt.zero_grad()
        loss = critic_loss(y - v, lam)
        loss.backward()
        return loss

    opt.step(closure)
    assert float(v.detach()) == pytest.approx(target, abs=1e-4)


def test_actor_loss_gradient_points_toward_the_hedge():
    """Short a forward on 50 units, choose a spot holding a. Any risk-averse agent
    wants a near 50; the loss gradient must push a up from 0 and down from 100."""
    g = torch.Generator().manual_seed(5)
    n = 4096
    ds = 0.01 * torch.randn(n, generator=g, dtype=F64)
    lam = torch.exp(math.log(1e-2) + math.log(100.0) * torch.rand(n, generator=g, dtype=F64))

    def grad_at(a0: float, lam: torch.Tensor, baseline: float = 0.0) -> torch.Tensor:
        a = torch.full((n,), a0, dtype=F64, requires_grad=True)
        x = step_reward(a, torch.ones(n, dtype=F64), EPS) + a * (1.0 + ds) - 50.0 * ds
        loss = actor_loss(x, torch.full((n,), baseline, dtype=F64), lam)
        (grad,) = torch.autograd.grad(loss, a)
        return grad

    assert grad_at(0.0, lam).sum() < 0.0      # gradient descent increases a
    assert grad_at(100.0, lam).sum() > 0.0    # and decreases it past the hedge

    # Overflow guard. At lam = 1 a baseline of 1000 puts every exponent near
    # 1000, where exp overflows even in float64. The per-batch shift must keep
    # the gradient finite and leave its direction exactly unchanged.
    ones = torch.ones(n, dtype=F64)
    g_plain, g_shifted = grad_at(20.0, ones), grad_at(20.0, ones, baseline=1000.0)
    assert torch.isfinite(g_shifted).all() and g_shifted.abs().sum() > 0
    cos = torch.nn.functional.cosine_similarity(g_plain, g_shifted, dim=0)
    assert float(cos) == pytest.approx(1.0, abs=1e-12)


def test_soft_update_moves_target_by_exactly_tau():
    torch.manual_seed(0)
    online = torch.nn.Linear(4, 3, dtype=F64)
    target = torch.nn.Linear(4, 3, dtype=F64)
    before = [p.detach().clone() for p in target.parameters()]
    soft_update(target, online, 0.001)
    for p_bar, p0, p in zip(target.parameters(), before, online.parameters()):
        assert torch.allclose(p_bar - p0, 0.001 * (p - p0), rtol=1e-9, atol=1e-15)


# ---------------------------------------------------------------- skip connections

def test_zero_residual_is_exactly_the_bs_delta_hedge():
    paths = sample_paths(5000, seed=6)
    ac = ActorCritic(notional=NOTIONAL, eps=EPS)
    ac.fit_normalizer(sample_paths(5000, seed=7), K)
    for p in ac.actor.parameters():                 # hidden layers arbitrary, head zero
        if p is not ac.actor[-1].weight and p is not ac.actor[-1].bias:
            with torch.no_grad():
                p.add_(0.3)
    for lam in (1e-4, 0.1, 1.0):
        h = ac.rollout_holdings(paths, K, lam)
        assert h.shape == (paths.n_paths, paths.n_steps, 1)
        assert torch.allclose(h, bs_holdings(paths), rtol=0.0, atol=1e-10)
    # With a zero critic residual, V is the book value h S - notional * C_BS.
    state = HedgeState(spot=paths.spot[:, 5], var=paths.var[:, 5],
                       time=paths.times[5].expand(paths.n_paths), maturity=T_30D,
                       h_prev=bs_holdings(paths)[:, 4, 0], strike=torch.full((paths.n_paths,), K, dtype=F64),
                       lam=torch.full((paths.n_paths,), 0.1, dtype=F64))
    book = state.h_prev * state.spot - NOTIONAL * call_price(state.spot, K, state.tau, SIGMA)
    assert torch.allclose(ac.value(state), book, atol=1e-10)


def test_value_at_maturity_is_the_terminal_reward():
    """value() must agree with the trainer's V(s_T) = r_T even when the critic
    residual is nonzero; the residual was never fitted at tau = 0."""
    paths = sample_paths(1000, seed=12)
    n = paths.n_paths
    ac = ActorCritic(notional=NOTIONAL, eps=EPS, hidden=32)
    ac.fit_normalizer(sample_paths(3000, seed=13), K)
    with torch.no_grad():
        ac.critic[-1].bias.fill_(0.5)
    h = torch.full((n,), 40.0, dtype=F64)
    lam = torch.full((n,), 0.1, dtype=F64)
    state = HedgeState(spot=paths.spot[:, -1], var=paths.var[:, -1],
                       time=paths.times[-1].expand(n), maturity=T_30D, h_prev=h,
                       strike=torch.full((n,), K, dtype=F64), lam=lam)
    r_T = h * paths.spot[:, -1] - NOTIONAL * call_payoff(paths.spot, K)
    assert torch.allclose(ac.value(state), r_T, rtol=0.0, atol=1e-12)
    b = ac._build_batch(paths, torch.full((n,), K, dtype=F64), lam)
    assert torch.allclose(ac._value(ac.critic, b, paths.n_steps, h), r_T, rtol=0.0, atol=1e-12)
    # Before maturity the residual is still applied.
    mid = HedgeState(spot=paths.spot[:, 10], var=paths.var[:, 10],
                     time=paths.times[10].expand(n), maturity=T_30D, h_prev=h,
                     strike=torch.full((n,), K, dtype=F64), lam=lam)
    book = h * mid.spot - NOTIONAL * call_price(mid.spot, K, mid.tau, SIGMA)
    assert torch.allclose(ac.value(mid) - book, torch.full_like(book, 0.5), atol=1e-6)


def test_training_loop_feeds_losses_the_documented_quantities(monkeypatch):
    """With frozen, randomised networks, the u passed to critic_loss must be
    r_t + Vbar(s_{t+1}) - V(s_t) and the actor's x must be r_t + V(s_{t+1}),
    recomputed independently from value() on explicit states."""
    import copy

    import dhedge.actor_critic as ac_mod

    ac = ActorCritic(notional=NOTIONAL, eps=EPS, hidden=32, lr_actor=0.0, lr_critic=0.0,
                     tau=0.0, seed=3)
    ac.fit_normalizer(sample_paths(5000, seed=14), K)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for net in (ac.actor, ac.critic):
            net[-1].weight.copy_(0.2 * torch.randn(net[-1].weight.shape, generator=g))
            net[-1].bias.fill_(0.03)
    ac.target = copy.deepcopy(ac.critic)
    rec_c, rec_a = [], []
    orig_c, orig_a = ac_mod.critic_loss, ac_mod.actor_loss

    def spy_c(u, lam):
        rec_c.append((u.detach().double().clone(), lam.double().clone()))
        return orig_c(u, lam)

    def spy_a(x, base, lam):
        rec_a.append((x.detach().double().clone(), base.detach().double().clone()))
        return orig_a(x, base, lam)

    monkeypatch.setattr(ac_mod, "critic_loss", spy_c)
    monkeypatch.setattr(ac_mod, "actor_loss", spy_a)
    paths = sample_paths(512, seed=15)
    n = paths.n_paths
    strike = torch.full((n,), K, dtype=F64)
    lam = torch.exp(torch.linspace(math.log(1e-4), 0.0, n, dtype=F64))
    ac._train_episodes(ac._build_batch(paths, strike, lam).to("cpu", torch.float32))
    holdings = ac.rollout_holdings(paths, K, lam)[..., 0]

    def state(k, h):
        return HedgeState(spot=paths.spot[:, k], var=paths.var[:, k],
                          time=paths.times[k].expand(n), maturity=T_30D, h_prev=h,
                          strike=strike, lam=lam)

    h_prev = torch.zeros(n, dtype=F64)
    for t in range(paths.n_steps):
        h_t = holdings[:, t]
        y = step_reward(h_t - h_prev, paths.spot[:, t], EPS) + ac.value(state(t + 1, h_t))
        v = ac.value(state(t, h_prev))
        u, lam_seen = rec_c[t]
        x, base = rec_a[t]
        assert torch.allclose(lam_seen, lam, rtol=1e-6, atol=0.0)   # lam travels in float32
        assert torch.allclose(u, y - v, rtol=0.0, atol=2e-4)
        assert torch.allclose(x, y, rtol=0.0, atol=2e-4)
        assert torch.allclose(base, v, rtol=0.0, atol=2e-4)
        h_prev = h_t


def test_act_matches_training_feature_path():
    """act() builds features from bs.greeks directly; training uses a float64
    precompute plus the holding. Both must give the same trade."""
    paths = sample_paths(2000, seed=8)
    ac = ActorCritic(notional=NOTIONAL, eps=EPS)
    ac.fit_normalizer(sample_paths(5000, seed=9), K)
    _randomise_head(ac, seed=1)
    lam = torch.exp(torch.linspace(math.log(1e-4), 0.0, paths.n_paths, dtype=F64))
    holdings = ac.rollout_holdings(paths, K, lam)[..., 0]
    h_prev = torch.zeros(paths.n_paths, dtype=F64)
    for t in (0, 7, 29):
        if t > 0:
            h_prev = holdings[:, t - 1]
        state = HedgeState(spot=paths.spot[:, t], var=paths.var[:, t],
                           time=paths.times[t].expand(paths.n_paths), maturity=T_30D,
                           h_prev=h_prev, strike=torch.full((paths.n_paths,), K, dtype=F64), lam=lam)
        trade = ac.act(state, deterministic=True)
        assert torch.allclose(h_prev + trade, holdings[:, t], rtol=0.0, atol=1e-4)
        assert raw_features(state, NOTIONAL, ac.vol_floor).shape == (paths.n_paths, 13)


# ---------------------------------------------------------------- learning

TRAIN_ITERS = 150


@pytest.fixture(scope="module")
def trained():
    ac = ActorCritic(notional=NOTIONAL, eps=EPS, device="cpu", seed=0)
    hist = ac.train(sample_paths, K, n_iters=TRAIN_ITERS, batch=2048, seed=0)
    return ac, hist


@pytest.fixture(scope="module")
def held_out():
    return sample_paths(50_000, seed=987_654)


def test_actor_critic_beats_delta_hedge_at_lam_0_1(trained, held_out):
    ac, hist = trained
    assert len(hist) == TRAIN_ITERS and all(math.isfinite(r["critic_loss"]) for r in hist)
    lam = 0.1
    u_ac = exp_utility(scored_pnl(held_out, ac.rollout_holdings(held_out, K, lam)), lam)
    u_bs = exp_utility(scored_pnl(held_out, bs_holdings(held_out)), lam)
    print(f"\nU_0.1: actor-critic {u_ac:.4f}  BS delta {u_bs:.4f}  "
          f"after {ac.episodes_seen} episodes")
    assert u_ac > u_bs


def test_lambda_conditions_the_policy(trained, held_out):
    """One network, many risk aversions: the more risk-averse policy trades
    closer to the hedge and so leaves a narrower P&L distribution."""
    ac, _ = trained
    h_lo = ac.rollout_holdings(held_out, K, 1e-4)
    h_hi = ac.rollout_holdings(held_out, K, 1.0)
    gap = (h_lo - h_hi).abs().mean().item()
    sd_lo = scored_pnl(held_out, h_lo).std().item()
    sd_hi = scored_pnl(held_out, h_hi).std().item()
    print(f"\nmean |h(1e-4) - h(1)| = {gap:.3f} units;  P&L sd {sd_lo:.3f} (lam 1e-4) "
          f"vs {sd_hi:.3f} (lam 1)")
    assert gap > 1.0
    assert sd_hi < sd_lo
