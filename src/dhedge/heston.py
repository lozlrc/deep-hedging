"""Heston market simulation as Buehler et al. (2019) describe it.

Section 5.2 and footnote 5 of the paper: V is sampled exactly from the CIR
transition law (a scaled noncentral chi-square), and V is replaced by a
piecewise constant process in the log-price SDE, a simplified Broadie-Kaya
scheme. On each fine step of length h:

    V_{j+1} = c * chi2'(df = 4 kappa theta / xi^2, nonc = V_j e^{-kappa h} / c),
    c = xi^2 (1 - e^{-kappa h}) / (4 kappa)
    log S_{j+1} = log S_j - V_j h / 2 + (rho / xi) (V_{j+1} - V_j - kappa (theta - V_j) h)
                  + sqrt((1 - rho^2) V_j h) Z_j

The rho term recovers the part of dB driven by dW from the exact variance
increment. Because V is frozen over the step, S is a martingale only up to a
discretisation bias; `substeps` refines the grid to measure and shrink it.

Variance swap (eq. 5.3-5.4): S2_t = I_t + L(t, V_t) with I_t the integrated
variance under the same piecewise-constant rule and
L(t, v) = (v - theta)/kappa (1 - e^{-kappa (T - t)}) + theta (T - t).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .spec import N_STEPS, T_30D, HestonParams, MarketPaths


def variance_swap_fair_leg_discrete(params: HestonParams, remaining_steps, h: float, v):
    """L_d = E[sum_{j >= k} V_j h | V_k = v] for the left-point sum the simulator uses.

    With the exact CIR conditional mean E[V_{k+m} | V_k] = theta + (v - theta) e^{-kappa m h},
    summing M remaining fine steps gives

        L_d = h [ M theta + (v - theta) (1 - e^{-kappa M h}) / (1 - e^{-kappa h}) ].

    S2 = I + L_d is then an exact martingale under the simulated dynamics, step by
    step. It is zero at maturity and tends to the continuous L as h -> 0.
    """
    k, th = params.kappa, params.theta
    if isinstance(remaining_steps, torch.Tensor):
        geom = -torch.expm1(-k * h * remaining_steps) / (-math.expm1(-k * h))
    else:
        m = np.asarray(remaining_steps, dtype=np.float64)
        geom = -np.expm1(-k * h * m) / (-math.expm1(-k * h))
    return h * (remaining_steps * th + (v - th) * geom)


def variance_swap_fair_leg(params: HestonParams, tau, v):
    """L(t, v) = E[int_t^T V ds | V_t = v] with tau = T - t. Works on floats, arrays, tensors."""
    k, th = params.kappa, params.theta
    if isinstance(tau, torch.Tensor):
        decay = -torch.expm1(-k * tau)
    else:
        decay = -np.expm1(-k * np.asarray(tau, dtype=np.float64))
    return (v - th) / k * decay + th * tau


def cir_moments(params: HestonParams, v0: float, t: float) -> tuple[float, float]:
    """Exact mean and variance of V_t given V_0 = v0."""
    k, th, xi = params.kappa, params.theta, params.xi
    e = math.exp(-k * t)
    mean = th + (v0 - th) * e
    var = v0 * xi * xi * e * (1.0 - e) / k + th * xi * xi * (1.0 - e) ** 2 / (2.0 * k)
    return mean, var


def spot_drift_defect(params: HestonParams, v, h: float):
    """E[S_{j+1} / S_j | V_j = v] - 1 for one step of this scheme, in closed form.

    Zero for an exact martingale. It is not zero here because V is frozen at V_j
    in the diffusion term while the rho term uses the exact V_{j+1}; the moment
    generating function of the scaled noncentral chi-square gives the exact
    size, which is how the substeps=1 bias is reported rather than guessed from
    Monte Carlo noise.
    """
    k, th, xi, rho = params.kappa, params.theta, params.xi, params.rho
    v = np.asarray(v, dtype=np.float64)
    c = xi * xi * (-math.expm1(-k * h)) / (4.0 * k)
    df = 4.0 * k * th / (xi * xi)
    lam = v * math.exp(-k * h) / c
    tc = (rho / xi) * c                     # MGF argument times the scale c
    log_m = (-0.5 * rho * rho * v * h - (rho / xi) * (v + k * (th - v) * h)
             + lam * tc / (1.0 - 2.0 * tc) - 0.5 * df * math.log1p(-2.0 * tc))
    return np.expm1(log_m)


def variance_swap_drift_defect(params: HestonParams, v, h: float):
    """E[S2_{j+1} - S2_j | V_j = v] for one step: (v - theta)(h - (1 - e^{-kappa h}) / kappa).

    The left-point integral adds v h while the fair leg expects the exact
    integral of the mean-reverting variance; the gap is O(h^2) and vanishes in
    expectation when v0 = theta, as in the 2019 parameters.
    """
    k, th = params.kappa, params.theta
    v = np.asarray(v, dtype=np.float64)
    return (v - th) * (h + math.expm1(-k * h) / k)


def heston_paths(params: HestonParams, n_paths: int, T: float = T_30D, n_steps: int = N_STEPS,
                 seed: int = 0, substeps: int = 1, with_variance_swap: bool = False,
                 variance_swap: str = "discrete",
                 dtype: torch.dtype = torch.float64, device: str = "cpu") -> MarketPaths:
    """Simulate Heston paths on the trading grid t_k = k T / n_steps.

    Simulation runs in float64 numpy on a grid `substeps` times finer, and every
    `substeps`-th point is recorded; `dtype` only sets the returned precision.
    params.v0 must be set (MURRAY_2022 leaves it None until calibrated).

    variance_swap="discrete" (default) prices S2 with the fair leg of the discrete
    left-point integral the simulator actually accumulates, so S2 is an exact
    martingale. variance_swap="continuous" uses the paper's continuous-time L,
    which is NOT a martingale under this scheme: on the final step L(T, .) = 0,
    so the last increment (V - theta)(h - (1 - e^{-kappa h})/kappa) is known in
    advance and a policy free to size its variance-swap position harvests it as
    a riskless profit. Kept only to reproduce that finding.
    """
    if substeps < 1 or n_steps < 1:
        raise ValueError("n_steps and substeps must be positive")
    v0 = params.require_v0()
    k, th, xi, rho = params.kappa, params.theta, params.xi, params.rho
    rng = np.random.default_rng(seed)

    n_fine = n_steps * substeps
    h = T / n_fine
    ekh = math.exp(-k * h)
    c = xi * xi * (-math.expm1(-k * h)) / (4.0 * k)
    df = 4.0 * k * th / (xi * xi)
    sq = math.sqrt(1.0 - rho * rho)

    logs = np.full(n_paths, math.log(params.s0))
    v = np.full(n_paths, v0)
    integ = np.zeros(n_paths)

    spot = np.empty((n_paths, n_steps + 1))
    var = np.empty((n_paths, n_steps + 1))
    ivar = np.empty((n_paths, n_steps + 1))
    spot[:, 0], var[:, 0], ivar[:, 0] = params.s0, v0, 0.0

    for j in range(n_fine):
        v_next = c * rng.noncentral_chisquare(df, v * ekh / c)
        z = rng.standard_normal(n_paths)
        logs = (logs - 0.5 * v * h + (rho / xi) * (v_next - v - k * (th - v) * h)
                + sq * np.sqrt(v * h) * z)
        integ = integ + v * h
        v = v_next
        if (j + 1) % substeps == 0:
            i = (j + 1) // substeps
            spot[:, i], var[:, i], ivar[:, i] = np.exp(logs), v, integ

    times = np.linspace(0.0, T, n_steps + 1)
    if with_variance_swap:
        if variance_swap == "discrete":
            remaining = ((n_steps - np.arange(n_steps + 1)) * substeps)[None, :]
            s2 = ivar + variance_swap_fair_leg_discrete(params, remaining, h, var)
        elif variance_swap == "continuous":
            s2 = ivar + variance_swap_fair_leg(params, T - times[None, :], var)
        else:
            raise ValueError("variance_swap must be 'discrete' or 'continuous'")
        instruments = np.stack([spot, s2], axis=-1)
    else:
        instruments = spot[..., None]

    def t(a):
        return torch.as_tensor(a, dtype=dtype, device=device)

    return MarketPaths(times=t(times), spot=t(spot), instruments=t(instruments), var=t(var))
