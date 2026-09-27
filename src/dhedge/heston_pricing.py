"""Heston call price, its S and v sensitivities, and the 2019 model hedge.

Pricing formula. Lewis (2001) writes the zero-rate call as a single contour
integral of the characteristic function of x = log(S_T / S_t):

    C = [X >= 0] (S - K) - sqrt(S K) / (2 pi) * int_Gamma e^{i z X} phi(z - i/2) / (z^2 + 1/4) dz

with X = log(S / K). On the real line (Lewis's own contour) the bracket term is
S, not S - K; moving the contour above the pole at z = i/2 for X >= 0, or below
the pole at z = -i/2 for X < 0, collects the residue that turns the integral
into the put (ITM call) or the call itself (OTM call). Either way the integral
is the small time value, so no digits are lost to cancellation.

Quadrature. The real-line integrand is hopeless for this repo's regime: the
2019 parameters give a CIR shape parameter of 0.02, so V_t sits near zero most
of the time, and one trading day before expiry the characteristic function then
decays only like exp(-4e-5 |u|) while e^{iuX} oscillates thousands of times.
The sinh-acceleration change of variables of Boyarchenko and Levendorskii
(2019), z = i c0 + b sinh(y + i omega) - i b sin(omega), bends the contour into
the half-plane where e^{izX} decays and makes the integrand fall off doubly
exponentially in y, so a plain trapezoid rule with at most 361 nodes (fewer
once the negligible tail is trimmed) is converged to about 1e-16 in delta
across maturities from six hours to a year.

Characteristic function. The "little Heston trap" form of Albrecher, Mayer,
Schoutens and Tistaert (2007), which keeps the principal branch of the complex
log continuous. It is rewritten with (beta - d) = -xi^2 q / (beta + d) so that
nothing is divided by xi^2 after a cancellation, which is what keeps the
xi -> 0 limit (Black-Scholes) accurate.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .spec import HestonParams, MarketPaths

_TAU_FLOOR = 1e-12
# Contour: crossing height, opening angle, scale, trapezoid step and reach.
# c0 = 1 keeps the crossing 1/2 away from the poles at +-i/2 and needs the
# moments E[S^-1/2] (upper) and E[S^3/2] (lower). With the 2019 parameters the
# first explodes at tau = 2.05 years, hence the fallback in _integrals. omega = 0.4 and a step of 0.05 were chosen by a convergence sweep against a
# twice finer, longer contour (max deviation 2e-16 in delta over maturities
# from 6 hours to a year); omega = 0.6 already loses digits.
_C0 = 1.0
_OMEGA = 0.4
_B = 1.0
_ZETA = 0.05
_Y_MAX = 18.0
_TAIL_TOL = 1e-17
# Fraction of the moment-explosion horizon up to which the shifted crossing is
# used. At 73% of it (2019 parameters, tau = 1.5) the error is still 4e-12.
_EXPLOSION_MARGIN = 0.7


def _nodes(zeta: float, dtype=torch.float64, device="cpu") -> torch.Tensor:
    n = int(round(_Y_MAX / zeta))
    return torch.arange(n + 1, dtype=dtype, device=device) * zeta


def _omega(params: HestonParams) -> float:
    """Opening angle of the contour, kept inside the cone where the integrand decays.

    For large |w| the exponent A + D v grows like w (-i rho - sqrt(1 - rho^2) sgn Re w) / xi,
    whose real part is negative only for |arg w| < arccos|rho| on the side where
    rho sin(arg w) > 0. A fixed 0.4 leaves that cone once |rho| > 0.92 (rho = -0.95
    gave NaN prices) and sits so close to its edge at |rho| = 0.9 that the
    trapezoid rule is off by 1e-5. Half the cone centres the contour in the strip
    between the real axis and the edge, and for rho = -0.7 it is 0.398, the
    angle the convergence sweep picked.
    """
    if abs(params.rho) >= 1.0:
        raise ValueError("|rho| must be below 1")
    return min(_OMEGA, 0.5 * math.acos(abs(params.rho)))


def _contour(sign: float, c0: float, y: torch.Tensor, omega: float = _OMEGA):
    """Points z(y) and dz/dy on the upper (sign=+1) or lower (sign=-1) contour.

    The curve crosses the imaginary axis at i sign c0 and opens towards the
    half-plane where e^{izX} decays for X of that sign.
    """
    om = sign * omega
    yc = torch.complex(y, torch.full_like(y, om))
    z = 1j * sign * c0 + _B * torch.sinh(yc) - 1j * _B * math.sin(om)
    dz = _B * torch.cosh(yc)
    return z, dz


def char_exponents(w: torch.Tensor, tau, params: HestonParams):
    """(A, D) with E[exp(i w log(S_T/S_t)) | V_t = v] = exp(A + D v).

    w is complex; tau broadcasts against it. Little-trap form, principal branch.
    """
    kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
    if xi <= 0.0:
        raise ValueError("xi must be positive; take the limit numerically with a small xi")
    beta = kappa - rho * xi * 1j * w
    q = 1j * w + w * w
    d = torch.sqrt(beta * beta + xi * xi * q)
    bpd = beta + d
    r = -q / bpd                        # (beta - d) / xi^2, without the cancellation
    g = xi * xi * r / bpd               # (beta - d) / (beta + d)
    e = torch.exp(-d * tau)
    one_m_e = -torch.expm1(-d * tau)
    D = r * one_m_e / (1.0 - g * e)
    y = g * one_m_e / (1.0 - g)
    A = kappa * theta * (r * tau - 2.0 * torch.log1p(y) / (xi * xi))
    return A, D


def moment_explosion_time(p: float, params: HestonParams) -> float:
    """Horizon beyond which E[S_T^p] is infinite (Andersen and Piterbarg, 2007).

    With k = rho xi p - kappa and Delta = k^2 - xi^2 p (p - 1): no explosion if
    Delta >= 0 and k < 0; else T* = log((k + sqrt Delta)/(k - sqrt Delta))/sqrt Delta
    for Delta >= 0, and T* = 2 (pi [k < 0] + arctan(sqrt(-Delta)/k))/sqrt(-Delta)
    for Delta < 0.
    """
    k = params.rho * params.xi * p - params.kappa
    delta = k * k - params.xi ** 2 * p * (p - 1.0)
    if delta >= 0.0:
        if k < 0.0:
            return math.inf
        r = math.sqrt(delta)
        return math.inf if k == r else math.log((k + r) / (k - r)) / r
    r = math.sqrt(-delta)
    if k == 0.0:
        return math.pi / r
    return 2.0 * ((math.pi if k < 0.0 else 0.0) + math.atan(r / k)) / r


def _as_tensor(x, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(torch.float64)
    return torch.as_tensor(np.asarray(x, dtype=np.float64), device=device)


def _integrals(X, v, tau, params: HestonParams, want=("P", "S", "V"), zeta: float | None = None):
    """Contour integrals for price (P), S-derivative (S) and v-derivative (V).

    X, v are 1-D float64 tensors of equal length. tau is a python float (shared,
    so the characteristic exponents are computed once per node) or a tensor of
    the same length. Returns a dict of real tensors, one entry per `want`, plus
    "shifted": True where the residue form applies (see _brackets).
    """
    omega = _omega(params)
    # The trapezoid error decays like exp(-2 pi d / zeta) with d the distance to
    # the cone edge, which shrinks with omega, so the step shrinks in proportion.
    zeta = (_ZETA if zeta is None else zeta) * omega / _OMEGA
    dev = X.device
    y = _nodes(zeta, device=dev)
    wts = torch.full_like(y, 2.0 * zeta)
    wts[0] = zeta                      # symmetric trapezoid folded onto y >= 0
    shared_tau = not isinstance(tau, torch.Tensor)
    out = {k: torch.zeros_like(X) for k in want}
    out["shifted"] = torch.zeros_like(X, dtype=torch.bool)
    upper = X >= 0.0
    groups = []
    for sign, side in ((1.0, upper), (-1.0, ~upper)):
        # The shifted crossing needs E[S^p] with p = 1/2 -+ c0. Close to its
        # explosion horizon the strip of analyticity collapses, so fall back to
        # Lewis's own crossing at 0 (p = 1/2, always finite) and accept the
        # cancellation against S there.
        t_star = moment_explosion_time(0.5 - sign * _C0, params)
        ok = tau < _EXPLOSION_MARGIN * t_star
        ok = torch.full_like(side, bool(ok)) if shared_tau else ok
        groups += [(sign, _C0, side & ok), (sign, 0.0, side & ~ok)]
    for sign, c0, mask in groups:
        if not bool(mask.any()):
            continue
        out["shifted"][mask] = c0 > 0.0
        z, dz = _contour(sign, c0, y, omega)
        w = z - 0.5j
        Xs, vs = X[mask], v[mask]
        if shared_tau:
            A, D = char_exponents(w, float(tau), params)            # [M]
            # Trim nodes whose largest possible contribution is negligible. The
            # exponent is affine in v, so its maximum over v sits at an endpoint,
            # and |e^{izX}| <= 1 on the side of the contour chosen for X.
            env = torch.maximum((A + D * vs.min()).real, (A + D * vs.max()).real)
            lor_abs = 1.0 / (z * z + 0.25).abs()
            wmax = 1.0 / (0.5 - 1j * z).abs() + (1.0 + D.abs()) * lor_abs
            mag = torch.exp(env) * dz.abs() * wmax
            keep = torch.nonzero(mag > _TAIL_TOL).flatten()
            m = int(keep.max()) + 1 if keep.numel() else 1
            A, D, zz, dzz, ww = A[:m], D[:m], z[:m], dz[:m], wts[:m]
            expo = A[None, :] + D[None, :] * vs[:, None] + 1j * zz[None, :] * Xs[:, None]
            Dm = D[None, :]
        else:
            taus = tau[mask]
            A, D = char_exponents(w[None, :], taus[:, None], params)  # [n, M]
            zz, dzz, ww = z, dz, wts
            expo = A + D * vs[:, None] + 1j * zz[None, :] * Xs[:, None]
            Dm = D
        E = torch.exp(expo)
        base = dzz * ww
        lor = 1.0 / (zz * zz + 0.25)
        # With a shared tau every weight is a vector over nodes, so each
        # integral is a single complex mat-vec rather than a broadcast and sum.
        if "P" in want:
            out["P"][mask] = (E @ (base * lor)).real
        if "S" in want:
            out["S"][mask] = (E @ (base / (0.5 - 1j * zz))).real
        if "V" in want:
            if shared_tau:
                out["V"][mask] = (E @ (base * D * lor)).real
            else:
                out["V"][mask] = (E * (Dm * (base * lor)[None, :])).sum(dim=1).real
    return out


def _brackets(S, K, X, shifted):
    """Terms outside the integral for price and delta.

    Shifted contour: [X >= 0] (S - K) and [X >= 0]. Lewis's real-line crossing:
    S and 1.
    """
    itm = (X >= 0.0).to(S.dtype)
    price = torch.where(shifted, itm * (S - K), S)
    delta = torch.where(shifted, itm, torch.ones_like(S))
    return price, delta


def _evaluate(S, v, tau, K, params: HestonParams, want, chunk: int = 20000):
    """Broadcast inputs, handle expiry, run the integrals in chunks."""
    is_torch = any(isinstance(a, torch.Tensor) for a in (S, v, tau, K))
    dev = next((a.device for a in (S, v, tau, K) if isinstance(a, torch.Tensor)), None)
    S_, v_, tau_, K_ = torch.broadcast_tensors(*(_as_tensor(a, dev) for a in (S, v, tau, K)))
    shape = S_.shape
    S_, v_, tau_, K_ = (a.reshape(-1) for a in (S_, v_, tau_, K_))
    v_ = torch.clamp(v_, min=0.0)
    X = torch.log(S_ / K_)
    live = tau_ > _TAU_FLOOR
    res = {}
    if "P" in want:
        res["P"] = torch.clamp(S_ - K_, min=0.0)
    if "S" in want:
        res["S"] = (S_ > K_).to(S_.dtype)
    if "V" in want:
        res["V"] = torch.zeros_like(S_)
    idx = torch.nonzero(live).flatten()
    for lo in range(0, idx.numel(), chunk):
        j = idx[lo:lo + chunk]
        tj = tau_[j]
        shared = bool((tj == tj[0]).all())
        I = _integrals(X[j], v_[j], float(tj[0]) if shared else tj, params, want)
        sk = torch.sqrt(S_[j] * K_[j]) / (2.0 * math.pi)
        b_price, b_delta = _brackets(S_[j], K_[j], X[j], I["shifted"])
        if "P" in want:
            res["P"][j] = b_price - sk * I["P"]
        if "S" in want:
            res["S"][j] = b_delta - torch.sqrt(K_[j] / S_[j]) / (2.0 * math.pi) * I["S"]
        if "V" in want:
            res["V"][j] = -sk * I["V"]
    res = {k: val.reshape(shape) for k, val in res.items()}
    if not is_torch:
        res = {k: (float(val) if val.dim() == 0 else val.numpy()) for k, val in res.items()}
    return res


def heston_call(S, v, tau, K, params: HestonParams):
    """Undiscounted Heston call price at spot S, variance v, time to expiry tau.

    Inputs broadcast. Returns a float64 tensor if any input is a tensor, else a
    numpy array (a float for scalar inputs). Exactly intrinsic at expiry.
    """
    return _evaluate(S, v, tau, K, params, ("P",))["P"]


def heston_delta(S, v, tau, K, params: HestonParams):
    """dC/dS, from differentiating the Lewis integrand in S analytically."""
    return _evaluate(S, v, tau, K, params, ("S",))["S"]


def heston_dv(S, v, tau, K, params: HestonParams):
    """dC/dv: the characteristic function is exp(A + D v), so d/dv multiplies it by D."""
    return _evaluate(S, v, tau, K, params, ("V",))["V"]


def heston_greeks(S, v, tau, K, params: HestonParams) -> dict:
    """Price, delta and dv from one pass over the nodes."""
    r = _evaluate(S, v, tau, K, params, ("P", "S", "V"))
    return {"price": r["P"], "delta": r["S"], "dv": r["V"]}


def variance_swap_dldv(params: HestonParams, tau):
    """dL/dv for L(t, v) = (v - theta)/kappa (1 - e^{-kappa tau}) + theta tau."""
    return -math.expm1(-params.kappa * tau) / params.kappa


# The hedge runs on up to 3e7 (path, date) points, so it uses a coarser step.
# Against the 0.05 step it moves delta by at most 5e-13 on the convergence
# sweep grid, far below anything a hedge can resolve, for 40% fewer nodes.
_HEDGE_ZETA = 0.08


def model_hedge_positions(paths: MarketPaths, params: HestonParams, K: float, T: float,
                          chunk: int = 20000) -> torch.Tensor:
    """Buehler et al. (2019) eq. (5.6): the perfect hedge in the continuous model.

    delta1 = dC/dS in the spot, delta2 = (dC/dv) / (dL/dv) in the variance swap,
    at every rebalancing date t_0..t_{n-1}. Returns [N, n, 2] float64.

    All paths share tau on a given date, so the characteristic exponents are
    built once per date and each path costs one complex exponential per node.
    """
    if paths.var is None:
        raise ValueError("the model hedge needs the variance process")
    spot = paths.spot.to(torch.float64).cpu()
    var = torch.clamp(paths.var.to(torch.float64).cpu(), min=0.0)
    times = paths.times.to(torch.float64).cpu()
    N, n = spot.shape[0], spot.shape[1] - 1
    pos = torch.empty(N, n, 2, dtype=torch.float64)
    for k in range(n):
        tau = float(T - times[k])
        S_k, v_k = spot[:, k], var[:, k]
        if tau <= _TAU_FLOOR:
            pos[:, k, 0] = (S_k > K).to(torch.float64)
            pos[:, k, 1] = 0.0
            continue
        X = torch.log(S_k / K)
        dldv = variance_swap_dldv(params, tau)
        # Sorting by v gives each chunk a narrow v range, so the node trimming
        # in _integrals (which bounds over the chunk's v range) cuts deeper.
        order = torch.argsort(v_k)
        for lo in range(0, N, chunk):
            sl = order[lo:lo + chunk]
            I = _integrals(X[sl], v_k[sl], tau, params, ("S", "V"), zeta=_HEDGE_ZETA)
            _, b_delta = _brackets(S_k[sl], K, X[sl], I["shifted"])
            pos[sl, k, 0] = b_delta - torch.sqrt(K / S_k[sl]) / (2.0 * math.pi) * I["S"]
            pos[sl, k, 1] = -torch.sqrt(S_k[sl] * K) / (2.0 * math.pi) * I["V"] / dldv
    return pos.to(paths.spot.device)
