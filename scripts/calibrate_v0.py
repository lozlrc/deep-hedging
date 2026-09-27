"""Recover the initial variance the 2022 paper never states.

Murray et al. (2022) give kappa, theta, xi, rho, S0 and T but not v0. It is
fitted on exactly ONE cell of their Table 1, the Black-Scholes delta hedge at
K = 1.00 (utility -3.78 at lambda = 0.1), and the other four Black-Scholes
cells are then predictions, not fits. If one free parameter reproduces all five
the setup is right; if it cannot, something else in the setup is unstated.

Setup, as the paper describes it: short call, notional 100 on S0 = 1, hedged
daily with the spot only, proportional cost 0.002 on each trade, the book valued
at maturity with no liquidation charge, exponential utility at lambda = 0.1.
The BS delta uses the square root of the variance process as implied vol.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch

from dhedge.baselines import bs_delta_positions
from dhedge.heston import heston_paths
from dhedge.pnl import call_payoff, hedge_pnl
from dhedge.risk import exp_utility
from dhedge.spec import MURRAY_2022, T_30D

PAPER_BS = {0.90: -11.10, 0.95: -7.06, 1.00: -3.78, 1.05: -1.52, 1.10: -0.43}
NOTIONAL, EPS, LAM, SUBSTEPS = 100.0, 0.002, 0.1, 4
ROOT = Path(__file__).resolve().parents[1]


def bs_utility(v0: float, K: float, n: int = 200_000, seed: int = 1) -> float:
    params = dataclasses.replace(MURRAY_2022, v0=v0)
    paths = heston_paths(params, n, seed=seed, substeps=SUBSTEPS)
    pos = bs_delta_positions(paths, K, T_30D, notional=NOTIONAL)
    Z = NOTIONAL * call_payoff(paths.spot, K)
    X = hedge_pnl(paths.instruments, pos.to(paths.instruments.dtype), Z, eps=EPS, liquidate=False)
    return float(exp_utility(X, LAM))


def fit_v0(target: float = PAPER_BS[1.00], lo: float = 1e-4, hi: float = 0.6, iters: int = 30) -> float:
    """Bisection: utility falls monotonically as the initial variance rises."""
    f_lo, f_hi = bs_utility(lo, 1.0) - target, bs_utility(hi, 1.0) - target
    if f_lo * f_hi > 0:
        raise RuntimeError(f"target not bracketed: {f_lo + target:.3f} .. {f_hi + target:.3f}")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = bs_utility(mid, 1.0) - target
        if (f > 0) == (f_lo > 0):
            lo, f_lo = mid, f
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    torch.set_num_threads(2)
    v0 = fit_v0()
    print(f"fitted v0 = {v0:.5f}  (implied vol {v0 ** 0.5:.1%}; long-run theta = "
          f"{MURRAY_2022.theta} -> {MURRAY_2022.theta ** 0.5:.1%})\n")
    rows = []
    print(f"{'K':>5} {'paper':>8} {'ours':>9} {'diff':>8}  role")
    for K, paper in PAPER_BS.items():
        ours = bs_utility(v0, K, n=400_000, seed=7)      # fresh paths for the check
        role = "FIT" if K == 1.00 else "prediction"
        rows.append({"K": K, "paper": paper, "ours": ours, "role": role})
        print(f"{K:>5.2f} {paper:>8.2f} {ours:>9.3f} {ours - paper:>+8.3f}  {role}")
    out = ROOT / "runs" / "2022"
    out.mkdir(parents=True, exist_ok=True)
    (out / "v0_calibration.json").write_text(json.dumps({"v0": v0, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
