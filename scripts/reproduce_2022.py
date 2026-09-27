"""Reproduce Table 1 of Murray, Wood, Buehler, Wiese, Pakkanen (2022), plus the
paper's central claim that ONE actor-critic serves every risk aversion.

Table 1 is the exponential utility (lambda = 0.1) of the hedged P&L of a short
call, notional 100 on S0 = 1, for five strikes under three policies: the
Black-Scholes delta hedge, vanilla (pathwise) deep hedging trained separately
per strike, and the risk-averse actor-critic trained once across all strikes and
all risk aversions. v0 is not stated in the paper; it was fitted on the single
BS-delta cell at K = 1.00 by scripts/calibrate_v0.py and is read from there.

Every method is scored the same way, by Monte Carlo on one shared held-out set,
through dhedge.pnl.hedge_pnl. Utilities are never read off the critic.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "2022"
NOTIONAL, EPS, LAM, SUBSTEPS = 100.0, 0.002, 0.1, 4
TABLE_K = [0.90, 0.95, 1.00, 1.05, 1.10]
TRAIN_K = [round(0.9 + 0.025 * i, 3) for i in range(9)]     # the paper's strike grid
LAM_GRID = [1e-4, 0.005, 0.1, 1.0]                           # the paper's dynamic-test values
TEST_SEED = 20260927

# (BS delta, vanilla deep hedging, actor-critic), Table 1 at lambda = 0.1.
PAPER = {0.90: (-11.10, -10.87, -10.91), 0.95: (-7.06, -6.74, -6.68), 1.00: (-3.78, -3.36, -3.41),
         1.05: (-1.52, -1.14, -1.11), 1.10: (-0.43, -0.26, -0.27)}


def params():
    from dhedge.spec import MURRAY_2022
    v0 = json.loads((OUT / "v0_calibration.json").read_text())["v0"]
    return dataclasses.replace(MURRAY_2022, v0=float(v0))


def sample(n: int, seed: int):
    from dhedge.heston import heston_paths
    return heston_paths(params(), n, seed=seed, substeps=SUBSTEPS)


def payoff_fn(K: float):
    from dhedge.pnl import call_payoff
    return lambda p: NOTIONAL * call_payoff(p.spot, K)


def utility(paths, holdings, K: float, lam: float = LAM) -> tuple[float, torch.Tensor]:
    from dhedge.pnl import hedge_pnl
    from dhedge.risk import exp_utility
    X = hedge_pnl(paths.instruments, holdings.to(paths.instruments.dtype), payoff_fn(K)(paths),
                  eps=EPS, liquidate=False)
    return float(exp_utility(X, lam)), X


def run_baselines(n_test: int) -> dict:
    """BS delta (Table 1 column 1) and, as an extra not in the paper, the
    Whalley-Wilmott no-trade band, the classic analytic answer to costs."""
    torch.set_num_threads(2)
    from dhedge.baselines import bs_delta_positions, whalley_wilmott_positions
    from dhedge.spec import T_30D
    test = sample(n_test, TEST_SEED)
    out = {}
    for K in TABLE_K:
        u_bs, _ = utility(test, bs_delta_positions(test, K, T_30D, notional=NOTIONAL), K)
        ww = whalley_wilmott_positions(test, K, T_30D, eps=EPS, risk_aversion=LAM, notional=NOTIONAL)
        u_ww, _ = utility(test, ww, K)
        out[K] = {"bs_delta": u_bs, "whalley_wilmott": u_ww}
    return out


def run_vanilla(job: dict) -> dict:
    """Vanilla deep hedging, one model per strike, pathwise on U_0.1. The paper
    used the actor's architecture (3 x 256, sigmoid). The network sees its
    previous holding (semi-recurrent), as in both papers: with costs the best
    trade depends on what is already held, and the actor gets the holding
    through its portfolio Greeks. semi_recurrent=False is kept as an ablation.
    Holdings are about 50 shares on notional 100, so the spot is priced per
    100 shares during training; the P&L is exactly invariant to that
    (tests/test_scaling.py)."""
    torch.set_num_threads(1)
    from dhedge.deep_hedge import SharedNet, heston_features, rollout_positions, train_pathwise
    from dhedge.scaling import rescale, scaled_sampler
    K, seed = job["K"], job["seed"]
    scales = torch.tensor([NOTIONAL], dtype=torch.float64)
    torch.manual_seed(seed)
    net = SharedNet(feat_dim=2, d=1, hidden=(256, 256, 256), activation="sigmoid", time_feature=True,
                    semi_recurrent=job.get("semi_recurrent", True))
    t0 = time.time()
    train_pathwise(net, scaled_sampler(sample, scales), payoff_fn(K), heston_features, risk="entropic",
                   lam=LAM, eps=EPS, steps=job["steps"], batch=256, lr=1e-3, liquidate=False,
                   seed=seed, n_pool=100_000)
    net.eval()
    test = rescale(sample(job["n_test"], TEST_SEED), scales)
    with torch.no_grad():
        pos = rollout_positions(net, test, heston_features)
    u, _ = utility(test, pos, K)
    return {**job, "utility": u, "train_s": time.time() - t0}


def run_actor_critic(job: dict) -> dict:
    from dhedge.actor_critic import ActorCritic, default_device
    torch.set_num_threads(2)
    dev = job.get("device") or default_device()
    ac = ActorCritic(notional=NOTIONAL, eps=EPS, device=dev, seed=job["seed"])
    t0 = time.time()
    hist = ac.train(sample, strikes=TRAIN_K, n_iters=job["iters"], batch=2048, seed=job["seed"],
                    max_seconds=None)
    train_s = time.time() - t0
    test = sample(job["n_test"], TEST_SEED)
    table = {}
    for K in TABLE_K:
        u, _ = utility(test, ac.rollout_holdings(test, strikes=K, lam=LAM), K)
        table[K] = u
    # One model, many risk aversions: score the policy conditioned on each lambda
    # at that same lambda, against the same policy frozen at lambda = 0.1.
    lam_rows = []
    for lam in LAM_GRID:
        u_own, _ = utility(test, ac.rollout_holdings(test, strikes=1.0, lam=lam), 1.0, lam)
        u_fix, _ = utility(test, ac.rollout_holdings(test, strikes=1.0, lam=LAM), 1.0, lam)
        lam_rows.append({"lam": lam, "conditioned": u_own, "frozen_at_0.1": u_fix})
    return {**job, "table": table, "lam_sweep": lam_rows, "train_s": train_s,
            "iters_done": len(hist), "device": dev}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-test", type=int, default=200_000)
    ap.add_argument("--vanilla-steps", type=int, default=15_000)
    ap.add_argument("--ac-iters", type=int, default=15_000)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 10])   # spaced: no shared training paths
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only-vanilla", action="store_true",
                    help="rerun vanilla deep hedging only, keeping saved baselines and actor-critic")
    ap.add_argument("--no-holding", action="store_true",
                    help="ablation: vanilla network without its previous holding as input")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    vjobs = [{"K": K, "seed": s, "steps": args.vanilla_steps, "n_test": args.n_test,
              "semi_recurrent": not args.no_holding} for s in args.seeds for K in TABLE_K]
    key = "vanilla_no_holding" if args.no_holding else "vanilla"
    if args.only_vanilla:
        saved = json.loads((OUT / "results.json").read_text())
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            saved[key] = []
            for r in ex.map(run_vanilla, vjobs):
                saved[key].append(r)
                print(f"  {key} K={r['K']:.2f} seed={r['seed']}  U={r['utility']:.3f}  ({r['train_s']:.0f}s)",
                      flush=True)
        (OUT / "results.json").write_text(json.dumps(saved, indent=2, default=float))
        print(f"done in {time.time() - t0:.0f}s -> {OUT / 'results.json'}")
        return
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        base = ex.submit(run_baselines, args.n_test)
        acs = [ex.submit(run_actor_critic, {"seed": s, "iters": args.ac_iters, "n_test": args.n_test})
               for s in args.seeds]
        vanilla = []
        for r in ex.map(run_vanilla, vjobs):
            vanilla.append(r)
            print(f"  vanilla K={r['K']:.2f} seed={r['seed']}  U={r['utility']:.3f}  ({r['train_s']:.0f}s)", flush=True)
        baselines = base.result()
        print("  baselines done", flush=True)
        ac = [f.result() for f in acs]
        for r in ac:
            print(f"  actor-critic seed={r['seed']} done ({r['train_s']:.0f}s on {r['device']})", flush=True)
    (OUT / "results.json").write_text(json.dumps(
        {"args": vars(args), "paper": {str(k): v for k, v in PAPER.items()},
         "v0": params().v0, "baselines": {str(k): v for k, v in baselines.items()},
         key: vanilla, "actor_critic": ac}, indent=2, default=float))
    print(f"done in {time.time() - t0:.0f}s -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
