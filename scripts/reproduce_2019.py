"""Reproduce the numerical claims of Buehler, Gonon, Teichmann, Wood (2019),
Section 5, and compare each against the published value.

Every experiment trains the paper's per-date semi-recurrent network on
simulated paths and scores it on a large held-out set. Claims:

  q        risk-neutral ATM call price in the Heston setup           1.69
  p0_050   deep-hedge price, 50% CVaR, no costs (Section 5.2)        1.94
  p0_099   deep-hedge price, 99% CVaR, no costs (Section 5.2)        3.49
  fig7     realized CVaRs of both strategies, normalized by q        table
  fig5     99% CVaR with eps = 0.01: recurrent vs simpler price      5.5137 vs 6.7446
  slopes   log(p_eps - q) vs log(eps), entropic lambda = 1           0.67 (BS), 0.71 (Heston)

Runs are single-threaded and parallel across processes: these networks are
so small that torch threading makes them slower, measured at 3x.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "2019"
K_HESTON = 100.0
EPS_GRID = [2.0 ** -(i + 5) for i in range(1, 6)]   # 2^-6 .. 2^-10, as in Section 5.3
# Section 2 defines delta_n = 0 (the book is liquidated at T), then states: "In
# our numerical examples we have assumed zero transaction costs at maturity."
# So liquidation happens but is not charged. Charging it adds eps * |delta| * S_T,
# which inflated every cost-bearing price here in proportion to eps until fixed.
LIQUIDATION_COST = False

PAPER = {
    "q": 1.69, "p0_050": 1.94, "p0_099": 3.49,
    "fig5_recurrent": 5.5137, "fig5_simpler": 6.7446,
    "fig7": {"cvar099": {"mean_loss": 0.2635, "cvar050": 0.527, "cvar099": 1.8034},
             "cvar050": {"mean_loss": 0.1514, "cvar050": 0.2531, "cvar099": 2.3631}},
    "slope_bs": 0.67, "slope_heston": 0.71,
}


def _market(kind: str):
    """Path sampler plus (feat_dim, d). Heston instruments are rescaled so the
    variance-swap hedge ratio is order one; the P&L is exactly invariant to this
    (tests/test_scaling.py), so every reported price is unaffected."""
    from dhedge.heston import heston_paths
    from dhedge.markets import gbm_paths
    from dhedge.scaling import instrument_scales, scaled_sampler
    from dhedge.spec import BUEHLER_2019, N_STEPS, T_30D
    if kind == "heston":
        raw = lambda n, s: heston_paths(BUEHLER_2019, n, seed=s, with_variance_swap=True)
        scales = instrument_scales(raw(100_000, 777))       # one fixed pilot, same for every job
        return scaled_sampler(raw, scales), 2, 2
    return (lambda n, s: gbm_paths(n, 100.0, 0.2, T_30D, N_STEPS, seed=s)), 1, 1


def run_job(job: dict) -> dict:
    """Train one deep hedge and score it on the shared held-out set."""
    import torch
    torch.set_num_threads(1)
    from dhedge.deep_hedge import (PerStepNet, bs_features, heston_features,
                                   indifference_price, train_pathwise)
    from dhedge.pnl import call_payoff
    from dhedge.spec import N_STEPS

    sample, feat_dim, d = _market(job["market"])
    feats = heston_features if job["market"] == "heston" else bs_features
    payoff = lambda p: call_payoff(p.spot, K_HESTON)
    risk_kw = {"alpha": job["alpha"]} if job["risk"] == "cvar" else {"lam": job["lam"]}

    torch.manual_seed(job["seed"])
    policy = PerStepNet(N_STEPS, feat_dim, d, semi_recurrent=job["recurrent"])
    t0 = time.time()
    hist = train_pathwise(policy, sample, payoff, feats, risk=job["risk"], eps=job["eps"],
                          steps=job["steps"], batch=256, lr=0.005, liquidate=LIQUIDATION_COST,
                          seed=job["seed"], n_pool=job["pool"], **risk_kw)
    train_s = time.time() - t0

    test = sample(job["test_paths"], 424242)          # identical held-out set for every job
    p0, pnl = indifference_price(policy, test, payoff, feats, risk=job["risk"], eps=job["eps"],
                                 liquidate=LIQUIDATION_COST, **risk_kw)
    tag = job["tag"]
    RUNS.mkdir(parents=True, exist_ok=True)
    np.save(RUNS / f"{tag}_pnl.npy", pnl.detach().cpu().numpy().astype(np.float64))
    losses = [float(x) for x in hist.get("loss", [])]
    Z = payoff(test)
    # Arbitrage check: in an arbitrage-free market the hedge's trading gains have
    # zero mean, so E[P&L] must equal -E[Z] up to costs. A positive mean gain
    # means the policy is harvesting a flaw in the simulator, not hedging.
    gain_mean = float(pnl.mean() + Z.to(pnl.dtype).mean())
    gain_se = float((pnl + Z.to(pnl.dtype)).std() / (len(pnl) ** 0.5))
    return {**job, "p0": float(p0), "train_s": train_s,
            "payoff_mean": float(Z.mean()), "gain_mean": gain_mean, "gain_se": gain_se,
            "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None}


def run_model_hedge(test_paths: int) -> dict:
    """Heston model hedge on the same held-out set: the zero-cost benchmark of Figure 2."""
    import torch
    torch.set_num_threads(4)
    from dhedge.heston_pricing import heston_call, model_hedge_positions
    from dhedge.pnl import call_payoff, hedge_pnl
    from dhedge.risk import cvar
    from dhedge.spec import BUEHLER_2019, T_30D
    from dhedge.heston import heston_paths
    # Raw, unscaled paths: model_hedge_positions returns original-unit holdings,
    # so it must be scored against original-unit instruments.
    test = heston_paths(BUEHLER_2019, test_paths, seed=424242, with_variance_swap=True)
    q = float(heston_call(100.0, BUEHLER_2019.v0, T_30D, K_HESTON, BUEHLER_2019))
    pos = model_hedge_positions(test, BUEHLER_2019, K_HESTON, T_30D)
    pnl = hedge_pnl(test.instruments, pos.to(test.instruments.dtype), call_payoff(test.spot, K_HESTON),
                    eps=0.0, premium=q)
    RUNS.mkdir(parents=True, exist_ok=True)
    np.save(RUNS / "model_hedge_pnl.npy", pnl.cpu().numpy())
    raw = pnl - q                                   # premium-free P&L, as the deep hedges report it
    return {"q": q, "model_hedge_mean": float(pnl.mean()), "model_hedge_std": float(pnl.std()),
            "model_hedge_cvar050": float(cvar(raw, 0.5)), "model_hedge_cvar099": float(cvar(raw, 0.99))}


def jobs(steps: int, seeds: int, test_paths: int, pool: int) -> list[dict]:
    base = {"steps": steps, "test_paths": test_paths, "pool": pool}
    out = []
    for s in range(seeds):
        out += [
            {**base, "tag": f"h_a050_e0_rec_s{s}", "market": "heston", "risk": "cvar", "alpha": 0.5,
             "eps": 0.0, "recurrent": True, "seed": s},
            {**base, "tag": f"h_a099_e0_rec_s{s}", "market": "heston", "risk": "cvar", "alpha": 0.99,
             "eps": 0.0, "recurrent": True, "seed": s},
            {**base, "tag": f"h_a099_e01_rec_s{s}", "market": "heston", "risk": "cvar", "alpha": 0.99,
             "eps": 0.01, "recurrent": True, "seed": s},
            {**base, "tag": f"h_a099_e01_simple_s{s}", "market": "heston", "risk": "cvar", "alpha": 0.99,
             "eps": 0.01, "recurrent": False, "seed": s},
        ]
        for i, e in enumerate(EPS_GRID):
            out += [
                {**base, "tag": f"bs_ent_eps{i}_s{s}", "market": "bs", "risk": "entropic", "lam": 1.0,
                 "eps": e, "recurrent": True, "seed": s},
                {**base, "tag": f"h_ent_eps{i}_s{s}", "market": "heston", "risk": "entropic", "lam": 1.0,
                 "eps": e, "recurrent": True, "seed": s},
            ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--test-paths", type=int, default=500_000)
    ap.add_argument("--pool", type=int, default=100_000)
    ap.add_argument("--only-costs", action="store_true",
                    help="rerun only cost-bearing jobs and keep existing zero-cost results")
    args = ap.parse_args()

    todo = jobs(args.steps, args.seeds, args.test_paths, args.pool)
    prior = []
    if args.only_costs:
        todo = [j for j in todo if j["eps"] > 0]
        prev = RUNS / "results.json"
        if prev.exists():
            prior = [r for r in json.loads(prev.read_text())["runs"] if r["eps"] == 0]
    print(f"{len(todo)} training runs, {args.steps} steps each, {args.workers} workers, "
          f"{args.test_paths} held-out paths", flush=True)
    t0 = time.time()
    results = list(prior)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        mh = ex.submit(run_model_hedge, args.test_paths)
        for i, r in enumerate(ex.map(run_job, todo), 1):
            results.append(r)
            print(f"  [{i}/{len(todo)}] {r['tag']:<22} p0={r['p0']:.4f}  ({r['train_s']:.0f}s)", flush=True)
        model = mh.result()
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "results.json").write_text(json.dumps(
        {"args": vars(args), "paper": PAPER, "model_hedge": model, "runs": results}, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {RUNS / 'results.json'}")


if __name__ == "__main__":
    main()
