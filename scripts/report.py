"""Build the comparison tables and figures from saved run outputs.

Reads runs/2019/results.json, runs/2022/results.json and the per-path P&L
arrays, writes bench/results_2019.txt, bench/results_2022.txt and docs/*.png.
Nothing here trains or simulates; every number comes from the saved runs.
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R19, R22 = ROOT / "runs" / "2019", ROOT / "runs" / "2022"
BENCH, DOCS = ROOT / "bench", ROOT / "docs"
EPS_GRID = [2.0 ** -(i + 5) for i in range(1, 6)]
INK, MUTED, ACCENT, SECOND, THIRD = "#1d1b17", "#8a8580", "#b5462f", "#2f5d8a", "#5d7a3a"


def _ms(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)


def _cvar(loss_pos_is_bad: np.ndarray, alpha: float) -> float:
    """Expected shortfall of X, the average of the worst (1 - alpha) outcomes of -X."""
    losses = np.sort(-loss_pos_is_bad)[::-1]
    k = max(1, int(round((1 - alpha) * len(losses))))
    return float(losses[:k].mean())


def _slope(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def report_2019() -> list[str]:
    R = json.loads((R19 / "results.json").read_text())
    runs, paper, mh = R["runs"], R["paper"], R["model_hedge"]
    q = mh["q"]
    by = lambda pre: [r for r in runs if r["tag"].startswith(pre)]
    out = ["# Buehler, Gonon, Teichmann, Wood (2019), Section 5", "",
           f"Training: {R['args']['steps']:,} steps per run, batch 256, lr 0.005, "
           f"{R['args']['seeds']} seeds. Held-out: {R['args']['test_paths']:,} paths, identical for every run.", ""]
    rows = [("Risk-neutral ATM price q", paper["q"], (q, 0.0), None)]
    for label, pre, key in [("Deep hedge price, 50% CVaR, no costs", "h_a050_e0_rec", "p0_050"),
                            ("Deep hedge price, 99% CVaR, no costs", "h_a099_e0_rec", "p0_099"),
                            ("1% costs, 99% CVaR, recurrent network", "h_a099_e01_rec", "fig5_recurrent"),
                            ("1% costs, 99% CVaR, simpler network", "h_a099_e01_simple", "fig5_simpler")]:
        xs = by(pre)
        rows.append((label, paper[key], _ms([r["p0"] for r in xs]), xs))
    out += ["| claim | paper | reproduced | mean trading gain |", "|---|---:|---:|---:|"]
    for label, pv, (m, s), xs in rows:
        gain = "" if xs is None else f"{st.mean(r['gain_mean'] for r in xs):+.4f}"
        rep = f"{m:.4f}" if s == 0 else f"{m:.4f} ± {s:.4f}"
        out.append(f"| {label} | {pv} | {rep} | {gain} |")
    out += ["", "Mean trading gain is E[P&L] + E[Z] on held-out paths: zero in an arbitrage-free "
            "market without costs, minus the average cost with them. A positive value would mean "
            "the policy is harvesting a flaw in the simulator rather than hedging.", ""]
    out += ["Heston model hedge (stock + variance swap) on the same held-out paths: "
            f"50% CVaR {mh['model_hedge_cvar050']:.4f}, 99% CVaR {mh['model_hedge_cvar099']:.4f}, "
            f"P&L mean {mh['model_hedge_mean']:+.4f}, std {mh['model_hedge_std']:.4f}.", ""]

    # Figure 7: realized CVaRs, normalized to the risk-neutral price.
    out += ["## Figure 7: realized CVaR of each strategy, net of the risk-neutral price", "",
            "| strategy | paper 50% CVaR | ours | paper 99% CVaR | ours |", "|---|---:|---:|---:|---:|"]
    for strat, pre in (("cvar099", "h_a099_e0_rec"), ("cvar050", "h_a050_e0_rec")):
        pnls = [np.load(R19 / f"{r['tag']}_pnl.npy") + q for r in by(pre)]
        c50 = _ms([_cvar(x, 0.5) for x in pnls]); c99 = _ms([_cvar(x, 0.99) for x in pnls])
        pp = paper["fig7"][strat]
        name = "trained on 99% CVaR" if strat == "cvar099" else "trained on 50% CVaR"
        out.append(f"| {name} | {pp['cvar050']} | {c50[0]:.4f} | {pp['cvar099']} | {c99[0]:.4f} |")
    out.append("")

    # Section 5.3: price asymptotics.
    out += ["## Section 5.3: price convergence under small proportional costs", "",
            "| market | paper slope | reproduced slope | theory |", "|---|---:|---:|---:|"]
    slopes = {}
    for mk, name, qv, key in (("bs", "Black-Scholes", 2.2870535, "slope_bs"), ("h", "Heston", q, "slope_heston")):
        per_seed, curves = [], []
        for s in range(R["args"]["seeds"]):
            pts = [[r for r in runs if r["tag"] == f"{mk}_ent_eps{i}_s{s}"][0]["p0"] for i in range(5)]
            xs, ys = [math.log(e) for e in EPS_GRID], [math.log(p - qv) for p in pts]
            per_seed.append(_slope(xs, ys)); curves.append(ys)
        m, sd = _ms(per_seed)
        slopes[mk] = (name, [math.log(e) for e in EPS_GRID], np.mean(curves, axis=0), m)
        out.append(f"| {name} | {paper[key]} | {m:.3f} ± {sd:.3f} | 0.667 |")
    out.append("")
    _figures_2019(q, slopes)
    return out


def _figures_2019(q, slopes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 160, "axes.edgecolor": MUTED, "axes.labelcolor": INK})
    deep = np.load(R19 / "h_a050_e0_rec_s0_pnl.npy") + q
    model = np.load(R19 / "model_hedge_pnl.npy")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 3.4))
    bins = np.linspace(-3, 3, 61)
    a1.hist(model, bins=bins, color=SECOND, alpha=0.35, label="Heston model hedge")
    a1.hist(deep, bins=bins, histtype="step", color=ACCENT, lw=1.4, label="deep hedge, 50% CVaR")
    a1.set_xlabel("hedging error, net of risk-neutral price"); a1.set_ylabel("paths")
    a1.set_title("Deep hedge recovers the model hedge (paper Fig. 2)", fontsize=9.5)
    a1.legend(frameon=False, fontsize=8)
    for ax, mk, color in ((a2, "bs", ACCENT), (a3, "h", SECOND)):
        name, xs, ys, m = slopes[mk]
        ax.plot(xs, ys, "o", color=color)
        b = np.polyfit(xs, ys, 1)
        grid = np.linspace(min(xs), max(xs), 20)
        ax.plot(grid, np.polyval(b, grid), color=color, lw=1.2, label=f"fit, slope {m:.2f}")
        ax.plot(grid, np.polyval([2 / 3, b[1] + (b[0] - 2 / 3) * np.mean(xs)], grid),
                color=MUTED, lw=1, ls="--", label="theory, slope 2/3")
        ax.set_xlabel("log eps"); ax.set_ylabel("log(p_eps - q)")
        ax.set_title(f"{name}: price vs cost (paper Fig. {10 if mk == 'bs' else 11})", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    fig.savefig(DOCS / "reproduction_2019.png", bbox_inches="tight")


def report_2022() -> list[str]:
    R = json.loads((R22 / "results.json").read_text())
    paper, base = R["paper"], R["baselines"]
    Ks = [0.9, 0.95, 1.0, 1.05, 1.1]
    a = R["args"]

    def vals(rows, K):
        return _ms([r["utility"] for r in rows if abs(r["K"] - K) < 1e-9])

    def ac(K):
        return _ms([r["table"][str(K)] for r in R["actor_critic"]])

    out = ["# Murray, Wood, Buehler, Wiese, Pakkanen (2022), Table 1", "",
           f"v0 = {R['v0']:.5f}, fitted on the single BS-delta cell at K = 1.00 (scripts/calibrate_v0.py); "
           "the other four BS-delta cells are predictions.", "",
           f"Held-out: {a['n_test']:,} paths shared by every policy. Vanilla deep hedging: "
           f"{a['vanilla_steps']:,} steps of 256 paths per strike. Actor-critic: {a['ac_iters']:,} "
           "iterations of 2048 episodes, one model for all strikes and all lambda (the paper ran "
           f"100,000 iterations). Seeds {a['seeds']}, reported as mean ± std across seeds.", "",
           "Exponential utility at lambda = 0.1 of the hedged P&L, higher is better:", "",
           "| K | BS delta, paper | ours | vanilla DH, paper | ours | actor-critic, paper | ours |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    fig_rows = []
    for K in Ks:
        pb, pv, pa = paper[str(K)]
        bs = base[str(K)]["bs_delta"]
        v, c = vals(R["vanilla"], K), ac(K)
        out.append(f"| {K:.2f} | {pb} | {bs:.3f} | {pv} | {v[0]:.3f} ± {v[1]:.3f} | {pa} | {c[0]:.3f} ± {c[1]:.3f} |")
        fig_rows.append({"K": K, "paper": (pb, pv, pa), "ours": (bs, v[0], c[0]),
                         "ww": base[str(K)]["whalley_wilmott"]})
    out += ["", "Gain over the BS delta hedge, which is what Table 1 is about:", "",
            "| K | vanilla DH, paper | ours | actor-critic, paper | ours |", "|---|---:|---:|---:|---:|"]
    for r in fig_rows:
        (pb, pv, pa), (bs, v, c) = r["paper"], r["ours"]
        out.append(f"| {r['K']:.2f} | {pv - pb:+.2f} | {v - bs:+.3f} | {pa - pb:+.2f} | {c - bs:+.3f} |")
    out += ["", "## Not in the paper: an analytic band and an ablation", "",
            "Whalley-Wilmott is the classic closed-form no-trade band for proportional costs "
            "(risk aversion set to the same lambda). The ablation is vanilla deep hedging with the "
            "previous holding removed from the network input.", "",
            "| K | BS delta | Whalley-Wilmott | vanilla DH | vanilla DH, no holding input | actor-critic |",
            "|---|---:|---:|---:|---:|---:|"]
    for r in fig_rows:
        K = r["K"]
        nh = vals(R["vanilla_no_holding"], K)[0] if "vanilla_no_holding" in R else float("nan")
        out.append(f"| {K:.2f} | {r['ours'][0]:.3f} | {r['ww']:.3f} | {r['ours'][1]:.3f} | {nh:.3f} | "
                   f"{r['ours'][2]:.3f} |")
    out += ["", "## One actor-critic across risk aversions (K = 1.00)", "",
            "The trained network conditioned on each lambda and scored at that lambda, against the "
            "same network frozen at lambda = 0.1 and scored at the same lambda. Higher is better.", "",
            "| lambda | conditioned on lambda | frozen at 0.1 | gain from conditioning |", "|---:|---:|---:|---:|"]
    lam_rows: dict[float, list] = {}
    for r in R["actor_critic"]:
        for row in r["lam_sweep"]:
            lam_rows.setdefault(row["lam"], []).append((row["conditioned"], row["frozen_at_0.1"]))
    for lam in sorted(lam_rows):
        c = st.mean(x[0] for x in lam_rows[lam]); f = st.mean(x[1] for x in lam_rows[lam])
        out.append(f"| {lam:g} | {c:.3f} | {f:.3f} | {c - f:+.3f} |")
    out.append("")
    _figure_2022(fig_rows)
    return out


def _figure_2022(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 160, "axes.edgecolor": MUTED, "axes.labelcolor": INK})
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    Ks = [r["K"] for r in rows]
    for idx, label, color in ((1, "vanilla deep hedging", SECOND), (2, "actor-critic", ACCENT)):
        ax.plot(Ks, [r["paper"][idx] - r["paper"][0] for r in rows], "o--", color=color, alpha=0.45,
                label=f"{label}, paper")
        ax.plot(Ks, [r["ours"][idx] - r["ours"][0] for r in rows], "o-", color=color,
                label=f"{label}, ours")
    ax.plot(Ks, [r["ww"] - r["ours"][0] for r in rows], "s:", color=THIRD, label="Whalley-Wilmott band, ours")
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.set_xlabel("strike"); ax.set_ylabel("utility gain over BS delta")
    ax.set_title("Learned hedges vs Black-Scholes delta under 0.2% costs (paper Table 1)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    fig.savefig(DOCS / "reproduction_2022.png", bbox_inches="tight")


def main() -> None:
    BENCH.mkdir(exist_ok=True)
    for name, fn, src in (("results_2019.txt", report_2019, R19), ("results_2022.txt", report_2022, R22)):
        if not (src / "results.json").exists():
            print(f"skip {name}: no results yet")
            continue
        text = "\n".join(fn())
        (BENCH / name).write_text(text + "\n")
        print(text + "\n")


if __name__ == "__main__":
    main()
