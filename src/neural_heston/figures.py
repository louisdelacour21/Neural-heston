"""Figures of the validation report, from report/results.json and report/raw.npz (run validate first).

  python -m neural_heston.figures"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import heston, model
from .validate import GREEKS, REPORT, implied_vol

BLUE, ORANGE, GREY = "#2563eb", "#ea580c", "#6b7280"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def binned(ax, v, err, bins, label, color):
    idx = np.digitize(v, bins) - 1
    mid = (bins[1:] + bins[:-1]) / 2
    q50 = [np.nanmedian(np.abs(err[idx == i])) for i in range(len(mid))]
    q99 = [np.nanquantile(np.abs(err[idx == i]), 0.99) for i in range(len(mid))]
    ax.plot(mid, q50, color=color, label=f"{label} median")
    ax.plot(mid, q99, color=color, ls="--", label=f"{label} 99th pct")


def main():
    res = json.load(open(os.path.join(REPORT, "results.json")))
    raw = np.load(os.path.join(REPORT, "raw.npz"))
    names = [n for n in ("dml", "plain") if n in res]
    colors = {"dml": BLUE, "plain": ORANGE}
    x = raw["dml__x"]

    # 1. implied-vol error against maturity and moneyness
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    for n in names:
        e = raw[f"{n}__iv_err_bp"]
        binned(axs[0], np.sqrt(x[:, 6]), e, np.linspace(np.sqrt(0.05), np.sqrt(2), 12), n, colors[n])
        binned(axs[1], x[:, 5] / np.sqrt(x[:, 6]), e, np.linspace(-0.8, 0.5, 12), n, colors[n])
    axs[0].set(xlabel="sqrt(T)", ylabel="|implied vol error| (bp)", yscale="log")
    axs[1].set(xlabel="k / sqrt(T)", yscale="log")
    axs[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT, "fig_iv_error.png"))

    # 2. smiles: network against reference for three parameter sets
    m = model.load(os.path.join(REPORT, "..", "models", "dml.pt"))
    sets = [("calm", [2.0, 0.03, 0.3, -0.5, 0.02]), ("stressed, Feller violated", [0.8, 0.12, 0.9, -0.9, 0.15]),
            ("short-dated skew", [4.0, 0.05, 0.7, -0.8, 0.08])]
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.8), sharey=False)
    for ax, (title, p) in zip(axs, sets):
        for T, ls in [(0.08, ":"), (0.5, "--"), (1.5, "-")]:
            xs = np.linspace(-0.8, 0.5, 60) * np.sqrt(T)
            pts = np.column_stack([np.tile(p, (60, 1)), xs, np.full(60, T)])
            ax.plot(xs, 100 * implied_vol(heston.call(pts).numpy(), xs, np.full(60, T)), color=GREY, ls=ls, lw=2.5, alpha=0.5)
            ax.plot(xs, 100 * implied_vol(m.predict(pts)[0], xs, np.full(60, T)), color=BLUE, ls=ls, lw=1, label=f"T={T}")
        ax.set(title=title, xlabel="k = ln(K/F)")
    axs[0].set_ylabel("implied vol (%)")
    axs[2].legend(frameon=False, fontsize=7, title="grey: reference", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT, "fig_smiles.png"))

    # 3. Greeks: network (automatic differentiation) against exact reference Greeks
    fig, axs = plt.subplots(2, 4, figsize=(9, 4.4))
    gn, gr = raw["dml__greeks_net"], raw["dml__greeks_ref"]
    for i, ax in enumerate(axs.ravel()):
        lo, hi = np.quantile(gr[:, i], [0.001, 0.999])
        ax.scatter(gr[:, i], gn[:, i], s=1, color=BLUE, alpha=0.3, rasterized=True)
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.8)
        ax.set(title=f"{GREEKS[i]}  (rel. RMSE {100 * res['dml']['greeks'][GREEKS[i]]['rel_rmse']:.1f}%)", xlim=(lo, hi), ylim=(lo, hi))
        ax.title.set_size(8)
    fig.supxlabel("reference", fontsize=8)
    fig.supylabel("network", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT, "fig_greeks.png"))

    # 4. calibration: recovered against true parameters, network and reference pricer
    cal = res["dml"].get("calibration")
    if cal:
        labels = ["kappa", "theta", "sigma", "rho", "v0"]
        fig, axs = plt.subplots(1, 5, figsize=(10, 2.3))
        truth = np.array([r["truth"] for r in cal])
        for name, c, mk in [("reference", GREY, "o"), ("network", BLUE, "x")]:
            q = np.array([r[name]["params"] for r in cal])
            for j, ax in enumerate(axs):
                ax.scatter(truth[:, j], q[:, j], s=12, color=c, marker=mk, label=name)
        for j, ax in enumerate(axs):
            lo, hi = truth[:, j].min(), truth[:, j].max()
            ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.6)
            ax.set(title=labels[j], xlabel="true")
        axs[0].set_ylabel("calibrated")
        axs[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(REPORT, "fig_calibration.png"))
    print("figures written to report/")


if __name__ == "__main__":
    sys.exit(main())
