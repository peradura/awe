"""Fig 3 (paper): external transfer — multi-hop MQAR (Part 5).

Panel (a): tau-sweep accuracy-vs-steps curves (10-seed mean±std bands, no
extrapolation) + fixed-depth frontier + calibrated operating points, as Fig 1.
Panel (b): depth grows with difficulty — dstate mean halt-step vs hop count H.

Usage: PYTHONPATH=src python scripts/fig3_mqarhop.py [--out results]
"""
import argparse
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {  # fixed assignment, matching Fig 1 hues per signal family
    "conv":    ("#0072B2", "-", 2.2, 6, True),
    "dstate":  ("#56B4E9", "-", 2.2, 5, True),
    "recon":   ("#CC79A7", (0, (5, 2)), 1.4, 3, False),
    "entropy": ("#D55E00", (0, (3, 1, 1, 1)), 1.4, 3, False),
}
LABEL = {"conv": "conv", "dstate": "dstate",
         "recon": "recon (state-mism.)", "entropy": "entropy"}


def band(runs, name, grid):
    rows = []
    for r in runs:
        pts = sorted((s, a) for s, a in r["arms"][name]["curve"])  # curve = [steps, acc]
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        row = np.interp(grid, xs, ys)
        row[(grid < xs.min()) | (grid > xs.max())] = np.nan
        rows.append(row)
    A = np.stack(rows) * 100
    full = ~np.isnan(A).any(0)
    return (np.where(full, np.nanmean(A, 0), np.nan),
            np.where(full, np.nanstd(A, 0), np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    runs = [json.load(open(f))
            for f in sorted(glob.glob(f"{args.out}/mqarhop_seed*.json"))]
    assert runs, "no mqarhop_seed*.json"
    T = len(runs[0]["fixed_frontier"])
    grid = np.linspace(1.0, float(T), 121)

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.6, 3.4),
                                  gridspec_kw={"width_ratios": [2.1, 1.0]})

    # (a) persist ceiling, frontier, bands, operating points
    ceil = np.array([r["persist_acc"] for r in runs]) * 100
    ax.axhline(ceil.mean(), color="0.35", lw=1.0, ls=":", zorder=1)
    ax.annotate(f"persist {ceil.mean():.1f}%", (1.06, ceil.mean() + 0.35),
                fontsize=8, color="0.35")
    F = np.stack([np.asarray(r["fixed_frontier"])[:, 0] for r in runs]) * 100
    ax.errorbar(np.arange(1, T + 1), F.mean(0), yerr=F.std(0), color="black",
                ls="--", lw=1.2, marker="o", ms=4, capsize=2, zorder=2,
                label="fixed depth")
    for name, (c, ls, lw, z, emph) in STYLE.items():
        m, s = band(runs, name, grid)
        ax.plot(grid, m, color=c, ls=ls, lw=lw, zorder=z, label=LABEL[name],
                alpha=1.0 if emph else 0.85)
        ax.fill_between(grid, m - s, m + s, color=c, alpha=0.13, lw=0, zorder=z - 1)
        st = np.array([r["arms"][name]["steps"] for r in runs])
        ac = np.array([r["arms"][name]["acc"] for r in runs]) * 100
        ax.errorbar(st.mean(), ac.mean(), xerr=st.std(), yerr=ac.std(), color=c,
                    marker="*" if emph else "s", ms=12 if emph else 5.5,
                    mec="white", mew=0.8, capsize=2, lw=1.0, zorder=z + 4)
    ax.set_xlabel(f"average latent steps used (of {T})", fontsize=9.5)
    ax.set_ylabel("accuracy (%)", fontsize=9.5)
    ax.set_xlim(0.9, T + 0.1)
    ax.grid(alpha=0.22, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    ax.set_title("(a) multi-hop MQAR bake-off", fontsize=9.5)

    # (b) depth grows with hop count (dstate halt-step vs H)
    Hs = sorted(runs[0]["depth_vs_K"].keys(), key=int)
    D = np.array([[r["depth_vs_K"][h] for h in Hs] for r in runs])
    axb.errorbar([int(h) for h in Hs], D.mean(0), yerr=D.std(0),
                 color=STYLE["dstate"][0], marker="o", ms=6, lw=2.0, capsize=3)
    for h, v in zip(Hs, D.mean(0)):
        axb.annotate(f"{v:.2f}", (int(h), v), (0, 8), textcoords="offset points",
                     ha="center", fontsize=8, color="0.15")
    axb.set_xlabel("hop count $H$", fontsize=9.5)
    axb.set_ylabel("dstate mean halt step", fontsize=9.5)
    axb.set_xticks([int(h) for h in Hs])
    axb.set_ylim(2.4, 4.0)
    axb.grid(alpha=0.22, lw=0.5)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    axb.tick_params(labelsize=8.5)
    axb.set_title(r"(b) depth $\propto$ difficulty", fontsize=9.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = f"{args.out}/fig3_mqarhop.{ext}"
        fig.savefig(p, dpi=300)
        print("saved ->", p)


if __name__ == "__main__":
    main()
