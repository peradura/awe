"""Fig 1 (paper): multi-seed tau-sweep bands for the canonical bake-off (Part 4c).

Replaces the seed-0-only PNG: for each signal, every seed's tau-sweep
(acc, avg-steps) curve is interpolated onto a common steps grid and drawn as a
mean line with a ±1 std band (population std, matching scripts/aggregate.py).
Overlaid: fixed-depth frontier (mean±std), no-halt ceiling, and each signal's
calibrated-tau operating point (mean±std error bars).

Design notes: Okabe-Ito CVD-safe palette (validated); convergence-family winners
(conv/dstate) solid+emphasized, magnitude-family losers dashed/muted — identity
survives grayscale via linestyle, not color alone.

Usage: PYTHONPATH=src python scripts/fig1_bakeoff_band.py [--out results]
"""
import argparse
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed assignment (never cycled); winners first, losers muted + dashed
STYLE = {  # name: (color, linestyle, linewidth, zorder, emphasized)
    "conv":   ("#0072B2", "-",  2.2, 6, True),
    "dstate": ("#56B4E9", "-",  2.2, 5, True),
    "dent":   ("#009E73", (0, (3, 1.5)), 1.4, 4, False),
    "rnorm":  ("#E69F00", (0, (1, 1)),   1.4, 3, False),
    "recon":  ("#CC79A7", (0, (5, 2)),   1.4, 3, False),
    "ent":    ("#D55E00", (0, (3, 1, 1, 1)), 1.4, 3, False),
}
LABEL = {"conv": "conv (sym-KL)", "dstate": "dstate (‖Δs‖²)", "dent": "dent (|Δent|)",
         "rnorm": "rnorm", "recon": "recon (decod.)", "ent": "ent (entropy)"}


def load(out):
    files = sorted(glob.glob(f"{out}/bakeoff_reachp2_s*.json"))
    assert files, f"no bakeoff_reachp2_s*.json under {out}/"
    return [json.load(open(f)) for f in files]


def band(runs, name, grid):
    """Interpolate each seed's (steps -> acc) tau-sweep onto `grid`. NaN outside a
    seed's observed steps range (no extrapolation); the band exists only where ALL
    seeds have coverage — flat edge-hold plateaus would read as fake data."""
    rows = []
    for r in runs:
        pts = sorted((s, a) for a, s in r["arms"][name]["curve"])  # curve = [acc, steps]
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        row = np.interp(grid, xs, ys)
        row[(grid < xs.min()) | (grid > xs.max())] = np.nan
        rows.append(row)
    A = np.stack(rows) * 100
    full = ~np.isnan(A).any(0)                      # keep grid points all seeds cover
    m = np.where(full, np.nanmean(A, 0), np.nan)
    s = np.where(full, np.nanstd(A, 0), np.nan)     # population std, repo convention
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    runs = load(args.out)
    T = len(runs[0]["fixed_frontier"])
    grid = np.linspace(1.0, float(T), 121)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    # no-halt ceiling
    ceil = np.array([r["nohalt_acc"] for r in runs]) * 100
    ax.axhline(ceil.mean(), color="0.35", lw=1.0, ls=":", zorder=1)
    ax.annotate(f"no-halt ceiling {ceil.mean():.1f}%", (1.05, ceil.mean() + 0.25),
                fontsize=8, color="0.35")

    # fixed-depth frontier (mean±std across seeds at each integer depth)
    F = np.stack([np.asarray(r["fixed_frontier"]) for r in runs]) * 100
    depths = np.arange(1, T + 1)
    ax.errorbar(depths, F.mean(0), yerr=F.std(0), color="black", ls="--", lw=1.2,
                marker="o", ms=4, capsize=2, zorder=2, label="fixed depth")

    # tau-sweep bands + calibrated operating points
    for name, (c, ls, lw, z, emph) in STYLE.items():
        m, s = band(runs, name, grid)
        ax.plot(grid, m, color=c, ls=ls, lw=lw, zorder=z, alpha=1.0 if emph else 0.8,
                label=LABEL[name])
        ax.fill_between(grid, m - s, m + s, color=c, alpha=0.13, lw=0, zorder=z - 1)
        st = np.array([r["arms"][name]["steps"] for r in runs])
        ac = np.array([r["arms"][name]["acc"] for r in runs]) * 100
        ax.errorbar(st.mean(), ac.mean(), xerr=st.std(), yerr=ac.std(), color=c,
                    marker="*" if emph else "s", ms=13 if emph else 5.5,
                    mec="white", mew=0.8, capsize=2, lw=1.0, zorder=z + 4)

    # direct labels on the two winners' operating points — in the clear band above
    # the ceiling, text in ink (identity comes from the star directly below)
    for name, tx in (("conv", 3.95), ("dstate", 6.05)):   # right-aligned, no overlap
        st = np.mean([r["arms"][name]["steps"] for r in runs])
        ac = np.mean([r["arms"][name]["acc"] for r in runs]) * 100
        ax.annotate(f"{name}  {ac:.1f}% @ {st:.2f}", (st, ac), (tx, 73.4),
                    fontsize=8, color="0.15", fontweight="bold",
                    ha="right", va="bottom")

    ax.set_xlabel("average latent steps used (of %d)" % T, fontsize=10)
    ax.set_ylabel("accuracy (%)", fontsize=10)
    ax.set_xlim(0.9, T + 0.1)
    ax.set_ylim(top=76)
    ax.grid(alpha=0.22, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9,
              title=f"tau-sweep, {len(runs)} seeds, mean±std", title_fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = f"{args.out}/fig1_bakeoff_band.{ext}"
        fig.savefig(p, dpi=300)
        print("saved ->", p)


if __name__ == "__main__":
    main()
