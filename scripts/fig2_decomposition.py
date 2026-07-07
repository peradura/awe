"""Fig 2 (paper): per-example failure decomposition, canonical bake-off (Part 4c).

One stacked bar per signal (10-seed means; segments sum to 1): early-right
(halted early, correct — the compute saving), premature (halted early, wrong,
full depth would be right — the failure mode), wrong-anyway (halted early,
wrong at any depth), full-depth (did not halt before the budget).

Design: semantic segment colors (positive blue / serious red / neutral grays),
distinct lightness ordering so the story survives grayscale; 2px white gaps
between segments; direct % labels on segments >= 3%.

Usage: PYTHONPATH=src python scripts/fig2_decomposition.py [--out results]
"""
import argparse
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["conv", "dstate", "dent", "rnorm", "recon", "ent"]  # Table-1 order
LABEL = {"conv": "conv", "dstate": "dstate", "dent": "dent",
         "rnorm": "rnorm", "recon": "recon\n(decod.)", "ent": "ent"}
SEGS = [  # (key, legend label, color)
    ("early_right", "early-right (compute saved)", "#0072B2"),
    ("premature",   "premature (foregone accuracy)", "#D55E00"),
    ("wrong_anyway", "wrong anyway", "#B8B8B8"),
    ("full_depth",  "full depth (no halt)", "#E8E4DC"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    runs = [json.load(open(f))
            for f in sorted(glob.glob(f"{args.out}/bakeoff_reachp2_s*.json"))]
    assert runs, "no bakeoff_reachp2_s*.json"

    means = {s: {k: float(np.mean([r["arms"][s]["decomposition"][k] for r in runs]))
                 for k, *_ in SEGS} for s in ORDER}

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    x = np.arange(len(ORDER))
    bottom = np.zeros(len(ORDER))
    for key, lab, color in SEGS:
        vals = np.array([means[s][key] for s in ORDER]) * 100
        ax.bar(x, vals, 0.62, bottom=bottom, color=color, label=lab,
               edgecolor="white", linewidth=1.2)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 3.0:  # direct label, ink not series color
                dark = key in ("early_right", "premature")
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white" if dark else "0.25")
        bottom += vals

    ax.set_xticks(x, [LABEL[s] for s in ORDER], fontsize=9)
    ax.set_ylabel("% of examples", fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.22, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncols=2, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = f"{args.out}/fig2_decomposition.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("saved ->", p)


if __name__ == "__main__":
    main()
