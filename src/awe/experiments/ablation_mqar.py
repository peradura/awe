"""External-task transfer: convergence-halting on MQAR.

Runs the SAME halting-signal bake-off as `ablation_reachp3` (entropy / conv /
dstate / recon, held-out tau, failure decomposition, fixed-depth + random-halt
baselines) on MQAR instead of partial-obs reachability — the externally legible
transfer test in `docs/mqar_design.md`. Reuses the reachp3 harness verbatim; only
the dataset and (single-hop, no curriculum) training loop differ.

Primary question: does a convergence halter (conv/dstate) match fixed-depth
accuracy at lower avg compute and beat entropy/recon, on a task with published
baselines? Secondary probe (unification): does the halt signal predict the
delta-rule write magnitude? (recorded as `corr_sig_write`.)

GPU NOTE: trains a model per seed; ask before launching on a shared GPU.
"""
import argparse
import glob
import json
import random

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from awe.datasets.mqar import make_batch, MqarConfig
from awe.models.memory import RuleReasoner
from awe.experiments.ablation_reachp2 import slice_q, aux_readback
from awe.experiments.ablation_reachp3 import (
    trace_stream, apply_halt, tau_grid, decompose, SIGNALS,
)


class Cfg(MqarConfig):
    pass


def train(model, cfg, opt, device, steps, batch, rng, lam_aux, log_every=400):
    """Single-hop MQAR: write revealed pairs, retrieve each probe, converge-and-hold
    supervision (traj[:, t] = value for all t) + aux read-back on revealed keys."""
    model.train()
    for it in range(1, steps + 1):
        data = make_batch(batch, cfg, rng, device)
        delta = model.new_memory(batch, device)
        loss = 0.0
        for q in range(cfg.Q):
            bq = slice_q(data, q)
            logits_t, _, delta = model.query(delta, bq, cfg.T, write=True)
            chain = sum(F.cross_entropy(logits_t[t], bq["traj"][:, t])
                        for t in range(cfg.T)) / cfg.T
            aux = aux_readback(model, delta, bq["keys"], bq["vals"])
            loss = loss + chain + lam_aux * aux
        loss = loss / cfg.Q
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % log_every == 0 or it == 1:
            print(f"  step {it:5d} | loss {loss.item():.4f}")


def run_seed(args, device):
    torch.manual_seed(args.seed); rng = random.Random(args.seed)
    cfg = Cfg()
    model = RuleReasoner(cfg.n, d=args.d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"seed={args.seed} device={device} | params="
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M | "
          f"n={cfg.n} m={cfg.m} Q={cfg.Q} T={cfg.T}")

    print("== train (single-hop MQAR, persist stream) ==")
    train(model, cfg, opt, device, args.steps, args.batch, rng, args.lam_aux)

    data_cal = make_batch(args.ncal, cfg, rng, device)      # held-out calibration
    data_ev = make_batch(args.neval, cfg, rng, device)      # evaluation
    tr_cal = trace_stream(model, cfg, data_cal)
    tr_ev = trace_stream(model, cfg, data_ev)
    T = cfg.T

    persist_acc = (tr_ev["preds"][:, -1] == tr_ev["tgt"]).float().mean().item()
    fixed_frontier = [
        ((tr_ev["preds"][:, d - 1] == tr_ev["tgt"]).float().mean().item(), float(d))
        for d in range(1, T + 1)
    ]
    out = {"seed": args.seed, "persist_acc": persist_acc,
           "fixed_frontier": fixed_frontier, "arms": {}}
    print(f"== persist ceiling (fixed depth T={T}): {persist_acc*100:.1f}% ==")

    for name in SIGNALS:
        best = (-1.0, None)
        curve = []
        for tau in tau_grid(tr_cal[name]):
            acc_c, _, _ = apply_halt(tr_cal["preds"], tr_cal[name], tr_cal["tgt"], tau)
            acc_e, st_e, _ = apply_halt(tr_ev["preds"], tr_ev[name], tr_ev["tgt"], tau)
            curve.append((st_e, acc_e))
            if acc_c > best[0]:
                best = (acc_c, tau)
        tau_star = best[1]
        acc, steps, hs = apply_halt(tr_ev["preds"], tr_ev[name], tr_ev["tgt"], tau_star)
        dec = decompose(tr_ev["preds"], hs, tr_ev["tgt"], T)
        t0 = 1 if name == "conv" else 0
        sig0 = tr_ev[name][:, t0]
        m = torch.isfinite(sig0)
        corr = torch.corrcoef(torch.stack([tr_ev["ans"][m].float(), sig0[m]]))[0, 1].item()
        out["arms"][name] = {
            "tau": tau_star, "acc": acc, "steps": steps,
            "gap_vs_persist_pp": (acc - persist_acc) * 100,
            "corr_ans": corr, "decomposition": dec, "curve": curve,
        }
        print(f"  {name:8s} | acc {acc*100:5.1f}% ({(acc-persist_acc)*100:+.1f}pp)"
              f" | steps {steps:4.2f} | early-wrong {dec['early_wrong']*100:.0f}%")

    # depth vs load (K = # keys in memory): does the halter grade compute with load?
    hs_dstate = apply_halt(tr_ev["preds"], tr_ev["dstate"], tr_ev["tgt"],
                           out["arms"]["dstate"]["tau"])[2] + 1
    K = tr_ev["K"]; dvl = {}
    for k in sorted(set(int(x) for x in K.tolist())):
        mk = (K == k)
        if mk.sum() >= 20:
            dvl[str(k)] = hs_dstate[mk].float().mean().item()
    out["depth_vs_load"] = dvl

    path = f"{args.out}/mqar_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved -> {path}")
    return out


def aggregate(args):
    files = sorted(glob.glob(f"{args.out}/mqar_seed*.json"))
    runs = [json.load(open(f)) for f in files]
    if not runs:
        print("no per-seed JSONs found"); return
    print(f"== aggregate over {len(runs)} seeds: {[r['seed'] for r in runs]} ==")
    pa = torch.tensor([r["persist_acc"] for r in runs])
    print(f"persist ceiling: {pa.mean()*100:.1f}% ± {pa.std(unbiased=False)*100:.1f}")
    for name in SIGNALS:
        acc = torch.tensor([r["arms"][name]["acc"] for r in runs])
        st = torch.tensor([r["arms"][name]["steps"] for r in runs])
        gap = torch.tensor([r["arms"][name]["gap_vs_persist_pp"] for r in runs])
        ew = torch.tensor([r["arms"][name]["decomposition"]["early_wrong"] for r in runs])
        print(f"  {name:8s} | acc {acc.mean()*100:5.1f}±{acc.std(unbiased=False)*100:.1f}%"
              f" | steps {st.mean():4.2f} | gap {gap.mean():+5.1f}pp"
              f" | early-wrong {ew.mean()*100:.0f}%")

    r0 = runs[0]
    plt.figure(figsize=(6.4, 4.6))
    fx = [(s, a) for a, s in r0["fixed_frontier"]]
    plt.plot([s for s, _ in fx], [a * 100 for _, a in fx], "k--", label="fixed depth", alpha=.6)
    for name in SIGNALS:
        pts = sorted(r0["arms"][name]["curve"])
        plt.plot([s for s, _ in pts], [a * 100 for _, a in pts], marker="o", ms=3, label=name)
    plt.axhline(r0["persist_acc"] * 100, color="green", lw=1, alpha=.6, label="persist ceiling")
    plt.xlabel("avg steps used"); plt.ylabel("MQAR accuracy (%)")
    plt.title(f"MQAR halting bake-off (seed {r0['seed']}; aggregate in log)")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    fig = f"{args.out}/mqar_bakeoff.png"
    plt.savefig(fig, dpi=130); print(f"saved figure -> {fig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam_aux", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ncal", type=int, default=500)
    ap.add_argument("--neval", type=int, default=500)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args); return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_seed(args, device)


if __name__ == "__main__":
    main()
