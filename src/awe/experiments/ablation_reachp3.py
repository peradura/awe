"""Halting-signal bake-off on the joint task (partial-obs reachability).

The review (docs/REVIEW.md) established that the remaining bottleneck is the
halting signal itself: with the base learner fixed (curriculum + aux), turning
entropy-halting ON still costs accuracy vs `persist`. This experiment races
halting signals on identical trajectories, with held-out tau and baselines.

Arms (halt when signal < tau; per-arm tau calibrated on a HELD-OUT batch):
  entropy : readout entropy (status quo)
  conv    : sym-KL between consecutive readout distributions (Part-1's
            convergence signal, online form; cannot halt at t=0)
  dstate  : ||s_t - s_{t-1}||^2 / d (state convergence; the "latent step as
            gradient descent on memory loss" reading: Δs ∝ ∇L)
  recon   : memory-prediction mismatch ||norm(s_{t+1}) - norm(r_t)||^2 —
            the thesis-faithful miss detector (r_t = memory read = predicted
            next latent). NOTE: low = transition known, which can be low
            mid-path — whether that halts prematurely is part of the question.

Baselines: fixed-depth frontier (acc at every fixed depth) and random halting.
Failure decomposition at the operating tau (chosen on calib): early-wrong /
early-right / budget (never crossed tau).

Per-seed JSON is written to results/; run with --aggregate to combine seeds.

GPU NOTE: trains a model per seed. Ask before launching on a shared GPU.
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

from awe.datasets.reachp import make_batch
from awe.models.memory import RuleReasoner
from awe.experiments.ablation_reachp2 import Cfg, train, slice_q

SIGNALS = ["entropy", "conv", "dstate", "recon"]


# --------------------------- tracing ----------------------------------------
@torch.no_grad()
def trace_stream(model, cfg, data):
    """Persist-memory stream; records per-step preds + all signals.

    Halting is applied post-hoc from these traces — valid because halting only
    selects which step's readout is used; memory writes happen before retrieval
    and are unaffected by the halt step.
    """
    device = data["probe"].device
    T = cfg.T
    B = data["probe"].shape[0]
    delta = model.new_memory(B, device)
    cols = {k: [] for k in ["preds"] + SIGNALS}
    tgt, ans, K = [], [], []
    for q in range(cfg.Q):
        bq = slice_q(data, q)
        delta = model.write(delta, bq["keys"], bq["vals"])   # write, then retrieve
        W = model.W_base.unsqueeze(0) + delta
        s = model.node(bq["probe"])
        logp_prev = None
        per = {k: [] for k in cols}
        for _ in range(T):
            r = torch.bmm(W, model.mem_ln(s).unsqueeze(-1)).squeeze(-1)
            s_new = s + model.step_mlp(torch.cat([s, r], dim=-1))
            logit = model.coda(model.coda_ln(s_new))
            logp = F.log_softmax(logit, dim=-1)
            p = logp.exp()
            per["preds"].append(logit.argmax(-1))
            per["entropy"].append(-(p * logp).sum(-1))
            if logp_prev is None:
                per["conv"].append(torch.full_like(per["entropy"][-1], float("inf")))
            else:
                p_prev = logp_prev.exp()
                symkl = (p * (logp - logp_prev)).sum(-1) + \
                        (p_prev * (logp_prev - logp)).sum(-1)
                per["conv"].append(symkl)
            per["dstate"].append(((s_new - s) ** 2).mean(-1))
            per["recon"].append(((F.normalize(s_new, dim=-1) -
                                  F.normalize(r, dim=-1)) ** 2).sum(-1))
            logp_prev = logp
            s = s_new
        for k in cols:
            cols[k].append(torch.stack(per[k], 1))          # (B,T)
        tgt.append(bq["target"]); ans.append(bq["ans"]); K.append(bq["K"])
    res = {k: torch.cat(v, 0) for k, v in cols.items()}      # (B*Q, T)
    res["tgt"] = torch.cat(tgt); res["ans"] = torch.cat(ans); res["K"] = torch.cat(K)
    return res


# --------------------------- halting eval ------------------------------------
def apply_halt(preds, sig, tgt, tau):
    T = preds.shape[1]
    below = sig < tau
    has = below.any(1)
    first = below.int().argmax(1)
    hs = torch.where(has, first, torch.full_like(first, T - 1))
    pred = preds[torch.arange(len(hs), device=hs.device), hs]
    acc = (pred == tgt).float().mean().item()
    steps = (hs + 1).float().mean().item()
    return acc, steps, hs


def tau_grid(sig_cal, n=15):
    finite = sig_cal[torch.isfinite(sig_cal)]
    qs = torch.linspace(0.02, 0.98, n, device=finite.device)
    return torch.quantile(finite.float(), qs).tolist()


def decompose(preds, hs, tgt, T):
    """early-wrong / early-right / budget-right / budget-wrong fractions."""
    idx = torch.arange(len(hs), device=hs.device)
    correct = preds[idx, hs] == tgt
    early = hs < (T - 1)
    n = len(hs)
    return {
        "early_wrong": ((early & ~correct).sum().item()) / n,
        "early_right": ((early & correct).sum().item()) / n,
        "budget_right": ((~early & correct).sum().item()) / n,
        "budget_wrong": ((~early & ~correct).sum().item()) / n,
    }


# --------------------------- per-seed run ------------------------------------
def run_seed(args, device):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    cfg = Cfg()
    model = RuleReasoner(cfg.n, d=args.d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"seed={args.seed} device={device} params="
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    print("== train (curriculum + aux, persist stream) ==")
    train(model, cfg, opt, device, args.steps, args.batch, rng, args.lam_aux)

    cfg.kcap = 4
    data_cal = make_batch(args.ncal, cfg, rng, device)    # held-out calibration
    data_ev = make_batch(args.neval, cfg, rng, device)    # evaluation
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

    print(f"== persist ceiling (fixed depth T={T}): acc {persist_acc*100:.1f}% ==")
    for name in SIGNALS:
        curve = []
        best = (-1.0, None)                                # (calib acc, tau)
        for tau in tau_grid(tr_cal[name]):
            acc_c, _, _ = apply_halt(tr_cal["preds"], tr_cal[name], tr_cal["tgt"], tau)
            acc_e, st_e, _ = apply_halt(tr_ev["preds"], tr_ev[name], tr_ev["tgt"], tau)
            curve.append((st_e, acc_e))
            if acc_c > best[0]:
                best = (acc_c, tau)
        tau_star = best[1]
        acc, steps, hs = apply_halt(tr_ev["preds"], tr_ev[name], tr_ev["tgt"], tau_star)
        dec = decompose(tr_ev["preds"], hs, tr_ev["tgt"], T)
        # signal-quality corr: signal@step0 vs answerable (conv has no step0)
        t0 = 1 if name == "conv" else 0
        sig0 = tr_ev[name][:, t0]
        m = torch.isfinite(sig0)
        corr = torch.corrcoef(torch.stack([tr_ev["ans"][m].float(), sig0[m]]))[0, 1].item()
        out["arms"][name] = {
            "tau": tau_star, "acc": acc, "steps": steps,
            "gap_vs_persist_pp": (acc - persist_acc) * 100,
            "corr_ans": corr, "decomposition": dec, "curve": curve,
        }
        print(f"  {name:8s} | acc {acc*100:5.1f}% ({(acc-persist_acc)*100:+.1f}pp vs persist)"
              f" | steps {steps:4.2f} | corr(ans) {corr:+.3f}"
              f" | early-wrong {dec['early_wrong']*100:.0f}%")

    # random-halting baseline curve
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    rand_curve = []
    N = len(tr_ev["tgt"])
    for m_target in range(1, T + 1):
        hs = torch.randint(0, min(2 * m_target - 1, T), (N,), generator=g).to(device)
        pred = tr_ev["preds"][torch.arange(N, device=device), hs]
        rand_curve.append(((hs + 1).float().mean().item(),
                           (pred == tr_ev["tgt"]).float().mean().item()))
    out["random_curve"] = rand_curve

    path = f"{args.out}/reachp3_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved -> {path}")
    return out


# --------------------------- aggregate ---------------------------------------
def aggregate(args):
    files = sorted(glob.glob(f"{args.out}/reachp3_seed*.json"))
    runs = [json.load(open(f)) for f in files]
    if not runs:
        print("no per-seed JSONs found"); return
    print(f"== aggregate over {len(runs)} seeds: {[r['seed'] for r in runs]} ==")
    pa = torch.tensor([r["persist_acc"] for r in runs])
    print(f"persist ceiling: {pa.mean()*100:.1f}% ± {pa.std(unbiased=False)*100:.1f}")
    rows = []
    for name in SIGNALS:
        acc = torch.tensor([r["arms"][name]["acc"] for r in runs])
        st = torch.tensor([r["arms"][name]["steps"] for r in runs])
        gap = torch.tensor([r["arms"][name]["gap_vs_persist_pp"] for r in runs])
        ew = torch.tensor([r["arms"][name]["decomposition"]["early_wrong"] for r in runs])
        rows.append((name, acc.mean().item(), acc.std(unbiased=False).item(),
                     st.mean().item(), gap.mean().item(), ew.mean().item()))
        print(f"  {name:8s} | acc {acc.mean()*100:5.1f}±{acc.std(unbiased=False)*100:.1f}%"
              f" | steps {st.mean():4.2f} | gap {gap.mean():+5.1f}pp"
              f" | early-wrong {ew.mean()*100:.0f}%")
    winner = max(rows, key=lambda r: r[1])
    print(f"== verdict: best arm = {winner[0]} "
          f"(gap {winner[4]:+.1f}pp vs persist at {winner[3]:.2f} steps) ==")

    plt.figure(figsize=(6.4, 4.6))
    r0 = runs[0]
    fx = [(s, a) for a, s in r0["fixed_frontier"]]
    plt.plot([s for s, _ in fx], [a * 100 for _, a in fx], "k--", label="fixed depth", alpha=.6)
    plt.plot([s for s, _ in r0["random_curve"]], [a * 100 for _, a in r0["random_curve"]],
             ":", color="gray", label="random halt", alpha=.7)
    for name in SIGNALS:
        pts = sorted(r0["arms"][name]["curve"])
        plt.plot([s for s, _ in pts], [a * 100 for _, a in pts], marker="o", ms=3, label=name)
    plt.axhline(r0["persist_acc"] * 100, color="green", lw=1, alpha=.6, label="persist ceiling")
    plt.xlabel("avg steps used"); plt.ylabel("accuracy (%)")
    plt.title(f"Halting-signal bake-off (seed {r0['seed']}; aggregate table in log)")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    fig = f"{args.out}/reachp3_bakeoff.png"
    plt.savefig(fig, dpi=130)
    print(f"saved figure -> {fig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
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
        aggregate(args)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_seed(args, device)


if __name__ == "__main__":
    main()
