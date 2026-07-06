"""Halting-signal bake-off — the decisive comparison (PROJECT.md §7).

One trained model per task; a single no-halt rollout per query yields traces
from which every halting rule is applied post-hoc (exact for RuleReasoner:
retrieval never mutates the memory). Signals compared (low = done -> halt):

  ent     readout entropy (the current method)
  recon   memory-read decodability: min_v ||r_t - node(v)||^2.
          NOTE: a self-supervised *proxy* for the TTT write loss — the literal
          write loss ||W k - v||^2 needs the true value v, which for the probe
          IS the answer, so it cannot be a halting signal; decodability is the
          closest label-free per-step analogue. Read verdicts accordingly.
  rnorm   negative read energy: -||r_t||^2 (memory miss -> weak read)
  dstate  latent convergence: ||s_t - s_{t-1}||^2 (Geiping-style)
  dent    surprise plateau: |ent_t - ent_{t-1}| (Delta-surprise arm; cannot
          halt at t=0 by construction -> structural +1-step floor vs others)

Controls:
  shuf_g  same halt-step marginal, permuted across ALL examples (weak null)
  shuf_b  permuted WITHIN each query-index block — preserves any per-position
          schedule, so beating it requires true per-example information
  fixed-t fixed depth t = 1..T (compute-matched frontier)

tau per signal is chosen on a HELD-OUT calibration stream (fewest avg steps
within --slack pp of no-halt calib accuracy; fallback max calib acc), and the
FULL tau-sweep (acc, steps) curve per signal is also evaluated on eval — the
matched-compute comparison required by the PROJECT.md §7 kill criterion reads
off these curves, not off the single operating points.

Also reported per signal: corr(answerable, signal) at its first informative
step, and a failure decomposition summing to 1 — early_right / premature
(wrong at halt, right at full depth) / wrong_anyway / full_depth — plus
premature as a fraction of early halts.

Usage (pass --ckpt saved by the ablation run to skip re-training):
  PYTHONPATH=src python -m awe.experiments.bakeoff --task rule
  PYTHONPATH=src python -m awe.experiments.bakeoff --task reachp --ckpt results/ckpt_reachp_s0.pt
"""
import argparse
import os
import random

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from awe.models.memory import RuleReasoner
from awe.datasets import rule as rule_ds
from awe.datasets import reachp as reachp_ds
from awe.experiments.ablation_rule import train as train_rule
from awe.experiments.ablation_reachp import train as train_reachp

SIGNALS = ["ent", "recon", "rnorm", "dstate", "dent"]


def slice_q(data, q):
    return {k: v[:, q] for k, v in data.items()}


@torch.no_grad()
def stream_traces(model, cfg, data, persist=True):
    """Mirror RuleReasoner.query/retrieve but record every candidate signal.
    Returns per-example (N = B*Q, T) traces. Post-hoc halting is exact here
    because retrieval never mutates the memory."""
    model.eval()
    B = data["probe"].shape[0]
    T = cfg.T
    dev = data["probe"].device
    E = model.node.weight                          # (n, d)
    e2 = (E * E).sum(-1)                           # (n,)
    delta = model.new_memory(B, dev)
    tr = {k: [] for k in ("logits", "ent", "recon", "rnorm", "dstate")}
    tgt_all, ans_all = [], []
    for q in range(cfg.Q):
        bq = slice_q(data, q)
        d_in = delta if persist else model.new_memory(B, dev)
        d_in = model.write(d_in, bq["keys"], bq["vals"])
        if persist:
            delta = d_in
        W = model.W_base.unsqueeze(0) + d_in
        s = model.node(bq["probe"])
        row = {k: [] for k in tr}
        for _ in range(T):
            r = torch.bmm(W, model.mem_ln(s).unsqueeze(-1)).squeeze(-1)
            s_new = s + model.step_mlp(torch.cat([s, r], dim=-1))
            logit = model.coda(model.coda_ln(s_new))
            logp = F.log_softmax(logit, dim=-1)
            row["logits"].append(logit)
            row["ent"].append(-(logp.exp() * logp).sum(-1))
            row["recon"].append(
                ((r * r).sum(-1, keepdim=True) - 2.0 * (r @ E.t()) + e2).min(-1).values)
            row["rnorm"].append(-(r * r).sum(-1))
            row["dstate"].append(((s_new - s) ** 2).sum(-1))
            s = s_new
        for k in tr:
            tr[k].append(torch.stack(row[k], 1))
        tgt_all.append(bq["target"])
        ans_all.append(bq["ans"])
    out = {k: torch.cat(v, 0) for k, v in tr.items()}
    out["preds"] = out.pop("logits").argmax(-1)     # (N, T)
    # Delta-surprise: undefined at t=0 -> +inf so it can never halt there
    out["dent"] = torch.cat(
        [torch.full_like(out["ent"][:, :1], 1e9),
         (out["ent"][:, 1:] - out["ent"][:, :-1]).abs()], 1)
    out["target"] = torch.cat(tgt_all)
    out["ans"] = torch.cat(ans_all).float()
    out["qidx"] = torch.arange(cfg.Q, device=dev).repeat_interleave(B)
    return out


def apply_halt(tr, sig, tau):
    """First step with signal < tau, else last. Returns acc, avg steps, hs, has."""
    s = tr[sig]
    T = s.shape[1]
    below = s < tau
    has = below.any(1)
    first = below.int().argmax(1)
    hs = torch.where(has, first, torch.full_like(first, T - 1))
    pred = tr["preds"][torch.arange(hs.shape[0]), hs]
    acc = (pred == tr["target"]).float().mean().item()
    return acc, (hs + 1).float().mean().item(), hs, has


def choose_tau(tr_calib, sig, slack_pp):
    """Fewest avg steps within slack of the no-halt calib accuracy.
    Returns (tau, within_slack, all_candidate_taus)."""
    nohalt = (tr_calib["preds"][:, -1] == tr_calib["target"]).float().mean().item()
    vals = tr_calib[sig].flatten().float()
    vals = vals[vals < 1e8]                        # drop the t=0 sentinel (dent)
    cands = torch.quantile(vals, torch.linspace(0.02, 0.98, 25)).tolist()
    best = None
    for tau in cands:
        acc, steps, _, _ = apply_halt(tr_calib, sig, tau)
        if acc >= nohalt - slack_pp / 100.0 and (best is None or steps < best[0]):
            best = (steps, tau)
    if best is not None:
        return best[1], True, cands
    for tau in cands:                              # fallback: max calib accuracy
        acc, steps, _, _ = apply_halt(tr_calib, sig, tau)
        if best is None or acc > best[0]:
            best = (acc, tau)
    return best[1], False, cands


def decompose(tr, hs, has):
    """Fractions over all N (sum to 1): early_right / premature / wrong_anyway /
    full_depth (no halt strictly before the last step). Plus premature as a
    fraction of early halts."""
    N, T = tr["preds"].shape
    idx = torch.arange(N, device=hs.device)
    ch = tr["preds"][idx, hs] == tr["target"]
    cf = tr["preds"][:, -1] == tr["target"]
    early = has & (hs < T - 1)
    f = lambda m: m.float().mean().item()
    er, pm, wa = f(early & ch), f(early & ~ch & cf), f(early & ~ch & ~cf)
    n_early = early.float().sum().item()
    return dict(early_right=er, premature=pm, wrong_anyway=wa,
                full_depth=1.0 - er - pm - wa,
                prem_per_halt=(early & ~ch & cf).float().sum().item() / max(n_early, 1.0))


def shuffled_acc(tr, hs, g, within_block, reps=5):
    """Null control: permute halt steps across examples (within_block=False)
    or within each query-index block (True — preserves any per-position
    schedule). Averaged over `reps` permutations."""
    N = hs.shape[0]
    idx = torch.arange(N, device=hs.device)
    accs = []
    for _ in range(reps):
        if within_block:
            perm = idx.clone()
            for q in tr["qidx"].unique():
                blk = (tr["qidx"] == q).nonzero(as_tuple=True)[0]
                perm[blk] = blk[torch.randperm(blk.shape[0], generator=g).to(blk.device)]
        else:
            perm = torch.randperm(N, generator=g).to(hs.device)
        pred = tr["preds"][idx, hs[perm]]
        accs.append((pred == tr["target"]).float().mean().item())
    return sum(accs) / len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rule", "reachp"], required=True)
    ap.add_argument("--steps", type=int, default=0,
                    help="0 = task default (rule 3000 / reachp 4000, matching the ablations)")
    ap.add_argument("--ckpt", type=str, default="",
                    help="load model state if the file exists (skip training), else train and save here")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--slack", type=float, default=1.0, help="acc slack (pp) for tau pick")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.task == "rule":
        cfg, make_batch, train = rule_ds.RuleConfig(), rule_ds.make_batch, train_rule
    else:
        cfg, make_batch, train = reachp_ds.ReachPConfig(), reachp_ds.make_batch, train_reachp
    if args.steps <= 0:
        args.steps = 3000 if args.task == "rule" else 4000
    model = RuleReasoner(cfg.n, d=args.d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    npar = sum(p.numel() for p in model.parameters())
    print(f"task={args.task} | device={device} | params={npar/1e6:.2f}M | seed={args.seed}")

    if args.ckpt and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"== loaded checkpoint {args.ckpt} (skip training; eval batches are fresh draws) ==")
    else:
        print("== train ==")
        train(model, cfg, opt, device, args.steps, args.batch, rng)
        if args.ckpt:
            torch.save(model.state_dict(), args.ckpt)
            print(f"saved ckpt -> {args.ckpt}")

    data = make_batch(1000, cfg, rng, device)
    calib = make_batch(256, cfg, rng, device)      # held-out: tau never sees `data`
    tr = stream_traces(model, cfg, data, persist=True)
    tc = stream_traces(model, cfg, calib, persist=True)
    N, T = tr["preds"].shape
    nohalt_acc = (tr["preds"][:, -1] == tr["target"]).float().mean().item()
    print(f"== eval (N={N}, T={T}) | no-halt acc {nohalt_acc*100:.1f}% @ {T} steps ==")

    print("== fixed-depth frontier (compute-matched control) ==")
    for t in range(T):
        a = (tr["preds"][:, t] == tr["target"]).float().mean().item()
        print(f"  fixed t={t+1:2d} | acc {a*100:5.1f}%")

    g = torch.Generator().manual_seed(args.seed)
    print("== signal arms (tau on held-out calib; shuf_g/shuf_b = global/within-block nulls) ==")
    print(f"  {'signal':7s} {'tau':>9s} {'acc':>6s} {'steps':>6s} {'shuf_g':>7s} {'shuf_b':>7s} "
          f"{'corr':>7s}  er/pm/wa/fd  pm|halt  tau_ok")
    results = {}
    for sig in SIGNALS:
        tau, ok, cands = choose_tau(tc, sig, args.slack)
        acc, steps, hs, has = apply_halt(tr, sig, tau)
        sg = shuffled_acc(tr, hs, g, within_block=False)
        sb = shuffled_acc(tr, hs, g, within_block=True)
        col = 1 if sig == "dent" else 0            # dent@0 is the sentinel
        c = torch.corrcoef(torch.stack([tr["ans"], tr[sig][:, col].float()]))[0, 1].item()
        d = decompose(tr, hs, has)
        curve = [apply_halt(tr, sig, t)[:2] for t in cands]   # full tau-sweep on eval
        results[sig] = (acc, steps, curve)
        print(f"  {sig:7s} {tau:9.4f} {acc*100:5.1f}% {steps:6.2f} {sg*100:6.1f}% {sb*100:6.1f}% "
              f"{c:+7.3f}  {d['early_right']:.2f}/{d['premature']:.2f}/"
              f"{d['wrong_anyway']:.2f}/{d['full_depth']:.2f}  {d['prem_per_halt']:7.2f}  "
              f"{'y' if ok else 'FALLBACK'}")
    print("== tau-sweep on eval per signal (steps:acc%) — read matched-compute comparisons "
          "(PROJECT.md §7 kill criterion) off these curves, not the single points ==")
    for sig, (_, _, curve) in results.items():
        print(f"  curve {sig:7s} " + " ".join(f"{s:.2f}:{a*100:.1f}" for a, s in curve))

    if args.fig:
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        fx = list(range(1, T + 1))
        fy = [(tr["preds"][:, t] == tr["target"]).float().mean().item() * 100 for t in range(T)]
        ax.plot(fx, fy, "--", color="gray", label="fixed-depth frontier")
        for sig, (acc, steps, curve) in results.items():
            pts = sorted((s, a) for a, s in curve)
            ax.plot([s for s, _ in pts], [a * 100 for _, a in pts],
                    marker=".", ms=4, alpha=0.8, label=sig)
            ax.scatter([steps], [acc * 100], zorder=3)
        ax.set(xlabel="avg steps", ylabel="accuracy (%)",
               title=f"Halting-signal bake-off ({args.task}, seed {args.seed})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.fig, dpi=130)
        print(f"saved figure -> {args.fig}")


if __name__ == "__main__":
    main()
