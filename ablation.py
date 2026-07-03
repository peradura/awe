"""Increment 2: train the surprise-unified reasoner, then run the 4-way
ablation and draw the accuracy-vs-compute frontier.

Configs (single trained model; eval-time toggles, fair ablation):
  1. fixed     : no halting, no TTT        -- Coconut-style fixed depth
  2. +halt     : surprise halting, no TTT  -- depth knob only
  3. +ttt      : fixed depth, TTT on       -- weight knob only
  4. both      : surprise halting + TTT    -- ONE signal drives both (ours)

Reported: for each config a set of (avg compute steps, accuracy) points, the
H1 check (does TTT let us halt earlier at matched accuracy?) and a saved
figure accuracy_vs_steps.png (the H2 frontier).

GPU NOTE: this script trains on GPU. Do not launch without the user's OK.
"""
import argparse
import random

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import GraphConfig, vocab_size, v0_pos, make_batch
from model_ttt import SurpriseReasoner


# ----------------------------- training --------------------------------------
def train(model, cfg, opt, device, steps, batch, rng, lr_inner, lam, log_every=300):
    model.train()
    T = cfg.T
    for it in range(1, steps + 1):
        tok, pad, traj, _ = make_batch(batch, cfg, rng, device)
        H, s0, pad = model.embed(tok, pad)
        logits_t, sur_t = model.rollout(H, s0, pad, T, ttt=True, lr_inner=lr_inner)
        ce = sum(F.cross_entropy(logits_t[t], traj[:, t]) for t in range(T)) / T
        aux = torch.stack(sur_t).mean()               # calibrate reconstruction
        loss = ce + lam * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % log_every == 0 or it == 1:
            print(f"  step {it:5d} | loss {loss.item():.4f} | ce {ce.item():.4f} | sur {aux.item():.4f}")


# ----------------------------- evaluation ------------------------------------
@torch.no_grad()
def rollout_traces(model, H, s0, pad, steps, ttt, lr_inner):
    logits_t, sur_t = model.rollout(H, s0, pad, steps, ttt=ttt, lr_inner=lr_inner)
    preds = torch.stack([l.argmax(-1) for l in logits_t], dim=1)   # (B,steps)
    sur = torch.stack(sur_t, dim=1)                                # (B,steps)
    return preds, sur


def halt_from_trace(preds, sur, tgt, tau):
    """Halt each example at first step with surprise < tau (else last step)."""
    B, T = preds.shape
    below = sur < tau                                  # (B,T)
    has = below.any(dim=1)
    first = torch.argmax(below.int(), dim=1)           # first True (0 if none)
    halt = torch.where(has, first, torch.full_like(first, T - 1))
    steps_used = (halt + 1).float()
    pred = preds[torch.arange(B, device=preds.device), halt]
    acc = (pred == tgt).float().mean().item()
    return acc, steps_used.mean().item()


@torch.no_grad()
def ablate(model, cfg, device, rng, lr_inner, n=6000):
    model.eval()
    tok, pad, traj, K = make_batch(n, cfg, rng, device)
    H, s0, pad = model.embed(tok, pad)
    tgt = traj[:, -1]
    Tev = cfg.T + 4

    # precompute traces once per ttt-mode
    pr_off, su_off = rollout_traces(model, H, s0, pad, Tev, ttt=False, lr_inner=lr_inner)
    pr_on, su_on = rollout_traces(model, H, s0, pad, Tev, ttt=True, lr_inner=lr_inner)

    def fixed_points(preds):
        pts = []
        for r in range(1, Tev + 1):
            acc = (preds[:, r - 1] == tgt).float().mean().item()
            pts.append((float(r), acc))
        return pts

    def halt_points(preds, sur):
        qs = torch.quantile(sur.flatten(),
                            torch.linspace(0.02, 0.9, 12, device=device))
        pts = []
        for tau in qs.tolist():
            acc, steps = halt_from_trace(preds, sur, tgt, tau)
            pts.append((steps, acc))
        return sorted(pts)

    res = {
        "1.fixed": fixed_points(pr_off),
        "2.+halt": halt_points(pr_off, su_off),
        "3.+ttt": fixed_points(pr_on),
        "4.both": halt_points(pr_on, su_on),
    }
    return res, K


def matched_steps(points, target_acc):
    """Min compute (steps) to reach >= target_acc on a config's frontier."""
    ok = [s for s, a in points if a >= target_acc]
    return min(ok) if ok else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_inner", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", type=str, default="accuracy_vs_steps.png")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GraphConfig()

    model = SurpriseReasoner(vocab_size(cfg.n), cfg.n, v0_pos(cfg.n), d=args.d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    npar = sum(p.numel() for p in model.parameters())
    print(f"device={device} | params={npar/1e6:.2f}M | lr_inner={args.lr_inner} lam={args.lam}")

    print("== train (TTT on) ==")
    train(model, cfg, opt, device, args.steps, args.batch, rng, args.lr_inner, args.lam)

    print("== 4-way ablation ==")
    res, K = ablate(model, cfg, device, rng, args.lr_inner)
    for name, pts in res.items():
        best = max(a for _, a in pts)
        print(f"  {name:8s} | best acc {best*100:5.1f}% | points: " +
              " ".join(f"({s:.1f},{a*100:.0f})" for s, a in pts))

    print("== H1: compute to reach matched accuracy ==")
    for ta in [0.90, 0.95, 0.99]:
        a2 = matched_steps(res["2.+halt"], ta)
        a4 = matched_steps(res["4.both"], ta)
        print(f"  acc>={ta:.2f}:  +halt {a2:>5} steps  vs  both {a4:>5} steps"
              f"   -> TTT saves {a2 - a4 if a2!=float('inf') and a4!=float('inf') else 'n/a'}")

    # H2 frontier figure
    plt.figure(figsize=(6, 4.2))
    for name, pts in res.items():
        xs = [s for s, _ in pts]
        ys = [a * 100 for _, a in pts]
        plt.plot(xs, ys, marker="o", ms=4, label=name)
    plt.xlabel("avg compute (latent steps)")
    plt.ylabel("accuracy (%)")
    plt.title("Accuracy vs test-time compute (4-way ablation)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.fig, dpi=130)
    print(f"saved figure -> {args.fig}")


if __name__ == "__main__":
    main()
