"""Amortization experiment: does a memory that PERSISTS across a stream of
queries on one graph let later queries halt in fewer hops?

Trains one model (memory persists across the query stream), then compares:
  both  : persist memory + surprise halting   (AWE amortization)
  +halt : halting, memory reset per query      (no cross-query adaptation)
  +amort: persist memory, fixed depth
  fixed : no memory, fixed depth

Headline (H1 for amortization): avg halt-step vs QUERY INDEX.
  both  -> should DECREASE across the stream (memory warms up)
  +halt -> flat (every query pays full search)
Saved figure: amort_curve.png.

GPU NOTE: trains a model. Do not launch on a shared GPU without the user's OK.
The Q-loop makes each training step ~Q x heavier than the single-query pilot;
GPU strongly recommended for the full run.
"""
import argparse
import random

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_amort import AmortConfig, vocab_size, v0_pos, make_batch
from model_amort import AmortReasoner


def train(model, cfg, opt, device, steps, batch, rng, lr_inner, lam, log_every=200):
    model.train()
    T = cfg.T
    for it in range(1, steps + 1):
        tok, pad, traj, _ = make_batch(batch, cfg, rng, device)
        delta = None
        loss = 0.0
        for q in range(cfg.Q):                       # stream; memory persists
            H, s0, p = model.embed(tok[:, q], pad[:, q])
            lt, st, delta = model.rollout(H, s0, p, T, delta=delta,
                                          ttt=True, lr_inner=lr_inner)
            ce = sum(F.cross_entropy(lt[t], traj[:, q, t]) for t in range(T)) / T
            loss = loss + ce + lam * torch.stack(st).mean()
        loss = loss / cfg.Q
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % log_every == 0 or it == 1:
            print(f"  step {it:5d} | loss {loss.item():.4f}")


@torch.no_grad()
def eval_stream(model, cfg, device, tok, pad, tgt, ttt, persist, halt, tau):
    """Returns (steps_by_q, acc_by_q, all_surprise)."""
    B = tok.shape[0]
    T = cfg.T
    delta = None
    steps_by_q, acc_by_q = [], []
    sur_pool = []
    for q in range(cfg.Q):
        H, s0, p = model.embed(tok[:, q], pad[:, q])
        din = delta if (persist and ttt) else None
        lt, st, delta_new = model.rollout(H, s0, p, T, delta=din, ttt=ttt)
        if persist and ttt:
            delta = delta_new
        preds = torch.stack([l.argmax(-1) for l in lt], dim=1)   # (B,T)
        sur = torch.stack(st, dim=1)                             # (B,T)
        sur_pool.append(sur)
        if halt:
            below = sur < tau
            has = below.any(dim=1)
            first = below.int().argmax(dim=1)
            hs = torch.where(has, first, torch.full_like(first, T - 1))
        else:
            hs = torch.full((B,), T - 1, device=device)
        pred = preds[torch.arange(B, device=device), hs]
        steps_by_q.append((hs + 1).float().mean().item())
        acc_by_q.append((pred == tgt[:, q]).float().mean().item())
    return steps_by_q, acc_by_q, torch.cat(sur_pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_inner", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", type=str, default="amort_curve.png")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = AmortConfig()
    model = AmortReasoner(vocab_size(cfg.n), cfg.n, v0_pos(cfg.n), d=args.d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    npar = sum(p.numel() for p in model.parameters())
    print(f"device={device} | params={npar/1e6:.2f}M | n={cfg.n} Q={cfg.Q} T={cfg.T}")

    print("== train (memory persists across the stream) ==")
    train(model, cfg, opt, device, args.steps, args.batch, rng, args.lr_inner, args.lam)

    # eval set
    tok, pad, traj, K = make_batch(500, cfg, rng, device)
    tgt = traj[:, :, -1]

    # calibrate tau on the 'both' surprises (median)
    _, _, sur_all = eval_stream(model, cfg, device, tok, pad, tgt,
                                ttt=True, persist=True, halt=True, tau=1e9)
    tau = torch.quantile(sur_all, 0.5).item()
    print(f"== eval (tau={tau:.4f}) ==")

    configs = {
        "fixed":  dict(ttt=False, persist=False, halt=False),
        "+halt":  dict(ttt=False, persist=False, halt=True),
        "+amort": dict(ttt=True,  persist=True,  halt=False),
        "both":   dict(ttt=True,  persist=True,  halt=True),
    }
    curves = {}
    for name, c in configs.items():
        sbq, abq, _ = eval_stream(model, cfg, device, tok, pad, tgt, tau=tau, **c)
        curves[name] = (sbq, abq)
        print(f"  {name:7s} | avg steps {sum(sbq)/len(sbq):5.2f} | "
              f"acc {sum(abq)/len(abq)*100:5.1f}% | steps/query " +
              " ".join(f"{s:.1f}" for s in sbq))

    # H1 amortization: does 'both' halt earlier on later queries vs '+halt'?
    b0, b1 = curves["both"][0][0], curves["both"][0][-1]
    h0, h1 = curves["+halt"][0][0], curves["+halt"][0][-1]
    print("== H1 (amortization) ==")
    print(f"  both : q0={b0:.2f} -> q{cfg.Q-1}={b1:.2f}  (Δ {b1-b0:+.2f} steps)")
    print(f"  +halt: q0={h0:.2f} -> q{cfg.Q-1}={h1:.2f}  (Δ {h1-h0:+.2f} steps)")
    print("  -> amortization if 'both' drops while '+halt' stays flat.")

    plt.figure(figsize=(6, 4.2))
    for name in ["both", "+halt", "+amort", "fixed"]:
        plt.plot(range(cfg.Q), curves[name][0], marker="o", ms=4, label=name)
    plt.xlabel("query index in stream")
    plt.ylabel("avg halt-step (compute)")
    plt.title("Amortization: compute per query across a shared-graph stream")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.fig, dpi=130)
    print(f"saved figure -> {args.fig}")


if __name__ == "__main__":
    main()
