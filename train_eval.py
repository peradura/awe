"""Increment 1: validate the harness on functional-graph reachability.

Checks:
  (1) the latent recurrent reasoner learns to follow f to the sink,
  (2) accuracy improves with test-time depth (per-K peaks at needed depth),
  (3) a convergence signal (step-to-step KL / entropy) tracks difficulty K
      -> steps-to-converge correlates with K  [the halting signal we'll use].

Usage: python train_eval.py --steps 3000
"""
import argparse
import random

import torch
import torch.nn.functional as F

from data import GraphConfig, vocab_size, v0_pos, make_batch
from model import Reasoner, entropy, step_kl


def train(model, cfg, opt, device, steps, batch, rng, log_every=300):
    model.train()
    T = cfg.T
    for it in range(1, steps + 1):
        tok, pad, traj, _ = make_batch(batch, cfg, rng, device)
        H, s0, pad = model.embed(tok, pad)
        _, _, logits_traj = model.run(H, s0, pad, T, return_traj=True)
        # supervise readout at every hop t -> f^t(v0)  (traj[:, t-1])
        loss = 0.0
        for t in range(T):
            loss = loss + F.cross_entropy(logits_traj[t], traj[:, t])
        loss = loss / T
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % log_every == 0 or it == 1:
            print(f"  step {it:5d} | loss {loss.item():.4f}")


@torch.no_grad()
def evaluate(model, cfg, device, rng, n=6000):
    model.eval()
    tok, pad, traj, K = make_batch(n, cfg, rng, device)
    H, s0, pad = model.embed(tok, pad)
    tgt = traj[:, -1]                              # sink = converged answer
    Tev = cfg.T + 4                                # allow a few extra hops

    # (2) accuracy vs fixed test-time depth
    _, _, lt = model.run(H, s0, pad, Tev, return_traj=True)
    depth_acc = {}
    for r in [1, 2, 4, 6, 8, cfg.T, Tev]:
        depth_acc[r] = (lt[r - 1].argmax(-1) == tgt).float().mean().item()

    # (3) convergence signal vs difficulty
    # steps-to-converge = first hop where prediction stops changing till the end
    preds = torch.stack([l.argmax(-1) for l in lt], dim=1)      # (n, Tev)
    conv = torch.full((n,), Tev, device=device)
    for t in range(Tev - 1, -1, -1):
        stable = (preds[:, t:] == preds[:, t:t + 1]).all(dim=1)
        conv = torch.where(stable, torch.full_like(conv, t + 1), conv)
    corr_conv = torch.corrcoef(torch.stack([K.float(), conv.float()]))[0, 1].item()

    # entropy at a deep step, per K
    ent_last = entropy(lt[cfg.T - 1])
    corr_ent = torch.corrcoef(torch.stack([K.float(), ent_last]))[0, 1].item()

    per_k = {}
    for k in range(0, int(K.max().item()) + 1):
        m = K == k
        if m.sum() >= 5:
            per_k[k] = (
                int(m.sum().item()),
                conv[m].float().mean().item(),
                (lt[cfg.T - 1].argmax(-1)[m] == tgt[m]).float().mean().item(),
            )
    return depth_acc, corr_conv, corr_ent, per_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GraphConfig()

    model = Reasoner(vocab_size(cfg.n), cfg.n, v0_pos(cfg.n), d=args.d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    npar = sum(p.numel() for p in model.parameters())
    print(f"device={device} | params={npar/1e6:.2f}M | n={cfg.n} sinks={cfg.n_sinks} T={cfg.T}")

    print("== train ==")
    train(model, cfg, opt, device, args.steps, args.batch, rng)

    print("== eval ==")
    depth_acc, corr_conv, corr_ent, per_k = evaluate(model, cfg, device, rng)
    print(" accuracy vs test-time depth:")
    for r, a in depth_acc.items():
        print(f"   r={r:3d} -> {a*100:5.1f}%")
    print(f" corr(K, steps-to-converge) = {corr_conv:+.3f}   [halting signal]")
    print(f" corr(K, entropy@deep)      = {corr_ent:+.3f}")
    print("  K :   n    conv-step   acc")
    for k, (cnt, cv, acc) in per_k.items():
        print(f"  {k:2d} : {cnt:5d}    {cv:6.2f}    {acc*100:5.1f}%")


if __name__ == "__main__":
    main()
