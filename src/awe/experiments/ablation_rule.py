"""Hidden-rule pilot: does a persistent surprise-gated memory yield
BOTH a capability gap (accuracy rises across the stream) AND amortization
(halt-steps fall across the stream)?

Configs (per-config tau -> fair, addresses the earlier calibration confound):
  both   : persist memory + halt     (AWE)
  +halt  : memory reset per query + halt
  persist: persist memory, fixed depth
  fixed  : reset, fixed depth

Also reports corr(surprise, true memory-miss) -- surprise should be high
exactly when the probe was NOT seen before.

GPU NOTE: trains a model. Ask before launching on a shared GPU.
"""
import argparse
import random

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from awe.datasets.rule import RuleConfig, make_batch
from awe.models.memory import RuleReasoner


def slice_q(data, q):
    return {k: v[:, q] for k, v in data.items()}


def train(model, cfg, opt, device, steps, batch, rng, log_every=300):
    model.train()
    T = cfg.T
    for it in range(1, steps + 1):
        data = make_batch(batch, cfg, rng, device)
        delta = model.new_memory(batch, device)
        loss = 0.0
        for q in range(cfg.Q):
            bq = slice_q(data, q)
            logits_t, _, delta = model.query(delta, bq, T, write=True)
            tgt = bq["target"]
            loss = loss + sum(F.cross_entropy(l, tgt) for l in logits_t) / T
        loss = loss / cfg.Q
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % log_every == 0 or it == 1:
            print(f"  step {it:5d} | loss {loss.item():.4f}")


@torch.no_grad()
def run_stream(model, cfg, data, persist, halt, tau):
    """Returns per-query accuracy, per-query avg steps, surprise@step0, ans."""
    model.eval()
    B = data["probe"].shape[0]
    T = cfg.T
    delta = model.new_memory(B, data["probe"].device)
    acc_q, steps_q, sur0, ansf = [], [], [], []
    for q in range(cfg.Q):
        bq = slice_q(data, q)
        d_in = delta if persist else model.new_memory(B, data["probe"].device)
        logits_t, sur_t, d_out = model.query(d_in, bq, T, write=True)
        if persist:
            delta = d_out
        preds = torch.stack([l.argmax(-1) for l in logits_t], 1)   # (B,T)
        sur = torch.stack(sur_t, 1)                                # (B,T)
        if halt:
            below = sur < tau
            has = below.any(1)
            first = below.int().argmax(1)
            hs = torch.where(has, first, torch.full_like(first, T - 1))
        else:
            hs = torch.full((B,), T - 1, device=preds.device)
        pred = preds[torch.arange(B, device=preds.device), hs]
        acc_q.append((pred == bq["target"]).float().mean().item())
        steps_q.append((hs + 1).float().mean().item())
        sur0.append(sur[:, 0])
        ansf.append(bq["ans"].float())
    return acc_q, steps_q, torch.cat(sur0), torch.cat(ansf)


@torch.no_grad()
def calib_tau(model, cfg, data, persist):
    _, _, sur0, _ = run_stream(model, cfg, data, persist, halt=False, tau=1e9)
    return torch.quantile(sur0, 0.5).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", type=str, default="rule_curve.png")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RuleConfig()
    model = RuleReasoner(cfg.n, d=args.d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    npar = sum(p.numel() for p in model.parameters())
    print(f"device={device} | params={npar/1e6:.2f}M | n={cfg.n} m={cfg.m} Q={cfg.Q} T={cfg.T}")

    print("== train (memory persists across the stream) ==")
    train(model, cfg, opt, device, args.steps, args.batch, rng)

    data = make_batch(1000, cfg, rng, device)
    # held-out calibration batch: tau must not be tuned on the scored batch
    calib = make_batch(256, cfg, rng, device)
    configs = {
        "fixed":   dict(persist=False, halt=False),
        "+halt":   dict(persist=False, halt=True),
        "persist": dict(persist=True,  halt=False),
        "both":    dict(persist=True,  halt=True),
    }
    print("== eval (per-config tau, held-out calibration) ==")
    curves = {}
    for name, c in configs.items():
        tau = calib_tau(model, cfg, calib, c["persist"]) if c["halt"] else 1e9
        acc_q, steps_q, sur0, ansf = run_stream(model, cfg, data, tau=tau, **c)
        curves[name] = (acc_q, steps_q)
        print(f"  {name:7s} | acc {sum(acc_q)/len(acc_q)*100:5.1f}% | "
              f"steps {sum(steps_q)/len(steps_q):4.2f} | "
              f"acc/q " + " ".join(f"{a*100:.0f}" for a in acc_q))

    # capability gap + amortization (both): q0 -> qlast
    ba, bs = curves["both"]
    print("== headline (both) ==")
    print(f"  accuracy: q0={ba[0]*100:.0f}% -> q{cfg.Q-1}={ba[-1]*100:.0f}%  (Δ {(ba[-1]-ba[0])*100:+.0f}pp)")
    print(f"  steps   : q0={bs[0]:.2f}  -> q{cfg.Q-1}={bs[-1]:.2f}   (Δ {bs[-1]-bs[0]:+.2f})")

    # confound check: surprise vs true memory-miss
    _, _, sur0, ansf = run_stream(model, cfg, data, persist=True, halt=False, tau=1e9)
    corr = torch.corrcoef(torch.stack([ansf, sur0]))[0, 1].item()
    print(f"  corr(answerable, surprise@0) = {corr:+.3f}  (expect negative: seen->low surprise)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in ["both", "+halt", "persist", "fixed"]:
        ax[0].plot(range(cfg.Q), [a * 100 for a in curves[name][0]], marker="o", ms=4, label=name)
        ax[1].plot(range(cfg.Q), curves[name][1], marker="o", ms=4, label=name)
    ax[0].set(xlabel="query index", ylabel="accuracy (%)", title="Capability gap (accuracy rises as memory fills)")
    ax[1].set(xlabel="query index", ylabel="avg halt-step", title="Amortization (compute falls across stream)")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.fig, dpi=130)
    print(f"saved figure -> {args.fig}")


if __name__ == "__main__":
    main()
