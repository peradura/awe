"""Partial-obs reachability, sharpened (Codex levers, in priority order):
  1. easier distribution + K-curriculum (ramp kcap 1->4 over training)
  2. auxiliary next-node loss (read each revealed key -> predict its value):
     gives short credit assignment for the memory, the key fix for chain-following
  3. more capacity (d=256)
Architecture unchanged (reuses awe.models.memory.RuleReasoner) so any gain is not
confounded by a new model.

GPU NOTE: heavier (8k steps); ask before launching on a shared GPU.
"""
import argparse
import random

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from awe.datasets.reachp import make_batch
from awe.models.memory import RuleReasoner


class Cfg:
    def __init__(self):
        self.n = 16
        self.n_sinks = 2
        self.chain_window = 3
        self.kcap = 1            # ramped by the curriculum
        self.m = 8               # more edges revealed -> memory fills faster
        self.Q = 12
        self.T = 6


def slice_q(data, q):
    return {k: v[:, q] for k, v in data.items()}


def aux_readback(model, delta, keys, vals):
    """Read each revealed key from post-write memory -> CE toward its value."""
    W = model.W_base.unsqueeze(0) + delta
    loss = 0.0
    for j in range(keys.shape[1]):
        k = model.mem_ln(model.node(keys[:, j]))
        read = torch.bmm(W, k.unsqueeze(-1)).squeeze(-1)
        loss = loss + F.cross_entropy(model.coda(read), vals[:, j])
    return loss / keys.shape[1]


def train(model, cfg, opt, device, steps, batch, rng, lam_aux, log_every=400):
    model.train()
    for it in range(1, steps + 1):
        cfg.kcap = 1 + min(3, (it - 1) // max(1, steps // 4))   # curriculum 1->4
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
            print(f"  step {it:5d} | kcap {cfg.kcap} | loss {loss.item():.4f}")


@torch.no_grad()
def run_stream(model, cfg, data, persist, halt, tau):
    B = data["probe"].shape[0]; T = cfg.T
    delta = model.new_memory(B, data["probe"].device)
    acc_q, steps_q = [], []
    corr_all, hs_all, K_all, ans_all, sur0_all = [], [], [], [], []
    for q in range(cfg.Q):
        bq = slice_q(data, q)
        d_in = delta if persist else model.new_memory(B, data["probe"].device)
        logits_t, sur_t, d_out = model.query(d_in, bq, T, write=True)
        if persist:
            delta = d_out
        preds = torch.stack([l.argmax(-1) for l in logits_t], 1)
        sur = torch.stack(sur_t, 1)
        if halt:
            below = sur < tau; has = below.any(1); first = below.int().argmax(1)
            hs = torch.where(has, first, torch.full_like(first, T - 1))
        else:
            hs = torch.full((B,), T - 1, device=preds.device)
        pred = preds[torch.arange(B, device=preds.device), hs]
        corr = (pred == bq["target"])
        acc_q.append(corr.float().mean().item()); steps_q.append((hs + 1).float().mean().item())
        corr_all.append(corr); hs_all.append(hs + 1); K_all.append(bq["K"])
        ans_all.append(bq["ans"]); sur0_all.append(sur[:, 0])
    return dict(acc_q=acc_q, steps_q=steps_q, corr=torch.cat(corr_all),
                hs=torch.cat(hs_all).float(), K=torch.cat(K_all),
                ans=torch.cat(ans_all).float(), sur0=torch.cat(sur0_all))


@torch.no_grad()
def calib_tau(model, cfg, data, persist):
    return torch.quantile(run_stream(model, cfg, data, persist, False, 1e9)["sur0"], 0.5).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam_aux", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", type=str, default="reachp2_curve.png")
    args = ap.parse_args()

    torch.manual_seed(args.seed); rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Cfg()
    model = RuleReasoner(cfg.n, d=args.d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    npar = sum(p.numel() for p in model.parameters())
    print(f"device={device} | params={npar/1e6:.2f}M | n={cfg.n} m={cfg.m} Q={cfg.Q} T={cfg.T} | curriculum kcap 1->4")

    print("== train (curriculum + aux next-node loss) ==")
    train(model, cfg, opt, device, args.steps, args.batch, rng, args.lam_aux)

    cfg.kcap = 4
    data = make_batch(1000, cfg, rng, device)
    configs = {"fixed": dict(persist=False, halt=False), "+halt": dict(persist=False, halt=True),
               "persist": dict(persist=True, halt=False), "both": dict(persist=True, halt=True)}
    print("== eval (per-config tau) ==")
    res = {}
    for name, c in configs.items():
        tau = calib_tau(model, cfg, data, c["persist"]) if c["halt"] else 1e9
        r = run_stream(model, cfg, data, tau=tau, **c); res[name] = r
        print(f"  {name:7s} | acc {sum(r['acc_q'])/cfg.Q*100:5.1f}% | steps {sum(r['steps_q'])/cfg.Q:4.2f} | "
              f"acc/q " + " ".join(f"{a*100:.0f}" for a in r['acc_q']))

    r = res["both"]; mask = (r["ans"] > 0.5) & r["corr"]
    print("== depth vs K (both; answerable & correct) ==")
    kx, ky = [], []
    for k in sorted(set(int(x) for x in r["K"][mask].tolist())):
        mk = mask & (r["K"] == k)
        if mk.sum() >= 10:
            kx.append(k); ky.append(r["hs"][mk].mean().item())
            print(f"  K={k}: halt-step {ky[-1]:.2f} (n={int(mk.sum())})")
    corr = torch.corrcoef(torch.stack([res['persist']['ans'], res['persist']['sur0']]))[0, 1].item()
    ba = res["both"]["acc_q"]
    print(f"== surprise vs miss corr = {corr:+.3f} | amortization both: {ba[0]*100:.0f}%->{ba[-1]*100:.0f}% ==")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in ["both", "+halt", "persist", "fixed"]:
        ax[0].plot(range(cfg.Q), [a*100 for a in res[name]["acc_q"]], marker="o", ms=4, label=name)
    ax[0].set(xlabel="query index", ylabel="accuracy (%)", title="Amortization"); ax[0].legend(); ax[0].grid(alpha=.3)
    if kx:
        ax[1].plot(kx, ky, marker="o", color="tab:blue")
        ax[1].plot([kx[0], kx[-1]], [kx[0]+1, kx[-1]+1], "--", color="gray", alpha=.6, label="y=K+1"); ax[1].legend()
    ax[1].set(xlabel="difficulty K", ylabel="halt-step", title="Depth vs difficulty"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(args.fig, dpi=130); print(f"saved figure -> {args.fig}")


if __name__ == "__main__":
    main()
