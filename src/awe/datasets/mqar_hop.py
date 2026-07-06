"""Multi-hop MQAR — the harder, signal-*discriminating* variant.

The single-hop MQAR (`mqar.py`) confirmed convergence-halting transfers but did
NOT discriminate signals: its errors were unanswerable probes (wrong at any depth),
so halting early was harmless for every signal. This variant reintroduces a
**depth∝difficulty** structure so that halting *too early* is costly again — the
condition under which reachp found entropy/recon lose but conv/dstate don't.

H-hop associative recall: a random map g: symbol→symbol over n symbols. Each query
reveals m bindings (k → g[k]) into the persistent memory; a probe asks for
**g^H(q)** — the symbol reached by following the binding H times (H ∈ 1..Hmax,
mixed = the difficulty axis K). Answering needs the H bindings on the path
(q, g[q], …, g^{H-1}[q]) all in memory, and the latent loop must *chase* the chain
— so a probe with H=3 is confidently wrong if halted at step 1. p_ans is high so
most errors are recall/depth errors, not unanswerable.

Per-hop supervision `traj[t] = g^{min(t+1,H)}(q)` (chase, then hold at the answer),
so the reasoner learns to follow one hop per latent step. Same streaming interface
as `reachp.py` / `mqar.py`; reuses `models.memory.RuleReasoner` and the reachp3
halting harness unchanged.
"""
import torch


class HopConfig:
    n = 64            # vocab (symbols; g maps symbol->symbol so values are composable)
    m = 6             # bindings revealed per query
    Q = 12            # stream length
    T = 6             # latent steps budget (>= Hmax)
    Hmax = 3          # max hops; difficulty K = H drawn uniform 1..Hmax
    p_ans = 0.85      # target fraction of answerable probes (full path revealed)


def make_batch(batch_size, cfg, rng, device="cpu"):
    """Returns tensors (B,Q,...) matching reachp.make_batch's interface:
      keys, vals (B,Q,m)  revealed bindings (k -> g[k]) this query
      probe (B,Q)         start symbol q
      target (B,Q)        g^H(q)  (the H-hop answer)
      traj (B,Q,T)        g^{min(t+1,H)}(q)  (chase then hold — per-hop supervision)
      K (B,Q)             H  (hop count = difficulty)
      ans (B,Q)           1 iff the H path keys (q..g^{H-1}(q)) are all revealed
    """
    n, m, Q, T, Hmax = cfg.n, cfg.m, cfg.Q, cfg.T, cfg.Hmax
    keys = torch.empty(batch_size, Q, m, dtype=torch.long)
    vals = torch.empty(batch_size, Q, m, dtype=torch.long)
    probe = torch.empty(batch_size, Q, dtype=torch.long)
    target = torch.empty(batch_size, Q, dtype=torch.long)
    traj = torch.empty(batch_size, Q, T, dtype=torch.long)
    K = torch.empty(batch_size, Q, dtype=torch.long)
    ans = torch.zeros(batch_size, Q, dtype=torch.long)
    for b in range(batch_size):
        g = [rng.randrange(n) for _ in range(n)]           # random map symbol->symbol
        revealed = set()
        for q in range(Q):
            ks = [rng.randrange(n) for _ in range(m)]       # reveal m bindings
            for j, x in enumerate(ks):
                keys[b, q, j] = x; vals[b, q, j] = g[x]
            revealed |= set(ks)                             # written before retrieval
            H = rng.randint(1, Hmax)
            # path keys that must be in memory: q, g[q], ..., g^{H-1}[q]
            def path_keys(start, h):
                p, x = [], start
                for _ in range(h):
                    p.append(x); x = g[x]
                return p
            start = None
            if rng.random() < cfg.p_ans:                    # try an answerable start
                cand = [x for x in revealed
                        if all(k in revealed for k in path_keys(x, H))]
                if cand:
                    start = rng.choice(cand)
            if start is None:
                start = rng.randrange(n)
            # full path start .. g^H(start)
            x = start; full = [x]
            for _ in range(H):
                x = g[x]; full.append(x)
            probe[b, q] = start
            target[b, q] = full[H]                          # g^H(start)
            K[b, q] = H
            for t in range(T):
                traj[b, q, t] = full[min(t + 1, H)]         # chase then hold at answer
            ans[b, q] = 1 if all(k in revealed for k in path_keys(start, H)) else 0
    return dict(
        keys=keys.to(device), vals=vals.to(device), probe=probe.to(device),
        target=target.to(device), traj=traj.to(device),
        K=K.to(device), ans=ans.to(device),
    )
