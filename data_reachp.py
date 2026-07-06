"""Partial-observation reachability across a query stream.

Combines the two knobs cleanly:
  - a functional graph f (with sinks) gives NATURALLY VARYING depth K =
    distance(v0, sink), discovered by convergence (sink is a fixed point) ->
    surprise-driven halting (the depth knob).
  - f is revealed only PARTIALLY per query and accumulated in memory across the
    stream (the weight knob) -> capability gap + amortization.

Each query reveals m edges (node -> f(node)); a probe start node v0 is answerable
iff every node on its path v0->...->sink had its edge revealed in a PRIOR query.
`ans` (answerable) is returned for the surprise-vs-miss check.

Reuses model_rule.RuleReasoner (associative memory + iterated retrieval); only
the data + training-supervision differ.
"""
import torch


class ReachPConfig:
    n = 24
    n_sinks = 3
    chain_window = 4
    kcap = 8
    m = 5            # edges revealed per query
    Q = 12           # stream length
    T = 10           # latent hops budget (>= kcap)


def _make_graph(cfg, rng):
    n = cfg.n
    order = list(range(n)); rng.shuffle(order)
    f = [0] * n; dist = [0] * n; placed = []
    for idx, node in enumerate(order):
        if idx < cfg.n_sinks:
            f[node] = node; dist[node] = 0
        else:
            lo = max(0, len(placed) - cfg.chain_window)
            cands = [c for c in placed[lo:] if dist[c] < cfg.kcap]
            if not cands:
                cands = [min(placed, key=lambda c: dist[c])]
            p = cands[rng.randrange(len(cands))]
            f[node] = p; dist[node] = dist[p] + 1
        placed.append(node)
    return f, dist


def _path(f, v0):
    p, x = [v0], v0
    while f[x] != x:
        x = f[x]; p.append(x)
    return p                                   # v0 ... sink (inclusive)


def make_batch(batch_size, cfg, rng, device="cpu"):
    """Returns tensors (B,Q,...):
      keys,vals (B,Q,m)  revealed edges (node -> f(node)) this query
      probe (B,Q)        start node v0
      target (B,Q)       sink(v0)
      traj (B,Q,T)       f^1..f^T(v0), clamped at sink (per-hop supervision)
      K (B,Q)            distance to sink
      ans (B,Q)          1 if full path was revealed in a PRIOR query
    """
    n, m, Q, T = cfg.n, cfg.m, cfg.Q, cfg.T
    keys = torch.empty(batch_size, Q, m, dtype=torch.long)
    vals = torch.empty(batch_size, Q, m, dtype=torch.long)
    probe = torch.empty(batch_size, Q, dtype=torch.long)
    target = torch.empty(batch_size, Q, dtype=torch.long)
    traj = torch.empty(batch_size, Q, T, dtype=torch.long)
    K = torch.empty(batch_size, Q, dtype=torch.long)
    ans = torch.zeros(batch_size, Q, dtype=torch.long)
    for b in range(batch_size):
        f, dist = _make_graph(cfg, rng)
        revealed_prior = set()
        for q in range(Q):
            ks = [rng.randrange(n) for _ in range(m)]      # reveal m edges
            for j, x in enumerate(ks):
                keys[b, q, j] = x; vals[b, q, j] = f[x]
            v0 = rng.randrange(n)
            path = _path(f, v0)
            probe[b, q] = v0
            target[b, q] = path[-1]
            K[b, q] = len(path) - 1
            # per-hop target trajectory (clamped at sink)
            x = v0
            for t in range(T):
                x = f[x]; traj[b, q, t] = x
            ans[b, q] = 1 if all(node in revealed_prior for node in path) else 0
            revealed_prior |= set(ks)
    return dict(
        keys=keys.to(device), vals=vals.to(device), probe=probe.to(device),
        target=target.to(device), traj=traj.to(device),
        K=K.to(device), ans=ans.to(device),
    )
