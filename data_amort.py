"""Amortization task: one graph, a STREAM of queries.

An episode is a single functional graph G plus Q queries (different start
nodes) on that same G. Each query is encoded as the proven single-query
sequence  [BOS, f(0..n-1), SEP, v0, EOS]. The point: a fast-weight memory
that PERSISTS across the Q queries adapts to G, so later queries should be
answerable with fewer latent hops (amortization = weight adaptation buying
depth). Memory resets at each new episode (new graph).

Termination guaranteed (sinks are self-loops; non-sinks point strictly
closer). Difficulty K = hops from v0 to its sink, capped at kcap.
"""
import torch


class AmortConfig:
    n = 24
    n_sinks = 2
    chain_window = 4      # smaller -> deeper chains -> broader K
    kcap = 12
    T = 12                # latent hops modeled per query
    Q = 8                 # queries per episode (the amortization stream)


def vocab_size(n=AmortConfig.n):
    return n + 4


def v0_pos(n=AmortConfig.n):
    return n + 2


def _make_graph(cfg, rng):
    n = cfg.n
    order = list(range(n))
    rng.shuffle(order)
    f = [0] * n
    dist = [0] * n
    placed = []
    for idx, node in enumerate(order):
        if idx < cfg.n_sinks:
            f[node] = node
            dist[node] = 0
        else:
            lo = max(0, len(placed) - cfg.chain_window)
            cands = [c for c in placed[lo:] if dist[c] < cfg.kcap]
            if not cands:
                cands = [min(placed, key=lambda c: dist[c])]
            p = cands[rng.randrange(len(cands))]
            f[node] = p
            dist[node] = dist[p] + 1
        placed.append(node)
    return f, dist


def _traj(f, v0, T):
    out, x = [], v0
    for _ in range(T):
        x = f[x]
        out.append(x)
    return out


def make_batch(batch_size, cfg, rng, device="cpu"):
    """Returns tok (B,Q,L), pad (B,Q,L), traj (B,Q,T), K (B,Q).

    B episodes; within an episode all Q queries share one graph.
    """
    n = cfg.n
    SEP, BOS, EOS = n, n + 1, n + 2
    L = n + 4
    tok = torch.empty(batch_size, cfg.Q, L, dtype=torch.long)
    traj = torch.empty(batch_size, cfg.Q, cfg.T, dtype=torch.long)
    K = torch.empty(batch_size, cfg.Q, dtype=torch.long)
    for b in range(batch_size):
        f, dist = _make_graph(cfg, rng)
        base = [BOS] + f + [SEP, 0, EOS]         # v0 slot at index n+2
        for q in range(cfg.Q):
            v0 = rng.randrange(n)
            seq = list(base)
            seq[n + 2] = v0
            tok[b, q] = torch.tensor(seq, dtype=torch.long)
            traj[b, q] = torch.tensor(_traj(f, v0, cfg.T), dtype=torch.long)
            K[b, q] = dist[v0]
    pad = torch.zeros(batch_size, cfg.Q, L, dtype=torch.bool)
    return tok.to(device), pad.to(device), traj.to(device), K.to(device)
