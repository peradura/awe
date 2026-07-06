"""Functional-graph reachability (per-example graph, in-context).

Each example gives a small functional graph f: [n]->[n] (as a token table)
plus a start node v0. The answer is the SINK (fixed point) reached by
following f from v0. Difficulty K = distance from v0 to its sink = the
number of latent hops genuinely required.

Difficulty control (Increment-2 hardening):
  Non-sink nodes attach to a RECENT already-placed node (a sliding window of
  size `chain_window`), which grows long chains -> a broad spread of K,
  including hard high-K instances. `kcap` caps the longest path so every
  example is solvable within T hops (headroom lives at depth < K, where the
  model is compute-constrained and weight-adaptation can help).

Termination is guaranteed: sinks are self-loops; every other node points to
a strictly-closer (already placed) node, so following f reaches a sink.

Encoding (vocab = n+4):
  node ids : 0 .. n-1
  SEP : n     BOS : n+1     EOS : n+2     PAD : n+3
  seq = [BOS, f(0), f(1), ..., f(n-1), SEP, v0, EOS]   (length n+4)
  v0 is always at position n+2.
"""
import torch


class GraphConfig:
    n = 28
    n_sinks = 2
    chain_window = 4     # smaller -> deeper chains -> more high-K mass
    kcap = 12            # max path length (== T); keeps every example solvable
    T = 12               # latent hops modeled during training


def vocab_size(n=GraphConfig.n):
    return n + 4


def v0_pos(n=GraphConfig.n):
    return n + 2


def _make_graph(cfg, rng):
    """Returns (f, dist) with dist[node] = hops to its sink, capped at kcap."""
    n = cfg.n
    order = list(range(n))
    rng.shuffle(order)
    f = [0] * n
    dist = [0] * n
    placed = []
    for idx, node in enumerate(order):
        if idx < cfg.n_sinks:
            f[node] = node                       # sink: self-loop
            dist[node] = 0
        else:
            lo = max(0, len(placed) - cfg.chain_window)
            window = placed[lo:]
            cands = [c for c in window if dist[c] < cfg.kcap]
            if not cands:                        # respect the depth cap
                cands = [min(placed, key=lambda c: dist[c])]
            p = cands[rng.randrange(len(cands))]
            f[node] = p
            dist[node] = dist[p] + 1
        placed.append(node)
    return f, dist


def _traj(f, v0, T):
    """f^1..f^T(v0), clamped once the sink (fixed point) is reached."""
    out, x = [], v0
    for _ in range(T):
        x = f[x]
        out.append(x)
    return out


def make_batch(batch_size, cfg, rng, device="cpu"):
    """Returns tokens (B,L), padmask (B,L), traj_tgt (B,T), K (B,)."""
    n = cfg.n
    SEP, BOS, EOS = n, n + 1, n + 2
    toks, trajs, Ks = [], [], []
    for _ in range(batch_size):
        f, dist = _make_graph(cfg, rng)
        v0 = rng.randrange(n)
        toks.append([BOS] + f + [SEP, v0, EOS])
        trajs.append(_traj(f, v0, cfg.T))
        Ks.append(dist[v0])
    tok = torch.tensor(toks, dtype=torch.long, device=device)
    pad = torch.zeros((batch_size, len(toks[0])), dtype=torch.bool, device=device)
    traj = torch.tensor(trajs, dtype=torch.long, device=device)
    K = torch.tensor(Ks, dtype=torch.long, device=device)
    return tok, pad, traj, K
