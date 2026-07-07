"""MQAR (multi-query associative recall) across a query stream.

Standard recall stress test (Zoology, Arora et al. 2312.04927; Based 2402.18668).
Adapted to the same streaming interface as `reachp.py` so the halting-signal
harness (`ablation_reachp3.trace_stream` etc.) is reused verbatim — the external
transfer test for convergence-halting (see `docs/mqar_design.md`).

Per example: a fixed dictionary D: key -> value over a vocab of n symbols. Each
query reveals m (k, D[k]) pairs (written to the fast-weight memory) and probes one
key; the probe is answerable iff its key was revealed by the END of this query.
Difficulty axis K = number of distinct keys in memory when probed (memory load /
interference), the MQAR analogue of reachability depth.

Single-hop retrieval, so per-step supervision `traj[:, t] = target` for all t
(converge to the value and hold — exactly the fixed-point convergence that
conv/dstate halting detects). Reuses `models.memory.RuleReasoner` unchanged.
"""
import torch


class MqarConfig:
    n = 64            # vocab (keys and values share it); MQAR canonical is 8192 —
    #                   smaller here for a ~1M-param synthetic pilot (coda = n classes)
    m = 4             # KV pairs revealed per query
    Q = 12            # stream length
    T = 6             # latent retrieval steps budget
    p_ans = 0.7       # fraction of probes drawn from already-revealed keys
    lazy = False      # large-vocab generator path (standard-config anchor); the
    #                   distribution is identical, only the rng call sequence differs
    #                   — keep False for reproducing the archived mini-MQAR runs


def make_batch(batch_size, cfg, rng, device="cpu"):
    """Returns tensors (B,Q,...) matching reachp.make_batch's interface:
      keys, vals (B,Q,m)  revealed (key -> D[key]) pairs this query
      probe (B,Q)         queried key
      target (B,Q)        D[probe]  (its value)
      traj (B,Q,T)        target repeated over T steps (converge-and-hold)
      K (B,Q)             # distinct keys in memory by end of this query (load)
      ans (B,Q)           1 iff probe key revealed by end of this query
    """
    n, m, Q, T = cfg.n, cfg.m, cfg.Q, cfg.T
    keys = torch.empty(batch_size, Q, m, dtype=torch.long)
    vals = torch.empty(batch_size, Q, m, dtype=torch.long)
    probe = torch.empty(batch_size, Q, dtype=torch.long)
    target = torch.empty(batch_size, Q, dtype=torch.long)
    traj = torch.empty(batch_size, Q, T, dtype=torch.long)
    K = torch.empty(batch_size, Q, dtype=torch.long)
    ans = torch.zeros(batch_size, Q, dtype=torch.long)
    lazy = getattr(cfg, "lazy", False)
    for b in range(batch_size):
        if lazy:
            # large-vocab path (standard-config anchor): identical distribution —
            # D[x] i.i.d. uniform assigned on first touch; unseen probe by rejection
            # sampling (uniform over unseen since |revealed| << n). O(Q*m), not O(n).
            D = {}
            def dval(x):
                if x not in D:
                    D[x] = rng.randrange(n)
                return D[x]
        else:
            Dl = [rng.randrange(n) for _ in range(n)]        # dictionary key -> value
            dval = Dl.__getitem__
        revealed = set()
        for q in range(Q):
            ks = [rng.randrange(n) for _ in range(m)]       # reveal m pairs
            for j, x in enumerate(ks):
                keys[b, q, j] = x; vals[b, q, j] = dval(x)
            revealed |= set(ks)                              # written before retrieval
            # probe: answerable (revealed) with prob p_ans, else an unseen key
            if revealed and rng.random() < cfg.p_ans:
                pk = rng.choice(tuple(revealed))
            elif lazy:
                pk = rng.randrange(n)                        # rejection sampling
                while pk in revealed:
                    pk = rng.randrange(n)
            else:
                unseen = [k for k in range(n) if k not in revealed]
                pk = rng.choice(unseen) if unseen else rng.choice(tuple(revealed))
            probe[b, q] = pk
            target[b, q] = dval(pk)
            traj[b, q, :] = dval(pk)                         # single-hop: hold the value
            K[b, q] = len(revealed)
            ans[b, q] = 1 if pk in revealed else 0
    return dict(
        keys=keys.to(device), vals=vals.to(device), probe=probe.to(device),
        target=target.to(device), traj=traj.to(device),
        K=K.to(device), ans=ans.to(device),
    )
