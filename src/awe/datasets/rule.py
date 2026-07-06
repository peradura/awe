"""Hidden-rule (permutation) task with partial observation across a stream.

Episode = one hidden permutation pi over [n]. A stream of Q queries. Each query
reveals m (x, pi(x)) pairs (partial), then tests on a probe x* that is NOT
among the current query's keys -> so x* is answerable ONLY if its mapping was
revealed in a PRIOR query and retained in memory. This forces cross-query
memory (amortization) and creates a genuine capability gap: early queries have
little accumulated knowledge (most probes unanswerable -> low accuracy),
later queries have a lot (high accuracy).

Ground truth `answerable` (= x* seen in a prior query) is returned so we can
correlate the model's surprise with true memory-miss (a confound Codex flagged).

This is deliberately memory-centric (no big graph in context); the whole point
is that the rule must be accumulated, not re-read each query.
"""
import torch


class RuleConfig:
    n = 20          # permutation on [n]
    m = 3           # (x, pi(x)) pairs revealed per query
    Q = 10          # queries per episode (the stream)
    T = 8           # latent retrieval steps budget


def make_batch(batch_size, cfg, rng, device="cpu"):
    """Returns dicts of tensors, all shaped (B, Q, ...):
      keys   (B,Q,m)  probe-context keys x_i
      vals   (B,Q,m)  their images pi(x_i)
      probe  (B,Q)    test input x*
      target (B,Q)    pi(x*)
      ans    (B,Q)    1 if x* revealed in a PRIOR query (answerable from memory)
    """
    n, m, Q = cfg.n, cfg.m, cfg.Q
    keys = torch.empty(batch_size, Q, m, dtype=torch.long)
    vals = torch.empty(batch_size, Q, m, dtype=torch.long)
    probe = torch.empty(batch_size, Q, dtype=torch.long)
    target = torch.empty(batch_size, Q, dtype=torch.long)
    ans = torch.zeros(batch_size, Q, dtype=torch.long)
    for b in range(batch_size):
        perm = list(range(n))
        rng.shuffle(perm)                     # pi: perm[x] = pi(x)
        revealed_prior = set()                # keys seen in strictly earlier queries
        for q in range(Q):
            ks = [rng.randrange(n) for _ in range(m)]     # this query's revealed keys
            for j, x in enumerate(ks):
                keys[b, q, j] = x
                vals[b, q, j] = perm[x]
            cur = set(ks)
            # probe x*: not among current keys -> must come from memory (prior)
            choices = [x for x in range(n) if x not in cur]
            xs = choices[rng.randrange(len(choices))]
            probe[b, q] = xs
            target[b, q] = perm[xs]
            ans[b, q] = 1 if xs in revealed_prior else 0
            revealed_prior |= cur             # this query's keys become "prior" for later
    return dict(
        keys=keys.to(device), vals=vals.to(device), probe=probe.to(device),
        target=target.to(device), ans=ans.to(device),
    )
