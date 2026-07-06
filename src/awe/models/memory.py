"""Memory-centric reasoner for the hidden-rule task.

Two knobs, driven by two related but DISTINCT signals (the shared-signal
unification is the open hypothesis — see README status note; the shared-signal
variant lives in awe.models.ttt):
  [B weight] a persistent associative fast-weight M writes each revealed pair
             (x -> pi(x)) via a normalized delta rule (write strength grows
             with the reconstruction miss). M persists across the query stream.
  [A depth]  retrieval runs latent steps reading M; surprise = readout entropy
             (low = confident hit -> halt; high = miss -> keep searching).

M carried across queries (persist) = amortization; reset per query = baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RuleReasoner(nn.Module):
    def __init__(self, n, d=128):
        super().__init__()
        self.n = n
        self.d = d
        self.node = nn.Embedding(n, d)                 # id -> vector (key/val/query)
        self.mem_ln = nn.LayerNorm(d)
        self.W_base = nn.Parameter(0.01 * torch.randn(d, d))
        self.step_mlp = nn.Sequential(
            nn.Linear(2 * d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )
        self.coda_ln = nn.LayerNorm(d)
        self.coda = nn.Linear(d, n)

    def new_memory(self, B, device):
        return torch.zeros(B, self.d, self.d, device=device)   # delta on top of W_base

    def write(self, delta, keys, vals, lr=0.5, decay=0.1):
        """Write each (key->val) pair into the fast weight (normalized delta rule).
        keys, vals: (B, m) node ids. Returns updated delta (detached)."""
        for j in range(keys.shape[1]):
            k = self.mem_ln(self.node(keys[:, j]))     # (B,d)
            v = self.node(vals[:, j])                  # (B,d)
            W = self.W_base.unsqueeze(0) + delta
            pred = torch.bmm(W, k.unsqueeze(-1)).squeeze(-1)
            err = (pred - v).detach()
            norm = (k * k).sum(-1, keepdim=True).clamp_min(1e-4)
            g = 2.0 * torch.bmm(err.unsqueeze(-1), (k / norm).detach().unsqueeze(1))
            delta = ((1.0 - decay) * delta - lr * g).detach()
        return delta

    def retrieve(self, delta, probe, steps):
        """Latent retrieval of pi(probe) reading memory. Returns per-step
        logits and surprise (entropy)."""
        W = self.W_base.unsqueeze(0) + delta
        s = self.node(probe)                           # (B,d) start from x*
        logits_t, sur_t = [], []
        for _ in range(steps):
            r = torch.bmm(W, self.mem_ln(s).unsqueeze(-1)).squeeze(-1)   # memory read
            s = s + self.step_mlp(torch.cat([s, r], dim=-1))
            logit = self.coda(self.coda_ln(s))
            logp = F.log_softmax(logit, dim=-1)
            ent = -(logp.exp() * logp).sum(-1)          # surprise = uncertainty
            logits_t.append(logit)
            sur_t.append(ent)
        return logits_t, sur_t

    def query(self, delta, batch_q, steps, write=True, lr=0.5, decay=0.1):
        """One query: (optionally) write its pairs into memory, then retrieve.
        batch_q: dict with keys/vals/probe for this query (each (B,...))."""
        if write:
            delta = self.write(delta, batch_q["keys"], batch_q["vals"], lr, decay)
        logits_t, sur_t = self.retrieve(delta, batch_q["probe"], steps)
        return logits_t, sur_t, delta
