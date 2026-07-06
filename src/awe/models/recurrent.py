"""Latent recurrent reasoner for functional-graph reachability.

  H     = Prelude(tokens)            # per-token encoding of table + start
  s0    = H[:, v0_pos]               # current-node latent = start node
  s_i   = s_{i-1} + Core(s_{i-1}, H) # one hop: cross-attend the table, step
  y_i   = Coda(s_i)                  # predicted current node after i hops

Core cross-attends the encoded table each step, so "apply f once" = look up
the successor of the current node. Running more steps = following more hops;
once the sink (fixed point) is reached the readout stops changing.

Surprise proxies exposed for later increments:
  - entropy(y_t)                : answer-distribution uncertainty
  - step_kl(y_t, y_{t-1})       : how much the readout still moves (Geiping)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Prelude(nn.Module):
    def __init__(self, vocab, d, nhead=4, nlayers=2, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d, nhead, dim_feedforward=4 * d, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens, padmask):
        L = tokens.shape[1]
        pos = torch.arange(L, device=tokens.device).unsqueeze(0)
        h = self.tok(tokens) + self.pos(pos)
        h = self.enc(h, src_key_padding_mask=padmask)
        return self.norm(h)                     # (B,L,d)


class Core(nn.Module):
    """One recurrent hop: cross-attend the table from the current node latent."""

    def __init__(self, d, nhead=4):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )

    def forward(self, s, H, padmask):
        q = self.ln_q(s).unsqueeze(1)                       # (B,1,d)
        ctx, _ = self.attn(q, H, H, key_padding_mask=padmask)
        u = self.mlp(torch.cat([s, ctx.squeeze(1)], dim=-1))
        return s + u


class Coda(nn.Module):
    def __init__(self, d, P):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, P)

    def forward(self, s):
        return self.out(self.ln(s))


class Reasoner(nn.Module):
    def __init__(self, vocab, P, v0_pos, d=128, nhead=4, nlayers=2, max_len=64):
        super().__init__()
        self.P = P
        self.v0_pos = v0_pos
        self.prelude = Prelude(vocab, d, nhead, nlayers, max_len)
        self.core = Core(d, nhead)
        self.coda = Coda(d, P)

    def embed(self, tokens, padmask):
        H = self.prelude(tokens, padmask)
        s0 = H[:, self.v0_pos]
        return H, s0, padmask

    def run(self, H, s0, padmask, steps, return_traj=False):
        s = s0
        traj = []
        for _ in range(steps):
            s = self.core(s, H, padmask)
            if return_traj:
                traj.append(self.coda(s))
        logits = self.coda(s)
        return (logits, s, traj) if return_traj else (logits, s)


def entropy(logits):
    logp = F.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(-1)


def step_kl(cur, prev):
    """KL(softmax(cur) || softmax(prev)) per example."""
    lp_c = F.log_softmax(cur, dim=-1)
    lp_p = F.log_softmax(prev, dim=-1)
    return (lp_c.exp() * (lp_c - lp_p)).sum(-1)
