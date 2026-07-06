"""Amortization model: same surprise-unified reasoner as awe.models.ttt, but
rollout takes an incoming fast-weight `delta` and returns the updated one, so
a driver can PERSIST the memory across a stream of queries on the same graph.

Persist across queries (ttt) -> memory adapts to the graph -> later queries
should converge (halt) in fewer hops. Reset per episode (new graph).
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
        return self.norm(h)


class Core(nn.Module):
    def __init__(self, d, nhead=4):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(3 * d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )

    def lookup(self, s, H, padmask):
        q = self.ln_q(s).unsqueeze(1)
        ctx, _ = self.attn(q, H, H, key_padding_mask=padmask)
        return ctx.squeeze(1)

    def step(self, s, ctx, mem_read):
        u = self.mlp(torch.cat([s, ctx, mem_read], dim=-1))
        return s + u


class Coda(nn.Module):
    def __init__(self, d, P):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, P)

    def forward(self, s):
        return self.out(self.ln(s))


class AmortReasoner(nn.Module):
    def __init__(self, vocab, P, v0_pos, d=128, nhead=4, nlayers=2, max_len=64):
        super().__init__()
        self.d = d
        self.P = P
        self.v0_pos = v0_pos
        self.prelude = Prelude(vocab, d, nhead, nlayers, max_len)
        self.core = Core(d, nhead)
        self.coda = Coda(d, P)
        self.mem_ln = nn.LayerNorm(d)
        self.W_base = nn.Parameter(0.01 * torch.randn(d, d))

    def embed(self, tokens, padmask):
        H = self.prelude(tokens, padmask)
        return H, H[:, self.v0_pos], padmask

    def rollout(self, H, s0, padmask, steps, delta=None,
                ttt=True, lr_inner=0.5, decay=0.1):
        """One query. `delta` carries the persistent memory in; returns it out.

        delta=None starts from the learned base (memory reset). Passing the
        previous query's delta persists the adaptation (amortization).
        """
        B = s0.shape[0]
        if delta is None:
            delta = torch.zeros(B, self.d, self.d, device=s0.device)
        s = s0
        logits_t, sur_t = [], []
        for _ in range(steps):
            W = self.W_base.unsqueeze(0) + delta
            ctx = self.core.lookup(s, H, padmask)
            s_ln = self.mem_ln(s)
            hat = torch.bmm(W, s_ln.unsqueeze(-1)).squeeze(-1)
            err = hat - ctx.detach()
            sur = (err * err).mean(-1)
            s = self.core.step(s, ctx, hat)
            logits_t.append(self.coda(s))
            sur_t.append(sur)
            if ttt:
                s_d = s_ln.detach()
                norm = (s_d * s_d).sum(-1, keepdim=True).clamp_min(1e-4)
                g = 2.0 * torch.bmm(err.detach().unsqueeze(-1),
                                    (s_d / norm).unsqueeze(1))
                delta = ((1.0 - decay) * delta - lr_inner * g).detach()
        return logits_t, sur_t, delta
