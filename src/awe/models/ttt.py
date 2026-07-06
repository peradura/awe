"""Increment 2: surprise-unified latent reasoner.

Adds an episodic fast-weight associative memory W (per query, reset each
example) on top of the Increment-1 reachability reasoner. ONE scalar signal
-- the memory's reconstruction error `surprise` -- drives BOTH knobs:

  A. halting  : stop when surprise is low (the path has settled / converged)
  B. weight   : update W by a graded step (closed-form delta rule) whose size
                grows with surprise (Titans / TTT-layers spirit)

Per latent step t:
  ctx_t   = Core.lookup(s_t, table)              # attend the table = "one hop"
  hat_c   = W_t @ LN(s_t)                         # memory predicts the lookup
  surprise= || hat_c - ctx_t ||^2                # shared signal (scalar / ex.)
  s_{t+1} = Core.step(s_t, ctx_t, hat_c)         # advance, reading the memory
  [B] if ttt:  W_{t+1} = W_t - lr * 2 (hat_c-ctx_t) LN(s_t)^T   # delta rule
  [A] if halt: stop this example once surprise < tau

W_t = W_base + delta_t. delta is detached (first-order / no meta-grad); the
backbone still trains W_base through Core's use of the memory readout, plus a
small auxiliary surprise term so reconstruction is well-calibrated.
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
    """One recurrent hop, split so the fast-weight can sit in between."""

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
        return ctx.squeeze(1)                       # (B,d)

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


class SurpriseReasoner(nn.Module):
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

    def rollout(self, H, s0, padmask, steps, ttt=True, lr_inner=0.5, decay=0.1):
        """Run `steps` hops. Returns per-step logits and surprise (lists).

        No halting here -- halting is applied post-hoc from the surprise trace
        (see awe.experiments.ablation_ttt.halt_from_trace) so a single rollout
        serves every tau.

        The fast-weight write is a NORMALIZED delta rule with a forgetting
        gate: g = 2 (hat-ctx) s_ln^T / ||s_ln||^2, delta <- (1-decay) delta - lr*g.
        Normalization + decay keep the episodic memory from diverging inside
        the loop (the failure mode a raw delta rule hits).
        """
        B = s0.shape[0]
        delta = torch.zeros(B, self.d, self.d, device=s0.device)
        s = s0
        logits_t, sur_t = [], []
        for _ in range(steps):
            W = self.W_base.unsqueeze(0) + delta                # (B,d,d)
            ctx = self.core.lookup(s, H, padmask)               # the hop
            s_ln = self.mem_ln(s)
            hat = torch.bmm(W, s_ln.unsqueeze(-1)).squeeze(-1)   # memory readout
            err = hat - ctx.detach()
            sur = (err * err).mean(-1)                          # (B,) shared signal
            s = self.core.step(s, ctx, hat)                     # advance
            logits_t.append(self.coda(s))
            sur_t.append(sur)
            if ttt:                                             # [B] graded update
                s_d = s_ln.detach()
                norm = (s_d * s_d).sum(-1, keepdim=True).clamp_min(1e-4)  # ||s||^2
                g = 2.0 * torch.bmm(err.detach().unsqueeze(-1),
                                    (s_d / norm).unsqueeze(1))   # (B,d,d) normalized
                delta = ((1.0 - decay) * delta - lr_inner * g).detach()
        return logits_t, sur_t
