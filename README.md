# AWE — Adaptive Weight & Exit

**One surprise signal, two test-time knobs.**

At inference an LLM has two ways to spend more compute: think *longer*
(more latent reasoning steps) or adapt its *weights* to the current input
(test-time training). Recent work drives one or the other. **AWE** drives
**both from a single `surprise` signal**:

> surprise ↑ (not yet understood) → keep thinking **and** update weights more
> surprise ↓ (converged)         → halt **and** stop updating

`surprise` is the fast-weight memory's self-supervised reconstruction error.
The same scalar decides **when to Exit** (halting) and **how much Weight to
adapt** (TTT) — hence *Adaptive Weight & Exit*.

---

## Why this might be new

Each half already exists; unifying them under one signal does not.

| axis | prior work | AWE |
|---|---|:--:|
| adaptive latent **depth / halting** | FR-Ponder (RL), Geiping (KL-convergence) | ✅ via surprise |
| test-time **weight** adaptation | PonderTTT (binary gate), Titans (memory) | ✅ graded |
| **one shared signal → both** | — none — | ✅ |

A theoretical hook that falls out of the reading: Geiping's *convergence*
halting ("stop when the latent stops moving") and Titans/PonderTTT's
*surprise* update ("write more when surprised") are the **same signal read two
ways**. AWE unifies them.

## The mechanism

```
s = embed(problem)                       # latent thought
W = W_base                               # episodic fast-weight (reset per query)
for t in range(max_steps):
    ctx      = lookup(s, context)        # one reasoning "hop"
    hat      = W @ LN(s)                  # memory predicts the hop
    surprise = || hat - ctx ||^2          # the shared signal
    s        = step(s, ctx, hat)          # advance the thought
    if TTT:  W = (1-decay)*W - lr * dSurprise/dW      # [Weight] graded update
    if EXIT and surprise < tau: break                # [Exit]  halt
answer = readout(s)
```

The delta-rule write is normalized (`/‖s‖²`) with a forgetting gate so the
episodic memory does not diverge inside the loop.

## The experiment

A tiny, fully controllable pilot: **functional-graph reachability** (follow a
per-example graph to its sink; difficulty *K* = hops needed). One trained
model, evaluated four ways:

| config | halting (Exit) | TTT (Weight) |
|---|:--:|:--:|
| `fixed` | ✗ | ✗ |
| `+halt` | ✅ | ✗ |
| `+ttt` | ✗ | ✅ |
| `both` (AWE) | ✅ | ✅ |

**Questions.** *H1*: does TTT let AWE halt earlier at matched accuracy?
*H2*: is `both` the best accuracy-per-compute frontier?

## Quickstart

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt   # Windows: .venv\Scripts\python.exe

python train_eval.py --steps 3000     # Increment 1: harness sanity
python ablation.py  --steps 4000      # Increment 2: 4-way ablation -> accuracy_vs_steps.png
```
Uses GPU if available, else CPU (the model is ~0.75M params — CPU is fine).

## Status

- [x] **Increment 1** — harness validated: accuracy rises with depth
  (r=1 → 64%, r≥6 → 100%); `corr(K, steps-to-converge) ≈ +0.92`
  (convergence tracks difficulty — the halting signal works).
- [x] **Increment 2** — surprise-unified model + 4-way ablation
  (`data.py` · `model_ttt.py` · `ablation.py`).
- [ ] H1 / H2 frontier analysis; harder-K regime; domain transfer.

## Files

| file | role |
|---|---|
| `data.py` | functional-graph reachability generator (difficulty = hops) |
| `model.py` / `train_eval.py` | Increment 1 recurrent-depth reasoner |
| `model_ttt.py` | Increment 2 surprise-unified fast-weight reasoner |
| `ablation.py` | 4-way ablation + frontier figure |
| `proposal.md` | full research proposal (motivation, gap, hypotheses, roadmap) |

## Related work

Coconut (arXiv:2412.06769) · FR-Ponder (2509.24238) · Recurrent Depth
(2502.05171) · PonderTTT (2601.00894) · Titans (2501.00663) · TTT layers
(2407.04620) · PonderNet (2107.05407). See `proposal.md` for how AWE sits
among them.

---

*Research WIP — Dongwan Yoo, DAIS @ KIER, 2026. Unpublished; see LICENSE.*
