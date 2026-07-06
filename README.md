# AWE — Adaptive Weight & Exit

**One surprise signal, two test-time knobs — the hypothesis under test.**

At inference a model has two ways to spend more compute: think *longer*
(more latent reasoning steps) or adapt its *weights* to the current input
(test-time training). Recent work drives one or the other. **AWE** asks
whether **both can be driven from a single `surprise` signal**:

> surprise ↑ (not yet understood) → keep thinking **and** update weights more
> surprise ↓ (converged)         → halt **and** stop updating

In the target design, `surprise` is the fast-weight memory's self-supervised
reconstruction error, and the same scalar decides **when to Exit** (halting)
and **how much Weight to adapt** (TTT) — hence *Adaptive Weight & Exit*.

> **⚠️ Implementation status (honest accounting).** The shared-signal design is
> implemented in `models/ttt.py` / `models/amort.py`, whose experiments were
> inconclusive (task too easy) / negative. The **positive results below use two
> related but distinct signals**: the delta-rule **reconstruction error gates the
> write**, while **readout entropy drives halting** (`models/memory.py`).
> "One scalar drives both knobs" is therefore still an open hypothesis, not a
> demonstrated result. Closing that gap is the top item on the roadmap.

---

## Why this might be new (narrowed 2026-07-06)

Each half exists separately; nearby cells are fuller than our first sweep found:

| axis | prior work | AWE |
|---|---|:--:|
| adaptive latent **depth / halting** | FR-Ponder (RL), Geiping (KL-convergence), HRM/TRM (learned Q-halting) | via surprise (hypothesis) |
| test-time **weight** adaptation | PonderTTT (binary gate), Titans (memory) | ✅ graded delta-rule |
| **depth–memory trade-off** | UT-Memory (2604.21999): halt steps fall as memory grows — but *train-time capacity*, learned ACT router | test-time accumulation, signal-driven |
| **TTT's own loss as the halting signal** (no learned router / Q-head / RL) | none found (as of 2026-07-06) | the remaining open cell — **not yet implemented in a positive experiment** |

A theoretical hook: Geiping's *convergence* halting ("stop when the latent
stops moving") and Titans/PonderTTT's *surprise* update ("write more when
surprised") may be the **same signal read two ways**. This equivalence holds
only when the latent step is (approximately) gradient descent on the memory
loss; establishing when it holds — and testing it against convergence-halting
and entropy-halting head-to-head — is the open theoretical question, not a
settled premise.

## The mechanism (target design)

```
s = embed(problem)                       # latent thought
W = W_base                               # episodic fast-weight (reset per query)
for t in range(max_steps):
    ctx      = lookup(s, context)        # one reasoning "hop"
    hat      = W @ LN(s)                  # memory predicts the hop
    surprise = mean((hat - ctx)^2)        # the shared signal (MSE)
    s        = step(s, ctx, hat)          # advance the thought
    if TTT:  delta = (1-decay)*delta - lr * dSurprise/dW   # [Weight] graded update
    if EXIT and surprise < tau: break                     # [Exit]  halt
answer = readout(s)
```

This is what `models/ttt.py` implements (with one decision-equivalent detail:
the exit threshold is applied post-hoc from the surprise trace so one rollout
serves every tau — see its docstring): the delta-rule write is normalized
(`/‖s‖²`) with a forgetting gate applied to the episodic `delta` (on top of a
frozen `W_base`), first-order/detached, so the memory does not diverge inside
the loop. **The positive-result model (`models/memory.py`) differs**: it writes
revealed pairs by the same normalized delta rule (error-gated), but halts on
readout **entropy**, not on the reconstruction error — see the status note above.

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
*(Status: this full-context task turned out too easy to separate the configs —
see Status below. H1/H2 remain untested and move to the bake-off protocol.)*

## Quickstart

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .     # installs the `awe` package

python -m awe.experiments.ablation_rule    --steps 3000   # memory-knob pilot (positive)
python -m awe.experiments.depth_sanity     --steps 3000   # depth-knob sanity
python -m awe.experiments.ablation_reachp2 --steps 8000   # sharpened joint (curriculum+aux)
```
(Or without install: `PYTHONPATH=src python -m awe.experiments.ablation_rule`.)
Uses GPU if available, else CPU (models are ~0.2–0.9M params — CPU is fine).

**Full overview → [`PROJECT.md`](PROJECT.md). Results → [`docs/RESULTS.md`](docs/RESULTS.md).
Experiment log → [`docs/exp_logs/LOG.md`](docs/exp_logs/LOG.md).**

## Status

- [x] **Increment 1** — reachability harness validated: accuracy rises with depth,
  `corr(K, steps-to-converge) ≈ +0.92` (hindsight prediction-stability halting;
  single seed; run log not archived — needs regeneration).
- [x] **Increment 2** — shared-signal fast-weight model (`ttt.py`) + 4-way ablation:
  plumbing verified, task too easy to separate configs (inconclusive).
- [x] **Reachability = negative result** — full-table graph makes memory redundant
  (all configs 100%, flat amortization). Motivated the pivot below. See `docs/RESULTS.md`.
- [x] **Hidden-rule pilot ✅ (memory knob positive)** — partial-observation permutation
  stream (`datasets/rule.py` · `models/memory.py` · `experiments/ablation_rule.py`):
  **capability gap** (persist 5%→81%, reset stays at chance), **amortization** (`both`
  latent steps 7.95→1.21 across the stream; write cost unaffected by halting),
  **entropy tracks memory-miss** (`corr = −0.96`). Halting here = readout entropy,
  not the write signal. See `docs/RESULTS.md`.
- [x] **Joint stress-test = negative for joint control** — partial-obs reachability
  (`datasets/reachp.py` · `experiments/ablation_reachp.py`): memory-only amortization
  holds (persist 22%→41%), **but turning halting ON costs accuracy**
  (`both` 25.7% vs `persist` 35.3%, −9.6pp; `+halt` 16.0% vs `fixed` 21.0%) and
  halt-steps jump toward the budget for K≥2 (pinned at 10 by K≥5). An `ans`-labeling artifact attenuating
  `corr` was found and fixed 2026-07-06 (see `docs/RESULTS.md` Part 3).
- [ ] **Top priority — make the headline true or retire it**: run Parts 2–3 with the
  shared reconstruction-error signal actually driving *both* knobs (`ttt.py`-style),
  in a halting-signal bake-off (recon-error vs entropy vs Δstate/step-KL vs
  random-matched-steps vs fixed-depth-matched), held-out tau, ≥5 seeds.
- [ ] Sharpen joint: K-curriculum + auxiliary next-node loss + easier config / more
  capacity (`ablation_reachp2`), plus per-example failure decomposition
  (early-wrong / early-right / never-confident).

## Layout

```
src/awe/
├── datasets/   reachability.py · amort.py · rule.py · reachp.py   (task generators)
├── models/     recurrent.py · ttt.py · amort.py · memory.py       (reasoners)
└── experiments/ depth_sanity · ablation_{ttt,amort,rule,reachp,reachp2}
docs/   proposal.md · RESULTS.md · REVIEW.md · exp_logs/LOG.md
results/  figures + run logs
```
See [`PROJECT.md`](PROJECT.md) for the full narrative and file roles.

## Known limitations (as of 2026-07-06)

- **Single seed everywhere.** All headline numbers are one training run at
  `--seed 0`, no error bars. Multi-seed reruns are queued with the bake-off.
- **tau calibration**: fixed 2026-07-06 — `calib_tau` now uses a held-out
  calibration batch in all experiments. The archived seed-0 logs predate the
  fix; re-reported numbers come from the multi-seed sweep (`scripts/sweep.sh`).
- **"3.3× less compute" counts latent retrieval steps only.** Delta-rule write
  cost is unaffected by halting; FLOPs accounting including writes is pending.
- **`ablation_amort` used one tau across configs** (unfair to `+halt`, whose
  surprise scale differs); later scripts use per-config tau. Its negative verdict
  stands for task-structure reasons, but the flat `+halt` curve is partly a
  tau-scale artifact.
- **Scale.** 0.2–0.9M-param models on synthetic tasks support *mechanism* claims
  only; nothing here yet supports claims about LLM-scale test-time compute.

## Related work

Coconut (arXiv:2412.06769) · FR-Ponder (2509.24238) · Recurrent Depth
(2502.05171) · PonderTTT (2601.00894, Jan 2026) · Titans (2501.00663) · TTT
layers (2407.04620) · PonderNet (2107.05407) · **UT-Memory (2604.21999)** —
depth–memory substitution under ACT, train-time capacity + learned router ·
**HRM (2506.21734)** / **TRM (2510.04871)** — tiny recursive models, learned
Q-halting + persistent state on synthetic reasoning. See `docs/proposal.md`
for how AWE sits among them (and what 2604.21999 does *not* cover).

---

*Research WIP — Dongwan Yoo, DAIS @ KIER, 2026. Unpublished; see LICENSE.*
