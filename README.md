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

> **⚠️ Status (2026-07-06, post 10-seed bake-off).** The *literal* "one
> reconstruction-error scalar drives both knobs" claim is **tested and does not
> hold for the halting knob**: the reconstruction scalar *loses* (−5.6pp, on par
> with entropy; with the tested definition + held-out `tau`). A convergence signal
> halts without accuracy cost — but `dstate` (≈‖∇_s L‖) does so *conservatively*
> (barely halts early), and the compute-efficient halter is readout convergence
> (`conv`), which has no write role. Honest reading: **depth and write want
> different observables**; "one signal, two knobs" is at best a cautious, untested
> interpretation. Next: does convergence-halting transfer to an external task
> (MQAR)? See [`docs/RESULTS.md`](docs/RESULTS.md) §"Part 4", `PROJECT.md` §4/§7,
> and [`docs/mqar_design.md`](docs/mqar_design.md).

---

## Why this might be new (narrowed 2026-07-06)

Each half exists separately; nearby cells are fuller than our first sweep found:

| axis | prior work | AWE |
|---|---|:--:|
| adaptive latent **depth / halting** | FR-Ponder (RL), Geiping (KL-convergence), HRM/TRM (learned Q-halting) | via surprise (hypothesis) |
| test-time **weight** adaptation | PonderTTT (binary gate), Titans (memory) | ✅ graded delta-rule |
| **depth–memory trade-off** | UT-Memory (2604.21999): halt steps fall as memory grows — but *train-time capacity*, learned ACT router | test-time accumulation, signal-driven |
| **TTT-module dynamics as the halting signal** (no learned router / Q-head / RL) | none found (as of 2026-07-06) | open cell — bake-off: raw recon loss **loses**, but latent-step *convergence* (`dstate`≈‖∇_s L‖) **preserves accuracy** (conservatively) |

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
*(Status: this full-context task turned out too easy to separate the configs.
H1/H2 were later tested on the partial-obs joint task via the bake-off: **H1** —
partially: a convergence signal halts earlier at matched accuracy (`conv` ~4.6 vs
6.0 steps, −0.0pp) **on 7/10 seeds** (bimodal — no saving on 3/10); **H2** — the
accuracy/compute frontier depends on the halting
signal (entropy/recon −5.6pp, conv/dstate −0.0pp). See `docs/RESULTS.md`
§"Part 4".)*

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
- [x] **Joint stress-test (entropy halt) = negative, later resolved** — partial-obs
  reachability (`datasets/reachp.py` · `experiments/ablation_reachp.py`): memory
  amortization holds, but *entropy* halting cost accuracy. An `ans`-labeling artifact
  attenuating `corr` was found and fixed 2026-07-06. **Superseded by the bake-off**
  (the cost was signal-specific).
- [x] **Sharpen joint (`ablation_reachp2`, 10 seeds)** — K-curriculum + aux next-node
  loss + d=256: persist **72.1±0.6%** (reset 43.3% → persist), isolating the halting
  signal as the bottleneck; multi-seed amortization / depth∝K figures with error bars.
- [x] **Halting-signal bake-off ×2 (10 seeds each) ✅ verdict** — (4a) strong
  learner, `ablation_reachp3`: entropy/recon lose −5.6pp (early-wrong 8–9%),
  `conv`/`dstate` −0.0pp vs persist 72.0±0.6%; (4b) independent implementation on
  the weak learners with stronger nulls + tau-sweep curves, `bakeoff.py`: dstate
  the only useful joint operating point, and recon(decodability) ≈ entropy on the
  memory-only task. The literal recon-scalar thesis **fails at halting on joint
  tasks**. ⚠️ 4a/4b use different tau rules + recon definitions — see
  `docs/RESULTS.md` Part 4.
- [x] **External task — MQAR (2026-07-07, 10 seeds each)**: convergence-halting
  *transfers* (single-hop: `conv` matches ceiling at 2.5/6 steps) and is the
  *discriminating winner* when the task has depth structure (multi-hop: conv/dstate
  beat entropy/recon **+2.5pp, 10/10 seeds**, at half the compute). reachp's finding
  generalizes. See `docs/mqar_design.md`.
- [x] **Canonical bake-off (`bakeoff.py --task reachp2` + `conv`, 10 seeds) ✅** —
  4a's bimodality caveat **was** a tau-rule artifact and dissolves: conv saves ~35%
  compute uniformly (3.89±0.09/6 steps, 10/10 seeds, ≤1pp cost); only conv/dstate
  admit a within-slack operating point (10/10 vs 0/10 for the rest). This is the
  canonical joint-task table (`docs/RESULTS.md` Part 4c).
- [ ] **Next**: write-magnitude probe (the non-tautological unification test);
  raise the multi-hop MQAR base learner + anchor vs a published baseline
  (Based 2402.18668 / DeltaNet 2406.06484, Zoology harness).

## Layout

```
src/awe/
├── datasets/   reachability · amort · rule · reachp · mqar · mqar_hop  (task generators)
├── models/     recurrent.py · ttt.py · amort.py · memory.py            (reasoners)
└── experiments/ depth_sanity · ablation_{ttt,amort,rule,reachp,reachp2,reachp3,mqar,mqar_hop} · bakeoff.py
scripts/  sweep.sh · aggregate.py · gpu_watch_run.sh  (multi-seed + polite shared-GPU runner)
docs/   proposal.md · RESULTS.md · mqar_design.md · REVIEW.md · RUNBOOK.md · exp_logs/LOG.md
results/  figures + run logs
```
See [`PROJECT.md`](PROJECT.md) for the full narrative and file roles.

## Known limitations (as of 2026-07-07)

- **Seeds differ by result.** The bake-offs (Part 4a/4b), reachp2, rule, and both
  MQAR runs are **10 seeds** with error bars; Part 1 (depth sanity) is still
  single-seed (`--seed 0`) — rerun pending.
- **tau calibration**: held-out everywhere as of 2026-07-07 — `bakeoff.py` and
  `ablation_reachp3` always were; `calib_tau` in the rule/reachp/reachp2 ablations
  now uses a held-out batch (archived seed-0 logs predate the fix; a held-out rerun
  reproduced the reachp2 numbers). Note 4a and 4b use **different tau objectives**
  (argmax-accuracy vs fewest-steps-within-slack) — don't merge their tables.
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
