# PROJECT — AWE (Adaptive Weight & Exit)

**One surprise signal, two test-time knobs — the hypothesis.**

> Hypothesis: a single `surprise` signal (a fast-weight associative memory's
> self-supervised reconstruction error) can drive, at inference, **both** how
> much to *think* (latent depth halting) **and** how much to *adapt*
> (fast-weight / TTT update). surprise high → keep thinking + update memory
> more. surprise low → halt + stop.
>
> **Not yet demonstrated.** The positive experiments so far drive the two knobs
> with two related but distinct signals (write: reconstruction error; halt:
> readout entropy). See §4 for the calibrated claim and §7 for the plan to
> close the gap.

---

## 1. Motivation

At inference an LLM can spend more compute two ways: **think longer** (more
latent reasoning steps) or **adapt its weights** to the current input
(test-time training). Recent work drives one *or* the other. AWE asks whether a
**single signal** can drive **both** — and reframes the value proposition as:

> *"surprise detects a memory miss; as memory accumulates, required depth shrinks."*

## 2. The two knobs + the unification

| knob | question | prior art |
|---|---|---|
| **A. depth (Exit)** | how many latent steps? | ACT, PonderNet, Coconut, Geiping (recurrent depth) |
| **B. weight (Weight)** | how much to adapt fast-weights? | TTT, TTT-layers, Titans, PonderTTT |

A theoretical hook: Geiping's *convergence* halting ("stop when the latent
stops moving") and Titans/PonderTTT's *surprise* update ("write more when
surprised") **may be the same signal read two ways** — but the identity holds
only when the latent step approximates gradient descent on the memory loss.
Treating this as a hypothesis to test (not a premise) is part of the project.

## 3. Literature positioning (verified 2026-07-03; revised 2026-07-06)

Building blocks (arXiv, all verified): Coconut 2412.06769 · FR-Ponder 2509.24238
· Recurrent Depth 2502.05171 · PonderTTT 2601.00894 (Jan 2026) · Titans
2501.00663 · TTT-layers 2407.04620 · PonderNet 2107.05407.

**2026-07-06 revision — the neighborhood is fuller than the first sweep found:**

- **UT-Memory 2604.21999** (Apr 2026) shows a depth–memory trade-off (mean halt
  steps 11.6→8.3 as memory grows, fixed accuracy, Sudoku-Extreme). *But*: its
  memory is a train-time architectural capacity (separately trained models per
  memory size) and its halting is a learned ACT router. It does **not** cover
  test-time memory accumulation or signal-driven halting — it must be cited and
  differentiated, and the earlier claim "nobody has compared the two knobs on
  the same budget line" is retired.
- **HRM 2506.21734 · TRM 2510.04871**: tiny recursive models with learned
  Q-halting + persistent state on synthetic reasoning — the same experimental
  niche, much stronger task results; also to be cited.
- The **remaining open cell (narrowed)**: *the TTT/memory module's own
  self-supervised loss serving as the halting signal* — no learned router,
  Q-head, or RL. No prior work found as of 2026-07-06. This is also exactly
  the configuration our positive experiments have **not yet run** (§4).

See `docs/proposal.md`.

## 4. Main claim (calibrated 2026-07-06)

> Each test-time axis works when driven by *a* suitable signal — depth halting
> by prediction-convergence (Part 1), fast-weight memory by error-gated
> delta-rule writes with entropy-based early exit (Part 2). **A single shared
> signal driving both, and joint control on a task that needs both, are not yet
> demonstrated** — in the joint task, turning halting on currently *costs*
> accuracy.

Evidence is a three-part chain:

| # | claim | task | key number | status |
|---|---|---|---|---|
| 1 | depth tracks difficulty | in-context reachability | `corr(K, halt)=+0.92`, 100% (convergence halting, hindsight; single seed, log not archived) | ✅ |
| 2 | memory buys accuracy + compute | hidden-rule (partial obs) | persist 5→81% (both 5→79%); both 2.4 vs 8.0 latent steps; `corr(ans, entropy)=−0.96` | ✅ |
| 3 | joint (depth + memory) | partial-obs reachability | amortization holds w/o halting (22→41%), **halting costs −9.6pp** (both 25.7% vs persist 35.3%); depth saturates K≥2 | 🔴 joint / 🟡 memory-only |

Caveats that apply to all rows: single seed, tau calibrated on the eval batch,
compute counted in latent steps (write cost excluded). A Part-3 `ans`-labeling
artifact (probes solvable from current-query reveals labeled unanswerable) was
fixed 2026-07-06; `corr` re-measured after the fix (see `docs/RESULTS.md`).

Full numbers + figures: **`docs/RESULTS.md`**. Experiment history: **`docs/exp_logs/LOG.md`**.

## 5. Repository layout

```
awe/
├── PROJECT.md            # this file — the whole picture
├── README.md             # quickstart
├── LOGGING.md            # experiment-logging convention
├── pyproject.toml        # installable package (src/ layout)
├── src/awe/
│   ├── datasets/         # reachability.py · amort.py · rule.py · reachp.py
│   ├── models/           # recurrent.py · ttt.py · amort.py · memory.py
│   └── experiments/      # depth_sanity · ablation_{ttt,amort,rule,reachp,reachp2}
├── docs/
│   ├── proposal.md       # full research proposal (snapshot + dated addendum)
│   ├── RESULTS.md        # results writeup (3-part + signal inventory)
│   ├── REVIEW.md         # 2026-07-06 external review record
│   └── exp_logs/LOG.md   # dated experiment index
└── results/              # figures + run logs
```

## 6. How to run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .        # installs the `awe` package
python -m awe.experiments.ablation_rule   --steps 3000   # memory-knob pilot (positive)
python -m awe.experiments.depth_sanity    --steps 3000   # depth-knob sanity
python -m awe.experiments.ablation_reachp2 --steps 8000  # sharpened joint (curriculum+aux)
```
(Or without install: `PYTHONPATH=src python -m awe.experiments.ablation_rule`.)
Uses GPU if available, else CPU (models are ~0.2–0.9M params — CPU is fine).

## 7. Roadmap (revised 2026-07-06)

- [x] Depth-only proof (in-context reachability) — convergence halting.
- [x] Memory-only proof (hidden-rule) — entropy halting + delta-rule writes.
- [x] Joint stress-test (partial-obs reachability) — **negative for joint control**
  (halting costs −9.6pp); memory-only amortization holds.
- [x] Fix Part-3 `ans` labeling artifact; re-measure corr (2026-07-06).
- [x] **Halting-signal bake-off — DONE, verdict in** (`docs/RESULTS.md` Part 4;
  two independent runs × 5 seeds: `experiments/bakeoff.py` on the weak-learner
  tasks, `experiments/ablation_reachp3.py` on the strong learner):
  **convergence-family signals (readout convergence / Δstate) make halting
  free** (71.9±0.3% = persist ceiling @ 5.18 steps; early-wrong 2%); entropy
  and recon thresholds fail on joint tasks; recon ≈ entropy on the memory-only
  task. Part 3's −9.6pp reinterpreted as tau-calibration artifact +
  signal-choice error.
- [x] Sharpen joint (`ablation_reachp2`): base-learner bottleneck fixed
  (persist 35→72%); numbers robust across protocols (median vs held-out tau,
  pre/post ans-fix: corr −0.44 twice independently).
- **Kill criterion: NOT triggered.** The literal "low recon-error → halt" is
  refuted on joint tasks, but the convergence reading wins everywhere — the
  thesis pivots, not dies: *halt when surprise stops moving the state, write
  in proportion to surprise* (Δs ∝ ∇L).
- [ ] **Next (theory-to-architecture)**: make the equivalence structural —
  latent step = gradient descent on the memory loss, so convergence-halting
  *provably equals* vanishing-surprise-gradient halting; then the single
  scalar drives both knobs by construction.
- [ ] Scale to externally legible tasks (MQAR / in-context regression) vs TTT
  baselines and learned-halting (Q-head/ACT) baselines — the two-signal
  comparison the bake-off did not yet include.
- [ ] Housekeeping: regenerate `depth_sanity` log; FLOPs accounting including
  write cost; pin dependency versions.

*Research WIP — Dongwan Yoo, DAIS @ KIER, 2026. See LICENSE.*
