# PROJECT — AWE (Adaptive Weight & Exit)

**One surprise signal, two test-time knobs — the hypothesis.**

> Hypothesis: a single `surprise` signal (a fast-weight associative memory's
> self-supervised reconstruction error) can drive, at inference, **both** how
> much to *think* (latent depth halting) **and** how much to *adapt*
> (fast-weight / TTT update). surprise high → keep thinking + update memory
> more. surprise low → halt + stop.
>
> **Status (2026-07-06, post-bake-off).** The *literal* form of this hypothesis
> does **not hold for the halting knob**: in a 10-seed bake-off the
> reconstruction-error scalar *loses* at halting (−5.6pp; with the tested signal
> definition + held-out `tau`). A convergence signal halts without accuracy cost —
> but `dstate` (≈‖∇_s L‖) does so *conservatively* (barely halts early), and the
> compute-efficient halter is readout convergence (`conv`), which has no write
> role. Honest reading: **depth and write want different observables**; the
> "one signal, two knobs" unification is at best a cautious, still-untested
> interpretation. See §4 for the recalibrated claim, §7 for what's next.

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
**Bake-off outcome (§4)**: the convergence reading (`dstate` ≈ ‖Δs‖ ≈ η‖∇_s L‖)
*does* halt without cost — but as the *conservative/safe* member of the family,
while the most compute-efficient halting signal is readout convergence (`conv`),
which is **not** the memory-loss gradient. The "two ways of one signal" identity
is thus directionally supported for `dstate`, but is not the whole story of what
halts best.

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
- The **remaining open cell (narrowed further by the bake-off)**: *the TTT/memory
  module's own dynamics serving as the halting signal* — no learned router,
  Q-head, or RL. No prior work found as of 2026-07-06. The bake-off (§4) ran this:
  the raw reconstruction loss (`recon`) **loses** at halting, but the latent-step
  convergence norm (`dstate` ≈ ‖∇_s L‖) is **accuracy-preserving** (conservatively)
  — so the open cell narrows to *convergence of the inner optimization*, not the
  raw miss, as the depth signal.

See `docs/proposal.md`.

## 4. Main claim (recalibrated 2026-07-06, post 10-seed bake-off)

> Each test-time axis works when driven by *a* suitable signal — depth halting by
> prediction/state **convergence**, fast-weight memory by error-gated delta-rule
> writes. The 10-seed bake-off shows the earlier "joint halting costs accuracy"
> was **not inevitable for every stopping rule**: entropy/recon halting cost −5.6pp
> (confident-wrong early exits), but convergence signals preserve accuracy (−0.0pp
> vs persist 72.0±0.6%). The earlier bimodality/conservatism caveats were
> **artifacts of the argmax-accuracy tau rule and are resolved by the canonical
> run (Part 4c)**: under the fewest-steps-within-slack objective, `conv` saves
> ~35% compute **uniformly** (3.89±0.09/6 steps, 10/10 seeds, ≤1pp cost), and
> only the convergence family admits a within-slack operating point at all
> (conv/dstate 10/10 vs 0/10 for entropy/recon/rnorm/Δ-entropy). `conv` remains
> **not** the memory-gradient / write signal. On the external multi-hop task
> (Part 5) the saving also holds: 2.5/6 steps on 10/10 seeds.
> So the *strong* thesis (one scalar drives both knobs) is dead; the plainer
> honest reading is that **depth and write want different observables**
> (convergence for depth, reconstruction-miss for write). A *weak, interpretive*
> unification — both are gradients of one memory loss (∇_s L / ∇_W L) — is
> consistent but near-trivial as stated; it earns content only if the halt signal
> is shown to *predict the write magnitude* (untested; the MQAR probe, §7).

Evidence chain (Part 1 single-seed; Parts 2/4/5 are 10 seeds; `tau` held-out
everywhere as of 2026-07-07):

| # | claim | task | key number | status |
|---|---|---|---|---|
| 1 | depth tracks difficulty | in-context reachability | `corr(K, halt)=+0.92`, 100% (convergence halting, hindsight; single seed, log not archived) | ✅ |
| 2 | memory buys accuracy + compute | hidden-rule (partial obs) | persist 5→81% (10 seeds: 50.5±0.4% overall); both 2.4 vs 8.0 latent steps; `corr(ans, entropy)=−0.96` | ✅ |
| 3 | joint, *entropy* halt (historical) | partial-obs reachability | amortization holds w/o halting; entropy halting costs accuracy | 🟡 superseded by Part 4 |
| 4a | joint halting **accuracy-preserving** w/ convergence signal | partial-obs reachability (strong learner), **10-seed** bake-off | conv/dstate −0.0pp vs persist 72.0±0.6%; entropy/recon −5.6pp | ✅ |
| 4b | independent bake-off converges (weak learners, stronger nulls, `bakeoff.py`) | rule + original reachp, 10 seeds | reachp: dstate only useful operating point; rule: recon(decodability) ≈ entropy | ✅ |
| 4c | **canonical** (4b protocol + conv × strong learner) — bimodality dissolves | reachp2 via `bakeoff.py`, 10 seeds | conv 71.2±0.5% @ **3.89±0.09** steps, uniform; within-slack tau exists only for conv/dstate (10/10 vs 0/10) | ✅ |
| 5 | convergence-halting **transfers externally** | MQAR single- & multi-hop, 10 seeds each | multi-hop: conv/dstate beat entropy/recon **+2.5pp, 10/10 seeds**, at 2.5 vs 5.4 steps | ✅ |

**Caution when quoting Part 4**: 4a and 4b differ in tau-selection objective
(argmax-accuracy vs fewest-steps-within-slack) and in the `recon` definition
(state-mismatch vs decodability) — **quote 4c (canonical)**; 4a/4b are
history/robustness. A Part-3 `ans`-labeling artifact was fixed 2026-07-06.

Full numbers + figures: **`docs/RESULTS.md`**. Experiment history: **`docs/exp_logs/LOG.md`**.

## 5. Repository layout

```
awe/
├── PROJECT.md            # this file — the whole picture
├── README.md             # quickstart
├── LOGGING.md            # experiment-logging convention
├── pyproject.toml        # installable package (src/ layout)
├── src/awe/
│   ├── datasets/         # reachability · amort · rule · reachp · mqar · mqar_hop
│   ├── models/           # recurrent.py · ttt.py · amort.py · memory.py
│   └── experiments/      # depth_sanity · ablation_{ttt,amort,rule,reachp,reachp2,reachp3,mqar,mqar_hop} · bakeoff.py
├── scripts/              # sweep.sh · aggregate.py · gpu_watch_run.sh (multi-seed + polite shared-GPU runner)
├── docs/
│   ├── proposal.md       # full research proposal (snapshot + dated addendum)
│   ├── RESULTS.md        # results writeup (Parts 1–5 + signal inventory)
│   ├── mqar_design.md    # external-task design + MQAR results
│   ├── REVIEW.md         # 2026-07-06 external review record
│   ├── RUNBOOK.md        # shared-GPU etiquette / how to run sweeps
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

## 7. Roadmap (revised 2026-07-06, post-bake-off)

- [x] Depth-only proof (in-context reachability) — convergence halting.
- [x] Memory-only proof (hidden-rule) — entropy halting + delta-rule writes.
- [x] Joint stress-test (partial-obs reachability) — memory amortization holds;
  entropy halting costs accuracy (later shown signal-specific).
- [x] Fix Part-3 `ans` labeling artifact; re-measure corr.
- [x] Sharpen joint: K-curriculum + aux + d=256 (`ablation_reachp2`, **10 seeds**)
  — persist 72.1±0.6%, which isolates the halting signal as the remaining bottleneck.
- [x] **Halting-signal bake-off ×2** (Part 4a `ablation_reachp3` strong learner,
  10 seeds; Part 4b `bakeoff.py` weak learners w/ stronger nulls + tau-sweep
  curves + infra, 10 seeds — parallel session): entropy/recon lose on joint tasks;
  conv/dstate preserve accuracy; on the memory-only task recon(decodability) ≈
  entropy. **Verdict** — the joint halting cost was a *signal choice*;
  convergence-halting is the right depth controller, but the *strong* unification
  (one scalar drives both optimally) is **dead**, and the *weak* "two gradients of
  one loss" reading is only a cautious interpretation (depth & write want
  different observables). See §4 · `docs/RESULTS.md` Part 4.
- [x] held-out tau everywhere + sweep/aggregate/polite-GPU infra
  (`scripts/`, `docs/RUNBOOK.md`) + reachp2 held-out-tau reproduction
  (corr −0.442 ≈ v2's −0.44 — numbers robust to protocol).
- [x] **Scale to an externally legible task** — **two MQAR runs done (2026-07-07,
  10 seeds each)**. *Single-hop*: convergence-halting **transfers** (`conv` matches
  the ceiling at 2.53/6 steps, ~58% saved, no bimodality) but does **not**
  discriminate signals (all cost-free — errors are unanswerable probes). *Multi-hop*
  (H-hop recall, depth∝difficulty restored): the **discrimination reproduces** —
  `conv`/`dstate` beat `entropy`/`recon` by **+2.5pp on 10/10 seeds** at <half the
  compute; depth grows with hop count. So reachp's "convergence signal is the right
  halter" **generalizes to an external associative-recall task**. Mechanism differs
  (entropy/recon here drift to budget rather than halt confident-wrong-early), same
  conclusion. See `docs/mqar_design.md`. Caveat: 41.6% base-learner ceiling on
  multi-hop chains — a mechanism-scale result.
- [x] **Canonical bake-off** (`bakeoff.py --task reachp2` + `conv`, 10 seeds,
  2026-07-07): **4a's caveats dissolve** — conv saves ~35% uniformly (3.89±0.09
  steps, 10/10), dstate 5.12±0.15; only conv/dstate admit a within-slack tau
  (10/10 vs 0/10 for ent/recon/rnorm/dent); conv +7.4pp above the within-block
  null. `ablation_reachp3` retired to history. (Remaining optional: same protocol
  on MQAR.)
- [ ] **Write the workshop paper NOW** (direction re-adjudicated 2026-07-07 by a
  codex + critic cross-review — both converged): the depth-knob diagnosis is
  already the publishable unit. Story: *"Convergence, not surprise, is the right
  halting observable for TTT-style memory reasoners"* — diagnosis + recipe,
  anchored on the canonical bake-off (Part 4c) + external transfer (Part 5).
  Narrative-lock first; run only experiments the paper needs.
- [ ] **Targeted backfill only** (the two reviewer-rejection gaps):
  (i) Part 1 `depth_sanity` multi-seed + archived logs (currently single-seed,
  log lost — weakest table row); (ii) ONE externally legible anchor — a
  standard-config zoology-style MQAR run (vocab 8192-class, not our mini n=64)
  against published baselines (Based 2402.18668, DeltaNet 2406.06484,
  Gated DeltaNet 2412.06464). Raising the multi-hop mini-MQAR base learner is
  explicitly NOT a goal in itself.
- [ ] *(post-paper, the high-ceiling follow-up — plausibly a second paper)*:
  the write-magnitude question, **re-designed** — the roadmap's earlier passive
  same-example correlation is retired as confounded twice over (difficulty
  common-cause + mechanical read-after-write dependency: retrieving key k reads
  the memory the write to k just produced). The real test moves to the
  in-context-linear-regression setting (Garg 2208.01066 / von Oswald 2212.07677)
  where ∇L_mem is tractable: (a) first verify the unconstrained latent step
  approximates GD at all; (b) then test the halt↔write coupling
  **interventionally** (counterfactually scale a write, difficulty pinned) against
  the GD-predicted *functional form*, and (c) demand **cross-regime invariance**
  of the coupling law. Sign-only correlations license nothing.
- **Kill criterion — scoped to the depth-control thesis, and NOT triggered**:
  convergence-halting did transfer (conv/dstate never lose >5pp to fixed depth at
  matched compute; depth∝difficulty appears on multi-hop MQAR). This is **not** a
  green light for the *unification* program — that bar (a recon/surprise scalar
  driving both knobs) failed on every joint task and stays failed unless the
  write-magnitude probe succeeds.
- *Fallback (only with a risky prediction attached)*: the GD-latent-step
  architecture (latent step = GD on the memory loss). **Deprioritized** — as a
  bare construction it is tautological (‖Δs‖=η‖∇L‖ by definition) and the
  per-query/per-step write-tie is incoherent; it is worth running only framed as
  "does enforcing gradient dynamics *improve* halting transfer / matched-compute
  accuracy vs unconstrained dynamics?" (see `docs/REVIEW.md`).

*Research WIP — Dongwan Yoo, DAIS @ KIER, 2026. See LICENSE.*
