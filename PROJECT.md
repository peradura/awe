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
> vs persist 72.0±0.6%). **Two honesty caveats**: `dstate` preserves accuracy
> largely by being *conservative* (5.93/6 steps — its optimal `tau` barely halts
> early), while the only signal that *also* saves compute is `conv` (5.0±0.6 steps,
> 34% correct early exits — but *seed-fragile*: real saving on 7/10 seeds,
> dstate-like on 3/10) — which is **not** the memory-gradient / write signal.
> So the *strong* thesis (one scalar drives both knobs) is dead; the plainer
> honest reading is that **depth and write want different observables**
> (convergence for depth, reconstruction-miss for write). A *weak, interpretive*
> unification — both are gradients of one memory loss (∇_s L / ∇_W L) — is
> consistent but near-trivial as stated; it earns content only if the halt signal
> is shown to *predict the write magnitude* (untested; the MQAR probe, §7).

Evidence chain (Parts 1–2 single-seed; **3-B is 10 seeds + held-out tau**; reachp2 is 10 seeds but its headline uses no halting, so tau-leakage is moot there — its `+halt`/`both` configs still use scored-batch tau):

| # | claim | task | key number | status |
|---|---|---|---|---|
| 1 | depth tracks difficulty | in-context reachability | `corr(K, halt)=+0.92`, 100% (convergence halting, hindsight; single seed, log not archived) | ✅ |
| 2 | memory buys accuracy + compute | hidden-rule (partial obs) | persist 5→81%; both 2.4 vs 8.0 latent steps; `corr(ans, entropy)=−0.96` | ✅ |
| 3 | joint, *entropy* halt (historical) | partial-obs reachability | amortization holds w/o halting; entropy halting costs accuracy | 🟡 superseded by 3-B |
| 3-B | joint halting **accuracy-preserving** w/ convergence signal (conv saves compute; dstate conservative) | partial-obs reachability, **10-seed** bake-off | conv/dstate −0.0pp vs persist 72.0±0.6%; entropy/recon −5.6pp (early-wrong 8–9%) | ✅ |

A Part-3 `ans`-labeling artifact (probes solvable from current-query reveals
labeled unanswerable) was fixed 2026-07-06; `corr` re-measured after the fix.
Full numbers + the bake-off decomposition: **`docs/RESULTS.md`** §"Part 3-B".

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
│   └── experiments/      # depth_sanity · ablation_{ttt,amort,rule,reachp,reachp2,reachp3}
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

## 7. Roadmap (revised 2026-07-06, post-bake-off)

- [x] Depth-only proof (in-context reachability) — convergence halting.
- [x] Memory-only proof (hidden-rule) — entropy halting + delta-rule writes.
- [x] Joint stress-test (partial-obs reachability) — memory amortization holds;
  entropy halting costs accuracy (later shown signal-specific).
- [x] Fix Part-3 `ans` labeling artifact; re-measure corr.
- [x] Sharpen joint: K-curriculum + aux + d=256 (`ablation_reachp2`, **10 seeds**)
  — persist 72.1±0.6%, which isolates the halting signal as the remaining bottleneck.
- [x] **Halting-signal bake-off** (`ablation_reachp3`, **10 seeds**, held-out tau,
  failure decomposition): entropy/recon lose −5.6pp; conv/dstate −0.0pp (conv also
  saves compute, dstate conservative). **Verdict** — the joint halting cost was a
  *signal choice*; convergence-halting is a good depth controller, but the *strong*
  unification (one scalar drives both optimally; recon drives both) is **dead**, and
  the *weak* "two gradients of one loss" reading is only a cautious interpretation
  (depth & write want different observables). See §4 · `docs/RESULTS.md` 3-B.
- [~] **Scale to an externally legible task** — **first MQAR run done (2026-07-07,
  10 seeds)**: convergence-halting *transfers* (`conv` matches the ceiling at
  2.53/6 steps, ~58% compute saved, no bimodality), but single-hop MQAR does **not**
  discriminate signals (all four are cost-free — its errors are the ~30%
  *unanswerable* probes, not confident-wrong multi-hop; no depth∝difficulty, only
  amortization). See `docs/mqar_design.md` "First-run result". **Queued**: a harder
  MQAR variant (interference ↑ / longer seq / multi-hop) to reproduce reachp's
  signal-discrimination. Baseline table for anchoring: Transformer ceiling, Based
  2402.18668, Mamba-2, DeltaNet 2406.06484, Gated DeltaNet 2412.06464
  (`HazyResearch/zoology` harness). In-context linear regression (Garg 2208.01066 /
  von Oswald 2212.07677) remains a follow-up mechanistic probe for ‖Δs‖∝‖∇L‖.
- **Kill criterion (weak thesis)**: if convergence-halting does not transfer to
  MQAR — best convergence controller loses to fixed-depth at matched compute by
  >5pp across ≥3 seeds — the unification is synthetic-only; pivot to a diagnosis writeup.
- *Fallback (only if a reviewer demands the exact identity)*: make the equivalence
  true by construction (latent step = GD on the memory loss). **Deprioritized** —
  the review found it largely tautological and the write-tie incoherent at the
  RuleReasoner's per-query/per-step granularity (see `docs/REVIEW.md`).

*Research WIP — Dongwan Yoo, DAIS @ KIER, 2026. See LICENSE.*
