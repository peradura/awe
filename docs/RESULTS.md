# Results

## Main claim (calibrated 2026-07-06)

> Each test-time axis works when driven by *a* suitable signal: **depth halting**
> tracks difficulty under prediction-convergence (Part 1), and an **error-gated
> fast-weight memory** buys both accuracy and compute with entropy-based early
> exit (Part 2). **What is NOT yet demonstrated**: (i) a *single shared* signal
> driving both knobs — the positive experiments use two distinct signals
> (write: reconstruction error, halt: readout entropy) — and (ii) joint control
> on a task that needs both knobs: in Part 3, turning halting ON **costs**
> accuracy (−9.6pp).

The evidence is a three-part chain:

| # | claim | task | key number | status |
|---|---|---|---|---|
| 1 | depth halting tracks difficulty | in-context reachability | `corr(K, halt) = +0.92` | ✅ |
| 2 | memory buys accuracy + compute | hidden-rule (partial obs) | persist 5%→81%; both 2.4 vs 8.0 latent steps; `corr(ans, entropy) = −0.96` | ✅ |
| 3 | joint (depth + memory) | partial-obs reachability | halting costs −9.6pp (both 25.7% vs persist 35.3%) | 🔴 joint / 🟡 memory-only |

**Cross-cutting caveats** (apply to every row; queued for the bake-off re-runs):
single seed (`--seed 0`) with no variance estimates; halting threshold `tau`
calibrated on the same batch that is scored (held-out calibration pending);
"compute" = latent retrieval steps only (delta-rule write FLOPs are unaffected
by halting); models are 0.2–0.9M params on synthetic tasks — these are
*mechanism* pilots, not LLM-scale evidence.

---

## Part 1 — Depth-only proof (`experiments/depth_sanity.py`, `models/recurrent.py`)

In-context functional-graph reachability (whole graph given each query). The
recurrent-depth reasoner's **halting tracks difficulty**: accuracy rises with
test-time depth (r=1 → 64%, r≥6 → 100%) and **`corr(K, steps-to-converge) ≈
+0.92`**.

**What the halting signal actually is here**: steps-to-prediction-stability
(first step after which the argmax prediction never changes), computed in
hindsight over the full rollout — i.e. *convergence to the sink fixed-point*,
not a surprise/reconstruction scalar, and not an online rule. It isolates the
**depth knob** and shows the task has the right difficulty structure; it does
not yet test surprise-driven halting. Memory is redundant here (graph fully in
context). *Provenance caveat: single seed; the run log was not archived —
regeneration is queued.*

## Part 2 — Memory-only proof (`datasets/rule.py`, `models/memory.py`, `experiments/ablation_rule.py`)

Hidden permutation π with **partial observation across a query stream** — the
probe is answerable only from memory accumulated in prior queries. This isolates
the **weight knob** and is the strongest result:

| config | accuracy | avg latent steps | accuracy across stream |
|---|---|---|---|
| fixed / +halt (reset) | ~5% | — | flat (chance) |
| persist | 50.1% | 8.0 | **5% → 81%** |
| both (persist+halt) | 50.0% | **2.4** | 5% → 79% |

- **Capability gap**: persistence drives 5% → 81%; reset stays at chance → memory
  across the stream is *necessary*. (Note this is partly by construction — the
  task is designed so only cross-query memory can answer — so it validates the
  harness and the retention/retrieval plumbing rather than being an independent
  discovery.)
- **Amortization**: `both` latent steps fall **7.95 → 1.21** across the stream.
- **Halting signal = readout entropy, not the write signal.** The write is
  gated by the delta-rule reconstruction error; halting thresholds the entropy
  of the answer readout. `corr(answerable, entropy@step0) = −0.96` shows the
  readout is well calibrated (confident exactly when the answer is in memory) —
  a useful sanity check, but *not* evidence for the shared-reconstruction-error
  thesis, which remains untested on this task.
- Net: same accuracy as `persist` at **2.4 vs 8.0 latent steps** (≈3.3× fewer
  *retrieval* steps; write cost is unchanged by halting — FLOPs accounting
  including writes pending).

![hidden-rule](../results/rule_curve.png)

## Part 3 — Joint stress-test (`datasets/reachp.py`, `experiments/ablation_reachp.py`)

Partial-observation reachability tries to combine both knobs: a functional graph
(varying depth K, convergence-halting) revealed only partially and accumulated in
memory. Honest outcome — **negative for joint control; memory-only amortization
holds** (`results/reachp_run.log`):

| config | accuracy | avg steps | across stream |
|---|---|---|---|
| fixed | 21.0% | 10.0 | flat |
| +halt | 16.0% | 3.2 | flat |
| persist | **35.3%** | 10.0 | 22% → 41% |
| both (AWE) | 25.7% | 5.4 | 20% → 27% |

- **Turning halting ON costs accuracy**: `both` 25.7% vs `persist` 35.3%
  (−9.6pp), and `+halt` 16.0% vs `fixed` 21.0% (−5pp). Since `both` halts early
  (5.4 < 10 steps) yet loses to the identical model without halting, a
  substantial fraction of halts are *confident-but-wrong / premature* — the
  entropy scalar reads "low, stop" on examples where more depth over the
  still-filling memory would have converted errors into hits (persist's 22→41
  vs both's 20→27 shows the foregone gains). This is the central open problem.
- Depth *increases* with K but as a step function: halt-step ≈ 2 for K≤1, then
  **jumps toward the budget for K≥2** (8.4 at K=2, 9.6–9.9 for K=3–4, pinned at
  10 for K≥5). The parsimonious reading: the base
  learner cannot resolve K≥2 chains (loss plateaus ~2.27), entropy never falls,
  and the halting signal correctly reports "not done" — i.e. saturation itself
  is not a signal failure, but there is no evidence of *graded* depth either.
- **Labeling artifact found and fixed (2026-07-06)**: `ans` counted only
  strictly-prior reveals while the model writes the *current* query's edges
  before retrieving, so probes solvable from current-query reveals were labeled
  unanswerable (with legitimately low surprise). Re-run with identical seed
  (`results/reachp_run_v2.log`; training/accuracies reproduce exactly, only
  `ans`-dependent statistics change): **corr moves −0.229 → −0.293**. The
  artifact therefore explains only a small part of the weak coupling — even
  with correct labels, entropy tracks answerability far more weakly here than
  on hidden-rule (−0.96), which is a real property of the joint task, not a
  measurement error. (The `fixed` baseline of 21% vs 1/24 ≈ 4% chance remains
  explained by within-query-solvable probes + ~3/24 sink probes + sink priors.)

**Diagnosis (revised)**: three confounded causes, not one — (a) base-learner
limits on chain-following over accumulated memory, (b) the measurement artifact
above, and (c) a genuine halting-rule failure for multi-hop (confident-wrong
early exits). The earlier "not a mechanism failure" wording was unsupported;
(c) *is* a mechanism problem for entropy-threshold halting. Next steps: the
halting-signal bake-off + per-example failure decomposition (early-wrong /
early-right / never-confident), K-curriculum + aux next-node loss + capacity
(`ablation_reachp2`) so learner and controller failures separate.

![partial-obs reachability](../results/reachp_curve_v2.png)

*(Figure regenerated after the labeling fix; the pre-fix artifacts
`reachp_curve.png` / `reachp_run.log` are retained for provenance.)*

## Negative baseline (retained)

Full-table reachability (`experiments/ablation_amort.py`): the whole graph in
every query's context makes memory redundant and random graphs admit no
shortcut, so all configs hit 100% and the amortization curve is flat. Documents
*why* the task must supply reusable structure + a capability gap — the
motivation for Parts 2–3. *Caveat: this script calibrates one `tau` (from the
ttt-on surprise distribution) and applies it to all configs, which is unfair to
`+halt` (different surprise scale — it can fail to ever cross `tau`); later
scripts use per-config tau. The negative verdict stands for task-structure
reasons, but the flat `+halt` curve is partly a tau-scale artifact.*

## Signal inventory (added 2026-07-06 — read this before quoting "surprise")

Different experiments use different scalars; the docs previously blurred them:

| experiment | write gate | halting signal |
|---|---|---|
| Part 1 `depth_sanity` | — (no memory) | prediction-stability (hindsight convergence) |
| `ablation_ttt` / `ablation_amort` | recon error (shared) | recon error (shared) — **the actual thesis; inconclusive/negative tasks** |
| Part 2 `ablation_rule` | recon error | **readout entropy** |
| Part 3 `ablation_reachp` | recon error | **readout entropy** |

The configuration the project's headline describes — one reconstruction-error
scalar driving both knobs on a task with a real capability gap — **has not yet
been run**. That is the top roadmap item (see `PROJECT.md` §7).
