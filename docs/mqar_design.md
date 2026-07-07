# External-task design — MQAR transfer of convergence-halting

**Status: two runs done (2026-07-07, 10 seeds each, GPU1). Single-hop confirms
transfer but doesn't discriminate signals; the multi-hop variant DOES — conv/dstate
beat entropy/recon by ~2.5pp (10/10 seeds) at <half the compute, reproducing
reachp's discrimination on an external associative-recall task.** Grounds `PROJECT.md §7`.

## Why this experiment, framed honestly

The 10-seed bake-off (`docs/RESULTS.md` §"Part 4") gave two separable results.
The design targets the **solid, non-unification** one and probes the speculative one:

- **Solid result to *transfer*** — on the synthetic joint task, a **convergence
  halting signal governs the depth knob well**: `conv` reaches the persist ceiling
  at 5.0/6 steps (−0.0pp accuracy, 34% correct early exits), while entropy/recon
  halt confidently-wrong (−5.6pp). This is a claim about *depth control*, standing
  on its own with **no** reference to weight-sharing. MQAR asks whether it holds on
  a task with a **published, competitive baseline table**.
- **Speculative claim to *probe*, not assume** — the "weak unification" (depth ∝
  ‖∇_s L‖, write ∝ ‖∇_W L‖, same memory loss) is, as written, near-trivial (every
  differentiable loss has both gradients; cross-check flagged this). It becomes a
  *finding* only if the halting signal is empirically shown to **predict the write
  magnitude / memory-loss decrease**. MQAR includes that measurement as a secondary
  probe — a negative result there is the honest verdict "the two knobs want
  different observables," which is fine.

**Primary question (non-unification):** does a convergence-halting controller
transfer to MQAR — matching fixed-depth accuracy at lower average compute, and
beating entropy/recon halting — vs a delta-rule/TTT baseline?

## Task — MQAR (multi-query associative recall)

Standard recall stress test for sub-quadratic / recurrent / SSM models
(Zoology, Arora et al. **arXiv:2312.04927**, ICLR 2024; extended by Based
**arXiv:2402.18668**). Reuse the `HazyResearch/zoology` generator config so numbers
are apples-to-apples with published baselines.

- **Format**: a sequence of key→value bigrams from a random vocabulary, followed by
  **many** interleaved query keys; at each query position emit the value paired with
  that key's first occurrence. (Single-query recall = induction head / H3
  2212.14052; MQAR generalizes to many scattered queries.)
- **Standard config**: vocab 8192; train seq len 256 with 4–64 KV pairs; eval seq
  len 1024 with 4–256 KV pairs (harder generalization). Metric = per-query
  exact-match accuracy.
- **Difficulty axis = number of KV pairs** (memory load). This is the external
  analogue of our reachability `K` — it lets us test **depth ∝ difficulty** halting
  directly (does the controller spend more steps when more pairs must be recalled?).

## Model — reuse the memory reasoner (minimal new code)

MQAR is a near-native fit for `models/memory.py::RuleReasoner`: `write(delta,
keys, vals)` already writes k→v pairs into the persistent fast-weight; `retrieve`
runs the latent think-loop over the read. Adaptation is mostly the dataset + a thin
wrapper, **not** a new architecture (keeps any gain unconfounded, as with reachp2).

- Write the KV pairs into `delta` (one pass), then retrieve each query.
- Baseline anchor: a small **softmax-attention** head (near-100% MQAR ceiling) and,
  as the sub-quadratic comparator, the **delta-rule** read at fixed depth (our own
  memory without halting) — plus, if time permits, cite DeltaNet (2406.06484) /
  Gated DeltaNet (2412.06464) numbers from the shared harness rather than re-run.

## Arms (≥5 seeds, held-out tau, matched avg compute)

Reuse the `ablation_reachp3.py` harness verbatim where possible —
`trace_stream` (per-step preds + all four signals), `apply_halt`, `tau_grid`,
`decompose`, held-out calib/eval split, `aggregate`.

| arm | halting signal | role |
|---|---|---|
| fixed-depth frontier | — (every fixed depth 1..T) | compute/accuracy frontier |
| entropy | readout entropy | status-quo halter |
| **conv** | readout sym-KL convergence | expected transfer winner (efficient) |
| **dstate** | ‖Δs‖² | accuracy-preserving conservative halter |
| recon | memory reconstruction miss | thesis signal (expected loser) |
| random-halt | random, matched avg steps | halting-vs-luck control |

**Metrics**: accuracy vs avg steps (frontier); per-example failure decomposition
(early-wrong / early-right / budget); **depth vs #KV-pairs** (does the controller
spend graded compute with load?); accuracy vs the attention ceiling.

**Secondary unification probe** (the only path to a *non-trivial* unification
claim): per example, correlate the halting signal at halt-step with (a) the
delta-rule **write magnitude** ‖Δdelta‖ for that key and (b) the subsequent
memory-loss decrease. If `dstate`/`recon` predict write magnitude, the "two
gradients of one loss" framing earns empirical content; if not, report the honest
"different observables" reading.

## Kill criterion

If convergence-halting does **not** transfer — the best convergence controller
(conv/dstate) loses to fixed-depth at matched compute by >5pp across ≥3 seeds, or
shows no depth∝load grading — the depth-control result is synthetic-only; pivot to
a diagnosis writeup. (This is the `PROJECT.md §7` weak-thesis kill criterion,
instantiated on MQAR.)

## Deliverables of this phase

- `src/awe/datasets/mqar.py` — Zoology-style MQAR generator (runnable now).
- `src/awe/experiments/ablation_mqar.py` — arms above, reusing the reachp3 harness
  (runnable skeleton; a CPU smoke run validates the pipeline before any GPU run).
- Design doc (this file).

## First-run result (2026-07-07, 10 seeds, GPU1)

Single-hop MQAR (vocab n=64, m=4 KV/query, Q=12, T=6, p_ans=0.7), 6k steps.
Persist ceiling **68.2±0.5%** (≈97% recall of the ~70% answerable probes; the
~30% unanswerable — unseen keys — are wrong at any depth). Halting bake-off:

| halt signal | acc | avg steps | early-wrong | early-right | corr(ans, sig₀) |
|---|---|---|---|---|---|
| `conv`    | 68.3% | **2.53** | 32% | 68% | −0.13 |
| `dstate`  | 68.3% | 4.41 | 32% | 8%  | +0.45 |
| `entropy` | 68.2% | 5.55 | 0%  | 10% | −0.82 |
| `recon`   | 68.2% | 5.37 | 18% | 4%  | −0.38 |

**Read:**
1. ✅ **Convergence-halting transfers.** `conv` matches the ceiling (+0.1pp) at
   **2.53/6 steps (~58% compute saved)** — a *larger* saving than on reachp (5.0
   steps), and robust (no bimodality here). Primary question = **yes**.
2. ⚠️ **This config does not discriminate signals.** *All four* signals are
   accuracy-cost-free (+0.0–0.1pp), unlike reachp (entropy/recon −5.6pp). Reason:
   MQAR's errors are the ~30% **unanswerable** probes (wrong at any depth), so
   halting early on them costs nothing — `conv`'s 32% "early-wrong" is *correctly
   abandoning unsolvable* recalls, not prematurely quitting solvable ones. The
   single-hop task has no confident-wrong-multi-hop failure mode, which is exactly
   what made signal choice matter on reachp.
3. **No depth∝difficulty; amortization instead.** `dstate` halt-step *decreases*
   with memory load (~4.8 at low load → ~3.3 at high) — later-stream recalls are
   faster, not harder. Single-hop MQAR has no per-probe difficulty axis.

**Verdict**: confirms "convergence-halting is a cost-free, compute-saving depth
controller on an external task," but does **not** reproduce reachp's
signal-discrimination. **Queued**: a harder variant (multi-query interference ↑ /
longer sequences / a multi-hop recall chain) to induce the confident-wrong failure
mode and test whether entropy/recon lose there too — run when the GPU is fully free.
Artifacts: `results/mqar_seed{0..9}.json`, `results/mqar_bakeoff.png`.

## Second run — multi-hop MQAR (2026-07-07, 10 seeds, the *discriminating* test)

`datasets/mqar_hop.py` / `experiments/ablation_mqar_hop.py`: H-hop associative
recall (target = g^H(q), H∈1..3 = difficulty), which restores depth∝difficulty so
that halting at the wrong point costs accuracy. 8k steps, p_ans=0.85.
Persist ceiling **41.6±0.9%** (base learner is weak on H-hop chains — like reachp's
K≥2 struggle; ~42% of ≤3-hop probes solved). Fixed-depth accuracy **peaks at depth
3 (41.9%)** then plateaus/slightly drifts to 41.6% at depth 6.

| halt signal | acc | avg steps | gap vs persist | early-wrong | corr(ans, sig₀) |
|---|---|---|---|---|---|
| `conv`    | **42.3±0.9%** | **2.48** | +0.7pp | 58% | +0.30 |
| `dstate`  | 42.2±0.9% | 2.72 | +0.6pp | 58% | +0.21 |
| `entropy` | 39.5±0.8% | 5.40 | −2.0pp | 7%  | −0.53 |
| `recon`   | 40.0±0.9% | 5.40 | −1.6pp | 6%  | −0.44 |

**The discrimination reproduces — cleanly and robustly:**
- **conv/dstate beat entropy/recon by ~2.5pp on *every* seed** (conv−entropy
  +2.76±0.35pp, conv>entropy 10/10; conv−recon +2.32pp, 10/10) **and at less than
  half the compute** (2.5 vs 5.4 steps). Convergence signals win on *both* axes.
- **Mechanism differs from reachp, conclusion is the same.** On reachp, entropy/recon
  lost by halting *confident-wrong early* (−5.6pp) while conv/dstate matched persist.
  Here, entropy/recon lose by *failing to detect convergence* — they drift to the
  budget (5.4/6 steps), past the depth-3 accuracy peak — while conv/dstate stop at
  the peak (~2.5 steps) and so slightly *beat* persist (the +0.6pp is a mild
  overshoot-avoidance effect, not a large win). Either way: **entropy/recon are the
  wrong halting signal; a convergence signal is right.**
- **depth∝difficulty appears** (absent in single-hop): dstate halt-step 2.75 (H=1)
  → 3.45 (H=2) → 3.51 (H=3) — graded, saturating.
- Caveat: 41.6% persist ceiling = weak base learner (mechanism-scale, not a strong
  MQAR result); the robust claim is the *conv-vs-entropy/recon discrimination*, not
  the absolute accuracy. Artifacts: `results/mqarhop_seed{0..9}.json`,
  `results/mqarhop_bakeoff.png`.

**Takeaway across both runs**: convergence-halting (conv/dstate) transfers as a
cost-free, compute-saving depth controller (single-hop) *and* is the discriminating
winner over entropy/recon when the task has real depth structure (multi-hop) — the
reachp finding generalizes to an external associative-recall task.

## Standard-config anchor (backfill design — 2026-07-07, run before executing)

**Purpose** (paper Limitations bullet): mini-MQAR uses vocab 64; the standard zoology
config is vocab 8192. ONE anchor run at standard *scale* so a reviewer can't attribute
the discrimination/transfer to the toy vocab (entropy over 64 classes ≠ entropy over
8192; decodability recon is O(vocab)-dependent).

**Design decisions (no tuning, by rule):**
- Config: vocab **n=8192** (standard), **m=4, Q=16** → ≤64 revealed pairs = the
  standard train config's KV upper end (seq 256 / 4–64 pairs). T=6, p_ans=0.7,
  d=256 (inside zoology's 64–512 model-dim sweep), steps/lr/lam_aux = `ablation_mqar`
  defaults **unchanged** — the anchor inherits the single-hop pilot's hyperparameters
  verbatim; if the base learner is weaker at 8192, that is reported, not tuned away.
- Arms/protocol: identical to the single-hop run (entropy/conv/dstate/recon(state-mism.),
  held-out tau, decomposition, depth-vs-load) — the claim under test is *transfer of
  the halting result to standard scale*, not a leaderboard entry.
- Positioning vs published baselines: cite the zoology table (attention ≈100%;
  sub-quadratic models degrade with KV load) as *context*; our model is a
  delta-rule-family memory, so the relevant published anchors are the DeltaNet-class
  rows. We report our per-query accuracy at matched vocab/load, not a leaderboard claim
  (streaming interface, not their harness).
- 10 seeds, GPU (user-approved 2026-07-07; GPUs idle).
- Implementation note: `datasets/mqar.py::make_batch` is O(n) per example (full dict +
  unseen scan) — prohibitive at n=8192. Anchor path uses a distribution-identical lazy
  generator (lazy dict for D — i.i.d. values on first touch; rejection sampling for
  unseen probes — uniform over unseen since |revealed| ≤ 64 ≪ 8192). Guarded to the
  anchor config so archived mini-MQAR runs stay byte-reproducible.
- Expected outcomes and their readings: (i) conv still matches ceiling at < budget →
  limitation resolved; (ii) base learner too weak at 8192 → report honestly,
  limitation stays but sharpened ("mechanism-scale at standard vocab"); (iii) signals
  reorder → real finding, must go in the paper.
- Artifacts: `results/mqar8k_seed{0..9}.json`, log per seed, aggregate in LOG.md.

## References (verified 2026-07-06)

Zoology 2312.04927 · Based 2402.18668 · H3 2212.14052 · DeltaNet 2406.06484 ·
Gated DeltaNet 2412.06464 · Mamba-2 2405.21060. In-context linear regression
(follow-up mechanistic probe for ‖Δs‖∝‖∇L‖): Garg 2208.01066 · von Oswald
2212.07677 · Akyürek 2211.15661. Harness: `HazyResearch/zoology`.
