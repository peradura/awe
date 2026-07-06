# External-task design — MQAR transfer of convergence-halting

**Status: design + skeleton (2026-07-06). Not yet run — training needs GPU approval.**
Grounds the roadmap item `PROJECT.md §7` ("scale to an externally legible task").

## Why this experiment, framed honestly

The 10-seed bake-off (`docs/RESULTS.md` §"Part 3-B") gave two separable results.
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

## References (verified 2026-07-06)

Zoology 2312.04927 · Based 2402.18668 · H3 2212.14052 · DeltaNet 2406.06484 ·
Gated DeltaNet 2412.06464 · Mamba-2 2405.21060. In-context linear regression
(follow-up mechanistic probe for ‖Δs‖∝‖∇L‖): Garg 2208.01066 · von Oswald
2212.07677 · Akyürek 2211.15661. Harness: `HazyResearch/zoology`.
