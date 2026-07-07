# Convergence, Not Surprise: Diagnosing the Right Halting Signal for Fast-Weight Memory Reasoners

*Workshop paper draft skeleton — v0 (2026-07-07). Target: mechanistic-interpretability
/ efficient-inference workshop (4–5 pages + appendix). All numbers below are final,
10-seed, held-out-tau results already in `docs/RESULTS.md`; pointers in [brackets].*

**Working titles (pick one):**
1. Convergence, Not Surprise: Diagnosing the Right Halting Signal for Fast-Weight Memory Reasoners
2. When Should a Test-Time Memory Stop Thinking? A Halting-Signal Bake-off
3. Halting Signals for TTT-Style Memory Reasoners: Surprise Loses, Convergence Wins

---

## Abstract (draft)

Test-time compute can be spent two ways: thinking longer (latent depth) or adapting
weights (test-time training). A natural hypothesis — implicit in Titans-style
surprise writing and PonderTTT-style gating — is that one *surprise* signal (the
memory's self-supervised reconstruction error) should govern both: keep thinking and
keep writing while surprised, stop when not. **We test this directly and find it
fails for the halting half.** On a recurrent-depth reasoner with a delta-rule
fast-weight memory, we race six candidate halting signals under a rigorous protocol
(held-out threshold calibration, within-block shuffle nulls, tau-sweep matched-compute
curves, 10 seeds): reconstruction-error and entropy thresholds lose 3–6pp of accuracy
or refuse to halt, while **convergence observables** (readout distribution
convergence; latent step norm) are the *only* signals that admit an operating point
within 1pp of the no-halt ceiling — reaching it at ~65% of the compute, uniformly
across seeds. The discrimination reproduces on an external multi-hop associative
recall task (+2.5pp over entropy/recon on 10/10 seeds at less than half the compute),
where halting depth also grows with hop count. The diagnosis is clean: the write gate
wants a reconstruction-miss observable, but the depth knob wants a *convergence*
observable — one signal does not serve both. We release the bake-off protocol as a
recipe for evaluating halting signals in test-time-adaptive models.

*(~180 words; trim to venue limit.)*

## 1. Introduction

- Two knobs of test-time compute: depth (ACT/PonderNet/Coconut/recurrent-depth) vs
  weights (TTT/TTT-layers/Titans). Recent work drives one or the other.
- The seductive unification: "surprise high → think more + write more; surprise low →
  halt + stop." Geiping's convergence-halting and Titans' surprise-writing *look like*
  two readings of one signal. Nobody had tested whether one scalar actually serves both.
- **Contributions:**
  1. A rigorous halting-signal bake-off protocol (held-out slack-based tau,
     within-block shuffle null, tau-sweep matched-compute curves, per-example failure
     decomposition) for test-time-adaptive reasoners. [experiments/bakeoff.py]
  2. A clean negative + positive diagnosis: surprise/entropy thresholds fail as
     halting rules on joint (memory + multi-hop) tasks; convergence observables are
     the only signals admitting a near-ceiling operating point — at ~65% compute. [Part 4c]
  3. External reproduction on multi-hop associative recall (+ depth∝difficulty
     grading), plus the memory-only counterweight where a reconstruction-family
     signal *does* match entropy. [Part 5, Part 4b]
- Honest framing up front: we set out to *confirm* the one-signal thesis and obtained
  the opposite; the paper is the diagnosis + recipe.

## 2. Related work (compressed; full table in appendix)

- Depth: ACT, PonderNet 2107.05407, Coconut 2412.06769, FR-Ponder 2509.24238,
  recurrent depth 2502.05171 (KL-convergence halting), HRM 2506.21734 / TRM
  2510.04871 (learned Q-halting on synthetic reasoning).
- Weights: TTT 1909.13231, TTT-layers 2407.04620, Titans 2501.00663 (surprise-gated
  write), PonderTTT 2601.00894 (binary gate).
- Depth–memory interaction: UT-Memory 2604.21999 (train-time capacity + ACT router —
  differentiate: ours is test-time accumulation + signal-driven halting, no router).
- Recall stress tests: Zoology 2312.04927, Based 2402.18668, DeltaNet 2406.06484,
  Gated DeltaNet 2412.06464 (context for the MQAR-style transfer task).
- Positioning: the open cell = the memory module's *own dynamics* as the halting
  signal (no learned router/Q-head/RL). Cite predictive coding lineage (Rao &
  Ballard 1999; Friston) for the "one error drives both settling and plasticity" idea.

## 3. Setup

### 3.1 Model
RuleReasoner [src/awe/models/memory.py]: node embeddings; persistent fast-weight
delta on top of W_base written by a normalized delta rule (write strength ∝
reconstruction miss); recurrent retrieval loop s_{t+1} = s_t + MLP([s_t, W·LN(s_t)]);
readout head. 0.86M params (d=256). Streaming episodes: Q=12 queries/episode, m
key→value pairs revealed per query, memory persists across the stream.

### 3.2 Tasks
- **Joint task (primary)**: partial-observation reachability [datasets/reachp.py] —
  functional graph revealed incrementally; probe = "which sink does v0 reach?";
  difficulty K = path length; needs memory (cross-query) *and* depth (multi-hop
  chase). Base learner fixed by K-curriculum + aux next-node loss (persist ceiling
  72%; without the fix 35% — reviewer note: controller verdicts require a competent
  base learner).
- **Memory-only control**: hidden-permutation rule task [datasets/rule.py].
- **External transfer**: single-hop and multi-hop (H∈1..3) associative recall in the
  MQAR family [datasets/mqar.py, mqar_hop.py]. *(Backfill: one standard-config
  zoology run — see TODO.)*

### 3.3 Candidate halting signals (halt when signal < τ)
ent (readout entropy) · recon (two forms — state-mismatch ‖n(s_{t+1})−n(r_t)‖² and
vocab-decodability min_v‖r_t−node(v)‖²; both reported, never conflated) · rnorm
(−‖r‖²) · dstate (‖Δs‖²) · conv (sym-KL(p_t, p_{t−1})) · dent (|Δent|).
Note which are *convergence-family* (dstate, conv, dent) vs *magnitude-family*
(ent, recon, rnorm).

### 3.4 Protocol (the recipe — box figure)
1. One no-halt rollout per query → all signals traced post-hoc (exact: retrieval
   never mutates memory).
2. τ per signal on a **held-out calibration stream**: fewest average steps within a
   1pp accuracy slack of the no-halt calibration accuracy (fallback: max calib
   accuracy, flagged). *[Ablation: the earlier argmax-accuracy rule produced
   spurious seed-bimodality — Appendix C.]*
3. Controls: fixed-depth frontier (every depth 1..T); random halting matched on
   steps; **within-block shuffle null** (permute halt steps within each query index —
   beating it requires per-example information).
4. Report: acc vs avg-steps operating point + full tau-sweep curve; per-example
   failure decomposition (early-right / premature / wrong-anyway / full-depth);
   corr(answerable, signal).
5. 10 seeds, mean±std everywhere.

## 4. Results

### 4.1 The bake-off (canonical, strong learner) — Table 1 [RESULTS Part 4c]
No-halt ceiling 71.9±0.5%. conv 71.2±0.5% @ 3.89±0.09/6 steps (~35% saved,
uniform 10/10); dstate 71.0±0.6% @ 5.12±0.15. **Binary discrimination**: only
conv/dstate admit a within-slack τ (10/10 seeds); ent/recon/rnorm/dent 0/10
(FALLBACK, −3 to −6pp). Premature-halt per halt: conv 0.02 vs ent 0.50. conv
beats the within-block null by +7.4pp.
- Figure 1: tau-sweep acc-vs-steps curves, all six signals + fixed-depth frontier
  [regenerate from results/bakeoff_reachp2_s*.json].

### 4.2 Why surprise fails: failure decomposition — Figure 2 [RESULTS 4a]
entropy/recon exit early-and-wrong on 8–9% of examples (signal reads "confident,
stop" mid-chain; corr(ans, sig) negative). Convergence signals wait out the chain.
Mechanism differs by task (drift-to-budget on multi-hop MQAR) — same verdict.

### 4.3 External transfer: multi-hop associative recall — Table 2 [RESULTS Part 5]
conv/dstate beat ent/recon +2.5pp on 10/10 seeds at 2.5 vs 5.4 steps; halt depth
grows with hop count (2.75 → 3.51). Single-hop control: all signals cost-free
(errors are unanswerable probes) — signal choice matters exactly when the task has
depth structure.

### 4.4 The counterweight (honesty section) — [RESULTS Part 4b]
On the memory-only task, recon(decodability) ≈ entropy at matched compute: the
reconstruction family fails on *joint* tasks, not universally. Weak-learner runs +
stronger nulls corroborate (dstate the only useful operating point there).

## 5. Discussion

- **Diagnosis**: the write gate and the depth knob want *different observables* —
  reconstruction-miss detects "unknown content" (right for writing), convergence
  detects "computation settled" (right for stopping). The one-signal unification is
  refuted in its literal form.
- The "two gradients of one memory loss" reading (∇_W L for write, ∇_s L for depth)
  is *consistent* with our data but untested as a mechanism; testing it properly
  needs interventional, functional-form, cross-regime evidence in a tractable
  setting (ICL linear regression) — future work, not claimed here.
- Practical recipe: if you add halting to a TTT-style model, use a convergence
  observable + slack-based held-out calibration; do not reuse the write signal.

## 6. Limitations (explicit section — reviewers reward this)

- Mechanism-scale: 0.2–0.9M params, synthetic tasks; no LLM-scale claims.
- Mini-MQAR is not the standard zoology config (vocab 64 vs 8192). [Backfill planned.]
- Part 1 depth∝difficulty (corr +0.92) currently single-seed. [Backfill planned.]
- Compute counted in retrieval steps (delta-rule write FLOPs unaffected by halting).
- Multi-hop MQAR base learner weak (41.6% ceiling) — discrimination result robust,
  absolute accuracy not meaningful.

## 7. Conclusion
One paragraph: surprise loses, convergence wins, use the recipe.

---

## Figures & tables plan

| # | content | source |
|---|---|---|
| Fig 1 | tau-sweep curves + fixed-depth frontier, 6 signals (canonical) | bakeoff_reachp2_s*.json (regenerate multi-seed version) |
| Fig 2 | failure decomposition stacked bars, per signal | reachp3/bakeoff JSONs |
| Fig 3 | MQAR multi-hop: acc-vs-steps + depth∝H | mqarhop_seed*.json |
| Tab 1 | canonical bake-off (acc, steps, tau_ok, premature, null) | RESULTS Part 4c |
| Tab 2 | external transfer summary | RESULTS Part 5 |
| App A | signal inventory + two recon definitions | RESULTS signal inventory |
| App B | weak-learner bake-off (4b) + rule counterweight | sweep logs |
| App C | tau-rule ablation (argmax vs slack — the bimodality artifact) | reachp3 vs bakeoff JSONs |

## TODO before submission (targeted backfill only — PROJECT §7)

- [ ] Part 1 `depth_sanity` multi-seed (10) + archive logs. (cheap, CPU-able)
- [ ] ONE standard-config zoology-style MQAR anchor run (vocab 8192-class) vs
      published baselines. (GPU; design before running — don't tune a curriculum)
- [ ] Regenerate Fig 1 as a multi-seed band plot (current PNG is seed-0 only).
- [ ] Venue pick + page-limit fit; LaTeX port once section content stabilizes.

## Writing conventions

- Every number: mean±std, 10 seeds, held-out tau — or explicitly flagged otherwise.
- Never conflate the two recon definitions; never merge 4a/4b/4c tables.
- The original thesis is reported as *tested and refuted for halting* — this is the
  paper's honesty asset, not a weakness.
