# Convergence, Not Surprise: Diagnosing the Right Halting Signal for Fast-Weight Memory Reasoners

*Workshop paper draft — v1 prose (2026-07-07). Target: mechanistic-interpretability
/ efficient-inference workshop (4–5 pages + appendix). All numbers below are final,
10-seed, held-out-tau results already in `docs/RESULTS.md`; pointers in [brackets].
Every headline number re-verified against the source JSONs on 2026-07-07.*

**Working titles (pick one):**
1. Convergence, Not Surprise: Diagnosing the Right Halting Signal for Fast-Weight Memory Reasoners
2. When Should a Test-Time Memory Stop Thinking? A Halting-Signal Bake-off
3. Halting Signals for TTT-Style Memory Reasoners: Surprise Loses, Convergence Wins

---

## Abstract

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
within 1pp of the no-halt ceiling — the best of them, readout convergence, reaching
it at ~65% of the compute uniformly across seeds. The discrimination reproduces on an external multi-hop associative
recall task (+2.3–2.8pp over entropy/recon, winning on all 10 seeds, at less than half the compute),
where halting depth also grows with hop count. The diagnosis is clean: the write gate
wants a reconstruction-miss observable, but the depth knob wants a *convergence*
observable — one signal does not serve both. We release the bake-off protocol as a
recipe for evaluating halting signals in test-time-adaptive models.

*(~180 words; trim to venue limit.)*

## 1. Introduction

Modern sequence models spend extra test-time compute along two distinct axes. One is
**latent depth**: iterate a recurrent computation and stop when the answer is ready
(adaptive computation time, PonderNet, Coconut, recurrent-depth transformers). The
other is **weight adaptation**: update parameters on the test stream itself, as in
test-time training (TTT), TTT layers, and Titans-style memory modules that write to a
fast-weight state. Recent systems drive one axis or the other, and each needs a
control signal: *when has the computation settled enough to stop?* and *when is the
input surprising enough to write?*

A seductive unification suggests the two questions share an answer. Titans gates its
writes by a **surprise** scalar — the memory's self-supervised reconstruction error —
so that novel content is written and familiar content is not. Recurrent-depth models
halt on **convergence** of the latent state. Read side by side, these *look like* two
readings of one underlying quantity: high surprise means "keep thinking and keep
writing," low surprise means "halt and stop writing." The intuition even has a tidy
predictive-coding gloss — one prediction error that drives both settling and
plasticity (Rao & Ballard 1999; Friston). To our knowledge nobody had tested whether
a single scalar actually serves both roles when the two axes are exercised together.

**We ran that test and got the opposite of the tidy story.** We built a small
recurrent-depth reasoner with a delta-rule fast-weight memory (§3.1) and a *joint*
task that provably needs both axes — partial-observation reachability, where the
answer requires memory accumulated across queries *and* multi-hop chasing at
inference (§3.2). We then raced six candidate halting signals under a protocol
designed to make the comparison honest (held-out threshold calibration, matched-compute
tau sweeps, shuffle nulls, per-example failure decomposition, 10 seeds; §3.4). The
surprise-family signals — reconstruction error and readout entropy — are the *wrong*
halting observable: they exit early-and-confident on unresolved multi-hop chains and
lose 3–6pp, or refuse to halt at all. The signals that work are **convergence**
observables of the computation itself. We set out to confirm the one-signal thesis and
instead diagnosed why it cannot hold for halting; the paper is that diagnosis plus the
recipe that produced it.

**Contributions.**
1. **A rigorous halting-signal bake-off protocol** for test-time-adaptive reasoners:
   one no-halt rollout with all signals traced post-hoc on identical trajectories,
   per-signal thresholds calibrated on a *held-out* stream by fewest-steps-within-slack,
   a *within-block shuffle null* that isolates per-example information, full tau-sweep
   matched-compute curves, and a four-way per-example failure decomposition.
   [experiments/bakeoff.py]
2. **A clean negative-plus-positive diagnosis.** On the joint task, surprise/entropy
   thresholds fail as halting rules (0/10 seeds admit a near-ceiling operating point,
   −3 to −6pp); convergence observables are the *only* signals that do (10/10 seeds,
   within 1pp of the no-halt ceiling — the best at ~65% of the compute). [Part 4c]
3. **External reproduction and a counterweight.** The discrimination reproduces on
   multi-hop associative recall (+2.3–2.8pp over entropy/recon on all 10 seeds at less than
   half the compute; halting depth grows with hop count) [Part 5]; and on a
   *memory-only* control a reconstruction-family signal *does* match entropy [Part 4b],
   showing the failure is specific to *joint* tasks, not universal.

We deliberately foreground the honesty of the result: the write gate wants a
reconstruction-miss observable and the depth knob wants a convergence observable, so
the literal one-signal unification is refuted for halting. That refutation, not a
speculative mechanism, is the contribution.

## 2. Related work

**Latent-depth halting.** Adaptive computation time and PonderNet learn a halting
distribution over recurrent steps; Coconut (2412.06769) and He & Tang (2509.24238,
"Learning to Ponder") extend ponder-style gating to latent reasoning; recurrent-depth transformers
(2502.05171) halt on *KL-convergence* of the latent state — the closest prior to our
convergence signals. HRM (2506.21734) and TRM (2510.04871) learn a Q-halting head on
synthetic reasoning. These works learn or hand-design a halting rule but do not race
candidate signals against each other on a task that also adapts weights.

**Weight adaptation.** TTT (1909.13231) and TTT layers (2407.04620) update parameters
on the test stream; Titans (2501.00663) writes to a fast-weight memory gated by a
**surprise** (reconstruction-error) scalar; Sim (2601.00894, "When to Ponder")
gates TTT compute with a binary ponder decision. The write signal in all of these is a reconstruction miss — exactly the signal
our bake-off finds to be the *wrong* halting observable.

**Depth–memory interaction.** Sapunov (2604.21999, "Universal Transformers Need
Memory") combines train-time capacity with
an ACT router; we differ on both counts — our adaptation is *test-time* accumulation
into a fast weight, and our halting is *signal-driven* with no learned router or
Q-head.

**Recall stress tests.** Our external transfer task lives in the MQAR family (Zoology
2312.04927; Based 2402.18668; DeltaNet 2406.06484; Gated DeltaNet 2412.06464), which
we use to check that the diagnosis is not an artifact of the reachability task.

**The open cell.** Prior halting work supplies the stopping rule from *outside* the
memory (a learned router, a Q-head, an RL objective). The unfilled position — and our
subject — is the memory module's *own dynamics* as the halting signal, with no learned
controller. Our finding is that within that space the choice of dynamical observable
is decisive: convergence works, reconstruction-miss does not.

## 3. Setup

### 3.1 Model
The **RuleReasoner** [src/awe/models/memory.py] is a recurrent-depth reasoner over a
persistent fast-weight memory (0.86M params, d=256). Node embeddings feed a fast-weight
delta on top of a base map W_base, written by a **normalized delta rule** whose write
strength is proportional to the reconstruction miss (the Titans-style surprise gate).
Retrieval is a recurrent loop `s_{t+1} = s_t + MLP([s_t, W·LN(s_t)])` run up to a
budget of T=6 steps, followed by a readout head. Episodes are streaming: Q=12 queries
per episode, m key→value pairs revealed per query, and **memory persists across the
stream** so later queries benefit from earlier writes. Crucially, retrieval never
mutates the memory, so a single no-halt rollout exposes the full latent trajectory and
every candidate halting signal can be traced *post-hoc* on identical dynamics (§3.4).

### 3.2 Tasks
**Joint task (primary): partial-observation reachability** [datasets/reachp.py]. A
functional graph is revealed incrementally across the query stream; each probe asks
"which sink does v0 reach?" with difficulty K = path length. The task needs *memory*
(edges seen in earlier queries) and *depth* (chase the K-hop path at inference), so it
exercises both knobs at once. A competent base learner is a prerequisite for any
controller verdict: with a naive learner the persist ceiling is only ~35%, but a
K-curriculum plus an auxiliary next-node loss (d=256) lifts it to **72.1±0.6%** (10
seeds) [ablation_reachp2].
We run the bake-off on this strong learner so that halting failures cannot be blamed on
a base learner that never solves the chain.

**Memory-only control: hidden-permutation rule** [datasets/rule.py]. A hidden
permutation answerable only from cross-query memory; isolates the weight knob (persist
5%→81%, 50.5±0.4% overall, 10 seeds). Used as the counterweight in §4.4.

**External transfer: MQAR** [datasets/mqar.py, mqar_hop.py]. Single-hop and multi-hop
(H∈1..3) associative recall in the Zoology MQAR family, to check the diagnosis outside
reachability. *(Backfill: one standard-config zoology run — see TODO.)*

### 3.3 Candidate halting signals
All signals halt when the signal falls below a threshold τ. We race six, and label
each by family:

- **Magnitude family** (surprise-like): `ent` = readout entropy; `recon` =
  reconstruction error (two forms, always reported separately and *never* conflated —
  state-mismatch ‖n(s_{t+1})−n(r_t)‖² in Part 4a/5, vocab-decodability
  min_v‖r_t−node(v)‖² in Part 4b/4c); `rnorm` = −‖r‖².
- **Convergence family** (settling-like): `dstate` = ‖Δs‖²/d (latent step norm);
  `conv` = sym-KL(p_t, p_{t−1}) (readout distribution convergence); `dent` = |Δent|.

The magnitude family measures *how surprising / how confident* the current state is;
the convergence family measures *how much the computation is still moving*. The whole
result is a statement about which family the depth knob wants.

### 3.4 Protocol (the recipe — box figure)
1. **One no-halt rollout per query**; all six signals traced post-hoc on the identical
   trajectory (exact, because retrieval never mutates memory).
2. **τ calibrated on a held-out stream**, per signal, as the fewest average steps
   within a 1pp accuracy slack of the no-halt calibration accuracy (fallback: max
   calibration accuracy, explicitly flagged as FALLBACK). *[Ablation: the earlier
   argmax-accuracy rule produced spurious seed-bimodality — Appendix C.]*
3. **Controls:** fixed-depth frontier (accuracy at every depth 1..T); random halting
   matched on steps; and a **within-block shuffle null** — permute halt steps within
   each query index, so beating it requires genuine *per-example* information rather
   than a good per-position schedule.
4. **Report:** the accuracy-vs-avg-steps operating point *and* the full tau-sweep
   curve; a per-example failure decomposition (early-right / premature / wrong-anyway /
   full-depth); and corr(answerable, signal).
5. **10 seeds, mean±std everywhere.**

## 4. Results

### 4.1 The bake-off (canonical, strong learner) — Table 1 [RESULTS Part 4c]

On the strong-learner joint task the no-halt ceiling is **71.9±0.5%**. Table 1 races
all six signals under the §3.4 protocol (slack = 1.0pp). The result is a **binary
discrimination**: only the two convergence signals `conv` and `dstate` admit *any*
threshold within the 1pp slack, and they do so on **all 10 seeds**; the four
magnitude/other signals (`ent`, `recon`-decodability, `rnorm`, `dent`) find no
within-slack τ on **any** seed and fall back to max-accuracy, still losing 3–6pp.

| signal | family | acc | avg steps | within-slack τ | premature/halt | shuffle-null |
|---|---|---|---|---|---|---|
| `conv` | convergence | **71.2±0.5%** | **3.89±0.09** | ✅ 10/10 | 0.02 | 63.8% (**+7.4pp**) |
| `dstate` | convergence | 71.0±0.6% | 5.12±0.15 | ✅ 10/10 | 0.05 | 68.7% |
| `dent` | convergence | 68.8±0.5% | 5.70±0.03 | ❌ 0/10 (FALLBACK) | 0.38 | 70.1% |
| `rnorm` | magnitude | 68.2±0.6% | 5.57±0.03 | ❌ 0/10 | 0.39 | 68.6% |
| `recon` (decod.) | magnitude | 67.1±0.5% | 5.45±0.02 | ❌ 0/10 | 0.44 | 67.5% |
| `ent` | magnitude | 66.2±0.5% | 5.43±0.03 | ❌ 0/10 | 0.50 | 67.1% |

Three things stand out. **(i) A real compute saving.** `conv` reaches within 1pp of
the ceiling at 3.89/6 steps — a **~35% saving** — and it is *uniform*: per-seed steps
lie in [3.80, 4.07] with no bimodality. **(ii) The saving carries information.** `conv`
beats its within-block shuffle null by **+7.4pp** (71.2 vs 63.8), so its halt decisions
depend on the specific example, not a per-position schedule. **(iii) Surprise halts
prematurely.** The premature-halt fraction per halt is 0.02 for `conv` versus 0.50 for
`ent` — half of every entropy halt fires on an example that had not yet resolved.
`dstate` is the *safe*
convergence signal (premature 0.05) but more conservative (5.12 steps), matching the
ceiling with a smaller saving.

- **Figure 1**: tau-sweep acc-vs-steps curves (10-seed mean±std bands) for all six
  signals plus the fixed-depth frontier and calibrated operating points —
  `results/fig1_bakeoff_band.{png,pdf}`, generated by `scripts/fig1_bakeoff_band.py`.

### 4.2 Why surprise fails: failure decomposition — Figure 2 [RESULTS Part 4c; corroboration Part 4a]

The canonical run's own per-example decomposition names the failure. Under each
signal's calibrated τ, **premature halts** — halted early and answered wrong where the
full-depth rollout answers right — hit 5.9±0.3% of examples for `ent` and 5.0±0.2%
for `recon` (decodability), versus 2.1±0.5% for `conv` and 1.9±0.5% for `dstate`;
per halt issued, half of entropy's halts are premature (0.50 vs `conv`'s 0.02;
Table 1). And the magnitude signals point the wrong way: corr(answerable, signal) is
*negative* (`ent` −0.46±0.01, `recon` −0.18±0.02) — the signal reads "confident,
stop" partway through an unresolved multi-hop chain — while the convergence signals
track answerability weakly *positively* (`conv` +0.15±0.01, `dstate` +0.05±0.01) and
simply keep moving until the computation settles. The retired strong-learner bake-off
corroborates independently (Part 4a, whose `recon` is the **state-mismatch** form —
a different scalar, same verdict): 8–9% early-and-wrong exits for entropy/recon vs
1–4% for the convergence family, corr −0.45/−0.25 [Appendix B]. The mechanism is
task-dependent — on multi-hop MQAR entropy/recon *drift to the budget* past the
accuracy peak rather than halting confident-wrong-early (§4.3) — but the verdict is
the same: magnitude signals are miscalibrated as halting rules exactly where the task
has depth structure.

### 4.3 External transfer: multi-hop associative recall — Table 2 [RESULTS Part 5]

The discrimination reproduces on an external task. On **multi-hop MQAR** (H-hop recall,
H∈1..3; persist ceiling 41.6±0.9%), convergence beats magnitude at **less than half the
compute** (2.5 vs 5.4 steps), winning on **all 10 seeds** — conv−entropy = +2.76±0.35pp,
conv−recon = +2.32±0.33pp (both 10/10):

| signal | acc | avg steps |
|---|---|---|
| `conv` | **42.3±0.9%** | 2.48±0.11 |
| `dstate` | 42.2±0.9% | 2.72±0.20 |
| `entropy` | 39.5±0.8% | 5.40±0.03 |
| `recon` (state-mism.) | 40.0±0.9% | 5.40±0.02 |

Halting depth also **grows with hop count** — `dstate` mean halt step 2.75±0.12 (H=1) →
3.45±0.13 (H=2) → 3.51±0.18 (H=3) — so the controller spends more steps on harder
probes, the behavior a depth knob should have. The **single-hop control** is the informative null: there all
four signals are cost-free (conv matches the 68.2±0.5% ceiling at 2.53±0.30 steps,
~58% saved), because single-hop errors are the ~30% *unanswerable* probes that are wrong at
any depth — so early halting is harmless and signal choice does not matter. Signal
choice matters **exactly when the task has depth structure**, which is the whole claim.

### 4.4 The counterweight (honesty section) — [RESULTS Part 4b]

We do not claim the reconstruction family is a bad signal *universally* — only for
*halting on joint tasks*. On the **memory-only** rule task, an independent bake-off
(`bakeoff.py`, weak learners, stronger nulls) finds that a reconstruction-family scalar
(**decodability** form) is as good a halting signal as entropy at matched compute
(recon 50.1±0.5% @ 2.38 steps ≈ ent 50.1±0.7% @ 2.22; persist 50.5±0.4%; steps are
10-seed means, stds in RESULTS Part 4b). On the
*joint* reachp task in the same bake-off, entropy simply *refuses to halt* (34.7% @
9.71 steps) and only `dstate` yields a useful operating point (33.8% @ 6.28 steps,
−1.3pp at 63% compute). Two caveats we keep visible: this recon is the **decodability**
scalar — the *same* definition as §4.1's and §4.2's canonical `recon` (all via
`bakeoff.py`), and a *different* one from the **state-mismatch** recon of Part 4a
(§4.2's corroboration) and Part 5 (we never treat
the two as one signal behaving inconsistently); and 4a/4b/4c differ in learner and
τ-rule and are reported separately — §4.1 is the canonical comparison, 4b is
robustness. The counterweight sharpens rather than softens the diagnosis: the write
gate's own signal is fine *for writing*, and even fine for halting when there is no
multi-hop depth to misjudge — it fails precisely on the joint regime the unification
was supposed to cover.

## 5. Discussion

**The diagnosis.** The write gate and the depth knob want *different observables*. A
reconstruction-miss detects "unknown content" — the right trigger for *writing* to
memory. A convergence observable detects "computation settled" — the right trigger for
*stopping*. These come apart on any task with depth structure, because a chain can be
far from settled while the current readout is confidently (and wrongly) low-entropy.
The one-signal unification is therefore refuted **in its literal form**: no single
scalar we tested serves both roles, and the two that halt cost-free (`conv`, `dstate`)
have no write role at all.

**What we do not claim.** A weaker, interpretive reading — that halting and writing
are two gradients of one memory loss (∇_s L for depth, ∇_W L for the write), with
‖Δs‖ ≈ η‖∇_s L‖ near a step's fixed point — is *consistent* with our data, but we
neither test nor claim it: it is near-tautological as stated, and our experiments
neither support nor refute it as a mechanism. Turning it into a finding requires showing the halt signal *predicts the
write magnitude*, which needs interventional, functional-form, cross-regime evidence in
a tractable setting (e.g. in-context linear regression) — future work, and explicitly
out of scope here. We also do not make any write-magnitude claim from the current
probe, which is doubly confounded.

**Practical recipe.** If you add halting to a TTT-style memory model: use a
*convergence* observable (readout-KL or latent step norm), calibrate its threshold on a
*held-out* stream by fewest-steps-within-slack, and validate against a within-block
shuffle null. Do **not** reuse the write/surprise signal as the halting rule — it halts
confident-and-wrong exactly on the multi-hop examples where extra depth would have paid
off.

## 6. Limitations

- **Mechanism scale.** Models are 0.2–0.9M params on synthetic tasks; these are
  mechanism pilots, not LLM-scale evidence.
- **Standard-vocab anchor (single-hop, resolved for transfer).** At the standard
  zoology scale (vocab 8192, ≤64 KV pairs, 10 seeds) the transfer holds: conv's
  eval tau-sweep admits a within-1pp-of-ceiling operating point at 2.24±0.16/6
  steps on every seed (curve read; the inherited argmax-τ headline barely halts —
  reproducing Appendix C's τ-rule artifact at scale). Ceiling 51.7±0.6% (weaker
  base learner at vocab 8192 — mechanism-scale). Single-hop remains
  non-discriminating as in §4.3; the *discriminating* multi-hop variant was not
  rerun at standard vocab.
- **Depth-sanity provenance (resolved).** The Part 1 depth∝difficulty sanity was
  regenerated at 10 seeds with archived logs (corr(K, steps-to-converge) =
  +0.997±0.001; supersedes the earlier unarchived single-seed +0.92).
- **Compute accounting.** "Compute" counts retrieval steps only; delta-rule *write*
  FLOPs are unaffected by halting and not counted.
- **Weak multi-hop learner.** The multi-hop MQAR base learner ceilings at 41.6%; the
  *discrimination* is robust across 10 seeds but the absolute accuracy is not
  meaningful.
- **Interpretation, not mechanism.** The "two gradients of one loss" reading is stated
  as an untested interpretation (§5), not a result.

## 7. Conclusion

We asked whether one surprise signal can govern both test-time knobs of a fast-weight
memory reasoner, and answered no for the halting half. Across a canonical 10-seed
bake-off on a joint memory+depth task and an external multi-hop recall transfer,
reconstruction-error and entropy thresholds halt confidently-and-wrong (or refuse to
halt), while convergence observables are the only signals that reach the no-halt
ceiling — the best of them at ~65% of the compute, uniformly across seeds. Surprise
loses, convergence
wins: the write gate and the depth knob want different observables. We release the
bake-off protocol as a reusable recipe for choosing halting signals in
test-time-adaptive models.

---

## Figures & tables plan

| # | content | source |
|---|---|---|
| Fig 1 | tau-sweep curves + frontier + operating points, 6 signals (canonical) | ✅ `results/fig1_bakeoff_band.{png,pdf}` via `scripts/fig1_bakeoff_band.py` |
| Fig 2 | failure decomposition stacked bars, per signal (canonical 4c) | ✅ `results/fig2_decomposition.{png,pdf}` via `scripts/fig2_decomposition.py` |
| Fig 3 | MQAR multi-hop: acc-vs-steps + depth∝H | ✅ `results/fig3_mqarhop.{png,pdf}` via `scripts/fig3_mqarhop.py` |
| Tab 1 | canonical bake-off (acc, steps, tau_ok, premature, null) | RESULTS Part 4c |
| Tab 2 | external transfer summary | RESULTS Part 5 |
| App A | signal inventory + two recon definitions | ✅ ported to latex/main.tex |
| App B | retired 4a table + weak-learner 4b table | ✅ ported to latex/main.tex |
| App C | tau-rule ablation (argmax vs slack — the bimodality artifact) | ✅ ported (per-seed steps from reachp3 JSONs) |

## TODO before submission (targeted backfill only — PROJECT §7)

- [x] Part 1 `depth_sanity` multi-seed (10) + archive logs — done 2026-07-07: corr
      +0.997±0.001, conv-step = K exactly, `results/depth_sanity_s{0..9}.log`.
- [x] ONE standard-config zoology-style MQAR anchor run (vocab 8192, m=4, Q=16,
      10 seeds) — done 2026-07-07: transfer holds (conv within-1pp point at
      2.24±0.16/6 steps, 10/10, curve read); single-hop non-discriminating as
      expected. See `docs/mqar_design.md` §anchor result. Optional residual:
      multi-hop at vocab 8192.
- [x] Regenerate Fig 1 as a multi-seed band plot — `scripts/fig1_bakeoff_band.py` (2026-07-07).
- [ ] Venue pick + page-limit fit; LaTeX port once section content stabilizes.

## Writing conventions

- Every number: mean±std, 10 seeds, held-out tau — or explicitly flagged otherwise.
- std convention: **population** (ddof=0), matching `scripts/aggregate.py` and the
  experiment aggregators — do not mix with sample std.
- Never conflate the two recon definitions; never merge 4a/4b/4c tables.
- The original thesis is reported as *tested and refuted for halting* — this is the
  paper's honesty asset, not a weakness.
