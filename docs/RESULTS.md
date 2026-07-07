# Results

## Main claim (reconciled 2026-07-07, after two independent bake-offs + MQAR transfer)

> **A convergence signal (readout sym-KL `conv` / state-step norm `dstate`) is the
> correct halting observable for the depth knob.** Across the 10-seed joint task
> (Part 4a), an independent weak-learner bake-off with stronger nulls (Part 4b),
> and single- *and* multi-hop MQAR (Part 5), convergence halting holds accuracy at
> the fixed-depth ceiling while entropy- and reconstruction-error halting lose
> 2–6pp — and on the *external* multi-hop task it does so at **less than half the
> compute on 10/10 seeds**.
>
> The compute saving is now **established, not caveated**: the canonical run
> (Part 4c — the rigorous protocol on the strong learner) shows `conv` reaching
> within 1pp of the ceiling at **3.89±0.09 / 6 steps (~35% saved), uniformly on
> 10/10 seeds** — the earlier "bimodal / conservative" caveat was an artifact of
> the argmax-accuracy tau rule and dissolves under the correct
> fewest-steps-within-slack objective. The discrimination is binary there:
> **only the convergence family admits a within-slack operating point at all**
> (conv/dstate 10/10; entropy/recon/rnorm/Δ-entropy 0/10).
>
> **What is *not* supported is the stronger "one surprise scalar drives both
> knobs"**: the compute-efficient halter (`conv`) is a pure readout statistic with
> no write-side role, and the literal thesis scalar (`recon`, state-mismatch form)
> *loses* at depth-halting on every joint task. Depth wants a convergence
> observable, the write wants a reconstruction-miss observable; the "two gradients
> of one memory loss" unification remains an **untested interpretation** — it
> becomes a finding only if the halt signal is shown to predict the write
> magnitude (the probe in `docs/mqar_design.md`). One counterweight from Part 4b:
> on the *memory-only* task a reconstruction-family scalar (decodability form)
> matches entropy at matched compute — recon-halting fails on *joint* tasks, not
> everywhere.

The evidence chain:

| # | claim | task | key number | status |
|---|---|---|---|---|
| 1 | depth halting tracks difficulty | in-context reachability | `corr(K, halt) = +0.997±0.001` (10 seeds, 2026-07-07 rerun) | ✅ |
| 2 | memory buys accuracy + compute | hidden-rule (partial obs) | persist 5%→81% (10 seeds: 50.5±0.4% overall); both 2.4 vs 8.0 latent steps; `corr(ans, entropy) = −0.96` | ✅ |
| 3 | joint, *entropy* halt (historical) | partial-obs reachability | entropy halting costs accuracy | 🟡 superseded by Part 4 |
| 4a | joint halting is **accuracy-preserving** with a convergence signal | partial-obs reachability (strong learner), **10-seed** bake-off | conv/dstate −0.0pp vs persist 72.0±0.6%; entropy/recon −5.6pp | ✅ |
| 4b | independent bake-off converges (weak learners, stronger nulls) | rule + original reachp, 10 seeds | reachp: dstate the only useful operating point; rule: recon(decodability) ≈ entropy | ✅ |
| 4c | **canonical**: rigorous protocol × strong learner; bimodality dissolves | reachp2 via `bakeoff.py`+conv, 10 seeds | conv 71.2±0.5% @ **3.89±0.09** steps (uniform); only conv/dstate admit a within-slack tau (10/10 vs 0/10) | ✅ |
| 5 | convergence-halting **transfers externally** | MQAR single- & multi-hop, 10 seeds each | multi-hop: conv/dstate beat entropy/recon **+2.5pp on 10/10 seeds** at 2.5 vs 5.4 steps | ✅ |

**Cross-cutting caveats**: "compute" = latent retrieval steps only (delta-rule
write FLOPs are unaffected by halting); models are 0.2–0.9M params on synthetic
tasks — these are *mechanism* pilots, not LLM-scale evidence. Provenance:
all Parts are 10 seeds as of 2026-07-07 (Part 1 regenerated with archived logs). `tau` is calibrated on a
**held-out batch everywhere as of 2026-07-07** (`bakeoff.py` and
`ablation_reachp3` always did; `calib_tau` in the rule/reachp/reachp2 ablations
was fixed to held-out — archived seed-0 logs predate the fix; a held-out rerun
reproduced the reachp2 numbers, `results/reachp2_run_heldout.log`).
**Part 4a and 4b are NOT one comparison** — different tau-selection objectives
and different `recon` definitions (see the warning in Part 4).

---

## Part 1 — Depth-only proof (`experiments/depth_sanity.py`, `models/recurrent.py`)

In-context functional-graph reachability (whole graph given each query). The
recurrent-depth reasoner's **halting tracks difficulty** — regenerated at
**10 seeds** (2026-07-07; logs archived): accuracy rises with test-time depth
(r=1: 15.7±0.4% → r=12: 100.0%), per-K accuracy at full depth is 100% for
every K ≤ 12, steps-to-converge equals K **exactly** (per-K mean conv-step = K
on every seed), and **`corr(K, steps-to-converge) = +0.997±0.001`** (per-seed
range [+0.997, +0.998]).

**What the halting signal actually is here**: steps-to-prediction-stability
(first step after which the argmax prediction never changes), computed in
hindsight over the full rollout — i.e. *convergence to the sink fixed-point*,
not a surprise/reconstruction scalar, and not an online rule. It isolates the
**depth knob** and shows the task has the right difficulty structure; it does
not yet test surprise-driven halting. Memory is redundant here (graph fully in
context). *Provenance: the original single-seed numbers (corr +0.92; r=1 → 64%,
r≥6 → 100%) do not reproduce under the current config (n=28, sinks=2, T=12) and
their log was never archived — they are **superseded** by the archived 10-seed
rerun above (`results/depth_sanity_s{0..9}.log`, 3000 steps, defaults).*

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
  vs both's 20→27 shows the foregone gains). *(Historical: this was the central
  open problem at the time; Part 4 below resolves it — the cost is specific to
  the entropy/recon signals, not the mechanism.)*
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

**Diagnosis (revised, then resolved by Part 4)**: three confounded causes, not one
— (a) base-learner limits on chain-following over accumulated memory, (b) the
measurement artifact above, and (c) a halting-rule failure for multi-hop
(confident-wrong early exits). `ablation_reachp2` (curriculum + aux + d=256, 10
seeds) fixes (a) — persist rises to **72.1±0.6%** (reset 43.3% → persist 72.1%) —
which isolates (c). The bake-off (**Part 4**) then shows (c) is a **signal
*choice*** problem, not a mechanism failure: entropy/recon halt confident-wrong,
but convergence signals do not.

![partial-obs reachability](../results/reachp_curve_v2.png)

*(Figure regenerated after the labeling fix; the pre-fix artifacts
`reachp_curve.png` / `reachp_run.log` are retained for provenance.)*

## Part 4 — Halting-signal bake-off (two independent implementations)

> ⚠️ **4a and 4b are not one comparison.** They differ in base learner
> (curriculum-fixed 72% vs original weak learners), **tau-selection objective**
> (4a: argmax calibration-accuracy; 4b: fewest steps within an accuracy slack),
> and **`recon` definition** (4a: state-vs-read mismatch ‖n(sₜ₊₁)−n(rₜ)‖²; 4b:
> vocab-decodability min_v‖rₜ−node(v)‖²). Quote them separately.
> **Part 4c below unifies them** (4b's protocol + `conv`, strong learner) and is
> the **canonical** table to quote; 4a/4b are retained as history/robustness.

### Part 4a — strong learner (`experiments/ablation_reachp3.py`, 10 seeds)

Joint task with the curriculum-fixed base learner (reachp2 config, d=256); races
four halting signals on **identical trajectories** with **per-arm `tau` calibrated
on a held-out batch**, plus fixed-depth / random-halt baselines and a per-example
failure decomposition. Persist ceiling (fixed depth T=6) **72.0±0.6%**.
(Same quantity as reachp2's persist **72.1±0.6%** — full depth, no halt — measured
by a different script on an independent eval draw; the 0.1pp is sampling noise.)

| halt signal | acc | avg steps | gap vs persist | early-wrong | corr(ans, sig₀) |
|---|---|---|---|---|---|
| `dstate` = ‖Δsₜ‖²/d | **72.0±0.6%** | 5.93±0.06 | **−0.0pp** | **1.1%** | +0.05 |
| `conv` = sym-KL(pₜ, pₜ₋₁) | 71.9±0.6% | 5.00±0.58 † | **−0.0pp** | 3.6% | +0.14 |
| `entropy` (status quo) | 66.4±0.9% | 5.44 | −5.6pp | 8.5% | −0.45 |
| `recon` = ‖n(sₜ₊₁)−n(rₜ)‖² | 66.3±0.7% | 5.39 | −5.6pp | 8.8% | −0.25 |

† conv's step count is **bimodal across seeds** here (std 0.58 vs dstate's 0.07):
7/10 seeds save compute (~4.6 steps), 3/10 collapse to ~5.9. **Resolved
(2026-07-07, Part 4c)**: this bimodality — and dstate's "barely-halts"
conservatism — was **an artifact of this script's tau rule** (`tau = argmax`
calibration *accuracy*, ignoring steps: on a near-flat acc-vs-tau curve the
operating point lands anywhere). Under Part 4c's fewest-steps-within-slack
objective on the *same* task/learner class, conv is a **uniform** saver
(3.89±0.09 steps, 10/10 seeds) and dstate also actually halts (5.12±0.15).
(corr(ans, sig₀) for conv is measured at step 1 — no finite step 0.)

**Read carefully — the result is real but narrow:**

- **The −5.6pp halting cost is signal-specific.** `entropy` and `recon` halt
  aggressively but *miscalibrated*: ~9% of examples exit early-and-wrong (their
  signal reads "confident, stop" on still-unresolved multi-hop probes; negative
  `corr(ans)` confirms the signal anti-tracks answerability). This — not the
  mechanism — is what cost accuracy in Part 3.
- **Convergence-family signals remove it.** Both `conv` and `dstate` match the
  persist ceiling (−0.0pp). But they do so differently, and only *sometimes*
  cheaply: **`conv` saves compute on most seeds** — it halts early *and correctly*
  (mean early-right 34%), reaching the ceiling in ~4.6 steps on **7/10** seeds —
  but this is **bimodal** (see † : on 3/10 seeds `conv` collapses to dstate-like
  conservatism, no saving). **`dstate` is the *safe* one** — early-wrong just 1.1%
  — but **conservative**: its accuracy-optimal `tau` almost never halts before the
  budget (5.93/6 steps), so it matches persist largely by *declining to halt*
  rather than by saving compute.
- **Implication for the thesis.** `dstate = ‖Δs‖²` is the signal the unification
  story points to (‖Δs‖ ≈ η‖∇_s L_mem‖ near the step's fixed point), and it
  preserves accuracy — but *conservatively* (5.93/6 steps, early-right ~3%): it
  mostly declines to halt, so "cost-free" here means **accuracy-preserving, not
  compute-saving**. The strong claim ("one scalar drives both knobs *optimally*")
  does **not** hold: the compute-efficient halter (`conv`) is a readout statistic
  with no write role, and the literal thesis scalar (`recon`) *loses* at halting
  (with this definition + held-out `tau`). The recon miss-detector is right for
  gating *writes*, wrong for gating *depth* — i.e. the two knobs want **different
  observables**. The "two gradients of one memory loss" reading (§ main claim) is
  a cautious interpretation, not established here; whether `dstate` predicts the
  write magnitude is the open probe (`docs/mqar_design.md`).

![halting-signal bake-off](../results/reachp3_bakeoff.png)

### Part 4b — weak learners, stronger nulls (`experiments/bakeoff.py`, 10 seeds)

An **independent implementation** (parallel session) on the *original* rule and
reachp tasks (no curriculum fix), with a stricter protocol: five signals (`ent`,
`recon`=decodability, `rnorm`, `dstate`, `dent`=Δ-surprise), a **within-block
shuffle null** (`shuf_b` — beating it requires true per-example information, not
just a per-position schedule), **full tau-sweep (acc, steps) curves** for
matched-compute reads, slack-based held-out tau, and checkpoint reuse.

| task (persist) | signal | acc | steps | verdict |
|---|---|---|---|---|
| rule (50.5±0.4%) | ent | 50.1±0.7% | 2.22 | works |
| rule | **recon** (decodability) | 50.1±0.5% | 2.38 | **≈ ent — recon-halting survives on the memory-only task** |
| rule | dstate | 50.3±0.6% | **1.90** | best |
| reachp (35.1±0.7%) | ent | 34.7±0.6% | 9.71 | refuses to halt (safe but useless) |
| reachp | recon (decodability) | 31.7±0.6% | 8.39 | −3.4pp, tau fallback |
| reachp | **dstate** | 33.8±0.6% | **6.28** | **only useful operating point** (−1.3pp at 63% compute) |

Two things 4b adds that 4a cannot: (i) on the **memory-only** task, a
reconstruction-family scalar is as good a halting signal as entropy — the
recon-halting failure is specific to *joint* (multi-hop-over-memory) tasks, not
universal; (ii) the earlier "−9.5pp halting cost" on weak-learner reachp is
**substantially a median-tau calibration artifact** — with slack-based held-out
tau, entropy doesn't *hurt*, it just provides no compute-saving operating point.
Note again: 4b's `recon` (decodability) is a **different scalar** from 4a's
(state-mismatch); "recon survives on rule / loses on joint" is a statement about
two definitions, not one signal behaving inconsistently.

### Part 4c — canonical run (`bakeoff.py --task reachp2` + `conv`, 10 seeds)

The unifying run: 4b's rigorous protocol (slack-based held-out tau, within-block
shuffle null, tau-sweep curves, ckpt reuse) with `conv` added to its signal set,
on 4a's **strong learner** (reachp2 curriculum + aux, d=256, per-seed retrained,
kcap=4 eval). No-halt ceiling **71.9±0.5%**; slack = 1.0pp.

| signal | acc | steps | within-slack tau exists? | premature/halt | shuf_b null |
|---|---|---|---|---|---|
| `conv` | **71.2±0.5%** | **3.89±0.09** | ✅ 10/10 | 0.02 | 63.8% (+7.4pp real info) |
| `dstate` | 71.0±0.6% | 5.12±0.15 | ✅ 10/10 | 0.05 | 68.7% |
| `dent` (Δ-entropy) | 68.8±0.5% | 5.70 | ❌ 0/10 FALLBACK | 0.38 | 70.1% |
| `rnorm` | 68.2±0.6% | 5.57 | ❌ 0/10 | 0.39 | 68.6% |
| `recon` (decodability) | 67.1±0.5% | 5.45 | ❌ 0/10 | 0.44 | 67.5% |
| `ent` | 66.2±0.5% | 5.43 | ❌ 0/10 | 0.50 | 67.1% |

- **The 4a caveats dissolve.** conv's per-seed steps are [3.84–4.07] — perfectly
  uniform (no bimodality), a **~35% compute saving at ≤1pp accuracy cost on every
  seed**. dstate's conservatism also relaxes (5.93 → 5.12). Both were artifacts
  of 4a's argmax-accuracy tau rule.
- **The discrimination becomes binary.** Only the convergence family *admits* a
  within-slack operating point (conv/dstate: `tau_ok` 10/10). All four
  non-convergence signals fail to find any tau within 1pp of the ceiling (0/10,
  falling back to max-accuracy and still losing 3–6pp). Premature-halt fraction:
  conv 0.02 vs ent 0.50.
- **Above the null.** conv beats the within-block shuffle null by +7.4pp — its
  halting decisions carry genuine per-example information, not a per-position
  schedule.
- This is now the **canonical joint-task bake-off**; `ablation_reachp3` (4a) is
  retired to history. Artifacts: `results/bakeoff_reachp2_s{0..9}.{json,log}`.

## Part 5 — External transfer: MQAR (10 seeds each; full details `docs/mqar_design.md`)

- **Single-hop MQAR**: convergence-halting **transfers** — `conv` matches the
  ceiling (68.2±0.5%) at **2.53/6 steps (~58% compute saved, no bimodality)**. But
  the task doesn't discriminate signals (all four cost-free): its errors are the
  ~30% *unanswerable* probes (wrong at any depth), so early halting is harmless.
- **Multi-hop MQAR** (H-hop recall, H∈1..3 = difficulty): the **discrimination
  reproduces externally** — conv/dstate beat entropy/recon by **+2.5pp on 10/10
  seeds** (conv−entropy +2.76±0.35pp) at less than half the compute (2.5 vs 5.4
  steps); depth grows with hop count (dstate halt-step 2.75→3.51). Mechanism
  differs from reachp (here entropy/recon *drift to the budget* past the depth-3
  accuracy peak, rather than halting confident-wrong-early) — same conclusion:
  **entropy/recon are the wrong halting observables; convergence is right.**
  Caveat: weak multi-hop base learner (persist 41.6±0.9%) — mechanism-scale.
- **Standard-config anchor** (2026-07-07, single-hop, vocab **8192**, m=4, Q=16
  ≈ zoology's KV upper end, 10 seeds): the **transfer holds at standard scale**
  — conv's eval tau-sweep admits a within-1pp-of-ceiling operating point at
  **2.24±0.16/6 steps on 10/10 seeds** (curve read; the inherited argmax-tau
  headline barely halts — the Part-4a tau-rule artifact reproduced at 8192).
  Ceiling 51.7±0.6% (weaker base learner at standard vocab); single-hop remains
  non-discriminating (all four signals +0.0pp), as at vocab 64. Multi-hop was
  *not* rerun at 8192 (optional residual). Full details + honest caveats:
  `docs/mqar_design.md` §"Standard-config anchor".

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

## Signal inventory (2026-07-06; updated after the bake-off — read before quoting "surprise")

Different experiments use different scalars; the docs previously blurred them:

| experiment | write gate | halting signal |
|---|---|---|
| Part 1 `depth_sanity` | — (no memory) | prediction-stability (hindsight convergence) |
| `ablation_ttt` / `ablation_amort` | recon error (shared) | recon error (shared) — the literal thesis; inconclusive/negative tasks |
| Part 2 `ablation_rule` | recon error | readout entropy |
| Part 3 `ablation_reachp` | recon error | readout entropy |
| **Part 4a `ablation_reachp3`** | recon error (write unchanged) | bake-off: entropy / conv / dstate / **recon = state-vs-read mismatch** |
| **Part 4b `bakeoff.py`** | recon error (write unchanged) | bake-off: ent / **recon = vocab decodability** / rnorm / dstate / dent |
| **Part 5 `ablation_mqar{,_hop}`** | recon error (write unchanged) | bake-off: entropy / conv / dstate / recon (state-mismatch) |

**Verdict on the literal "one reconstruction-error scalar" thesis**: tested and
it **loses at halting on every joint task** (4a −5.6pp; 4b −3.4pp with the
decodability form; Part 5 multi-hop −1.6pp / budget-drift) — but **matches
entropy on the memory-only task** (4b rule). A single scalar drives halting
cost-free only if it is a *convergence* signal (`conv` / `dstate`); the
write-relevant one (`dstate` ≈ ‖∇_s L‖) preserves accuracy but its compute
saving is tau-rule-dependent (Part 4a †). See §"Part 4" and `PROJECT.md` §4.
