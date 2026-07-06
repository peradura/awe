# PROJECT — AWE (Adaptive Weight & Exit)

**One surprise signal, two test-time knobs.**

> A single `surprise` signal (a fast-weight associative memory's self-supervised
> reconstruction error) drives, at inference, **both** how much to *think*
> (latent depth halting) **and** how much to *adapt* (fast-weight / TTT update).
> surprise high → keep thinking + update memory more. surprise low → halt + stop.

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

Nobody drives **both from one shared signal**. A theoretical hook: Geiping's
*convergence* halting ("stop when the latent stops moving") and
Titans/PonderTTT's *surprise* update ("write more when surprised") are the **same
signal read two ways** — AWE unifies them.

## 3. Literature positioning (verified 2026-07-03)

Building blocks (arXiv, all verified): Coconut 2412.06769 · FR-Ponder 2509.24238
· Recurrent Depth 2502.05171 · PonderTTT 2601.00894 · Titans 2501.00663 ·
TTT-layers 2407.04620 · PonderNet 2107.05407. Each covers **part**; the
**one-signal-drives-both** cell is empty. See `docs/proposal.md`.

## 4. Main claim (calibrated)

> A single surprise signal can reliably drive **each** axis (depth halting,
> fast-weight memory) and shows **initial directional evidence** when combined.

We do **not** claim optimal joint control yet (the joint task is base-learner
limited). Evidence is a three-part chain:

| # | claim | task | key number | status |
|---|---|---|---|---|
| 1 | depth tracks difficulty | in-context reachability | `corr(K, halt)=+0.92`, 100% | ✅ |
| 2 | surprise=miss; memory buys compute | hidden-rule (partial obs) | `corr=−0.96`, 5→81%, 2.4 vs 8 steps | ✅ |
| 3 | joint (depth + memory) | partial-obs reachability | directional 22→41%, depth saturates | 🟡 |

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
│   ├── proposal.md       # full research proposal
│   ├── RESULTS.md        # results writeup (3-part)
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

## 7. Roadmap

- [x] Depth-only proof (in-context reachability).
- [x] Memory-only proof (hidden-rule) — first positive.
- [x] Joint stress-test (partial-obs reachability) — directional.
- [ ] Sharpen joint: K-curriculum + aux next-node loss + d=256 (`ablation_reachp2`).
- [ ] Scale to main-claim tasks (MQAR / in-context regression) vs TTT baselines.

*Research WIP — Dongwan Yoo, DAIS @ KIER, 2026. See LICENSE.*
