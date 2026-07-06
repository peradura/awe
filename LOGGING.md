# LOGGING — how experiments are recorded

Keep results reproducible and skimmable. Two artifacts per experiment: a **run
log** (raw stdout) and an **index row** in `docs/exp_logs/LOG.md`.

## 1. Run logs → `results/`

Every training/eval run writes unbuffered stdout to a log file, and its figure(s)
next to it:

```bash
python -u -m awe.experiments.ablation_rule --steps 3000 \
  > results/rule_run.log 2>&1
# figures: results/<name>_curve.png
```

Naming: `results/<experiment>_run.log` and `results/<experiment>_curve.png`.
For long/detached runs use `setsid nohup … &` so a disconnect does not kill it.

## 2. Index → `docs/exp_logs/LOG.md`

After a run finishes, add **one row** to the table in `docs/exp_logs/LOG.md`:

```
| 날짜 | 실험 | 컨셉 (한 줄) | 결과 (핵심 수치/판정) | 산출물 |
```

Rules:
- **날짜**: absolute `YYYY-MM-DD` (the run date), newest at the bottom.
- **결과**: the decisive number(s) + a verdict tag — `✅ positive` / `🟡 directional` / `🔴 negative`. Negatives are kept, not deleted (they justify design choices).
- **산출물**: the experiment module + result files (`results/<...>.png`, `.log`).
- One row per *run you would cite*; ad-hoc dry-runs don't need a row.

## 3. Reproducibility checklist (before adding a row)

- [ ] fixed `--seed`; config (n, m, Q, T, steps) recorded in the log header.
- [ ] per-config threshold / tau calibrated fairly (not on one favored config).
- [ ] figure saved to `results/`; run log saved to `results/`.
- [ ] verdict is honest (report `directional`/`negative` when it is).

## 4. GPU etiquette (shared server)

- Ask before putting work on a GPU; check `nvidia-smi` and pick a free device.
- Keep footprint small; prefer CPU for these tiny models when GPUs are busy.
- Record the device used in the run log header (the scripts print `device=…`).
