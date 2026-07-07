#!/usr/bin/env bash
# Multi-seed sweep of the headline experiments + halting-signal bake-off.
# Ablation runs save checkpoints; bakeoff reuses them (no double training).
# A failing stage is logged to results/sweep_failures.log but does NOT abort
# the rest of the queue (don't forfeit a shared-GPU window to one crash).
#
# Usage:  bash scripts/sweep.sh                # seeds 0-4
#         SEEDS="0 1 2" bash scripts/sweep.sh  # custom seeds
set -uo pipefail
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0 1 2 3 4}"

fail() { echo "[sweep] FAILED: $*" | tee -a results/sweep_failures.log; }

for s in $SEEDS; do
  echo "=== seed $s ==="
  PYTHONPATH=src python3 -m awe.experiments.ablation_rule \
    --steps 3000 --seed "$s" --fig "results/rule_s${s}.png" \
    --ckpt "results/ckpt_rule_s${s}.pt" 2>&1 | tee "results/rule_s${s}.log" \
    || fail "ablation_rule seed $s"
  PYTHONPATH=src python3 -m awe.experiments.ablation_reachp \
    --steps 4000 --seed "$s" --fig "results/reachp_s${s}.png" \
    --ckpt "results/ckpt_reachp_s${s}.pt" 2>&1 | tee "results/reachp_s${s}.log" \
    || fail "ablation_reachp seed $s"
  PYTHONPATH=src python3 -m awe.experiments.bakeoff --task rule --seed "$s" \
    --ckpt "results/ckpt_rule_s${s}.pt" --fig "results/bakeoff_rule_s${s}.png" \
    2>&1 | tee "results/bakeoff_rule_s${s}.log" \
    || fail "bakeoff rule seed $s"
  PYTHONPATH=src python3 -m awe.experiments.bakeoff --task reachp --seed "$s" \
    --ckpt "results/ckpt_reachp_s${s}.pt" --fig "results/bakeoff_reachp_s${s}.png" \
    2>&1 | tee "results/bakeoff_reachp_s${s}.log" \
    || fail "bakeoff reachp seed $s"
done

python3 scripts/aggregate.py results
[ -f results/sweep_failures.log ] && { echo "!! some stages failed:"; cat results/sweep_failures.log; }
echo "Done. Commit logs/figures (NOT the .pt checkpoints):"
echo "  git add -f results/*_s*.log results/*_s*.png && git commit && git push"
