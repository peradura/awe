#!/usr/bin/env bash
# Multi-seed sweep of the headline experiments + halting-signal bake-off.
# Produces results/<exp>_s<seed>.{log,png}; aggregate with scripts/aggregate.py.
#
# Usage:  bash scripts/sweep.sh                # seeds 0-4
#         SEEDS="0 1 2" bash scripts/sweep.sh  # custom seeds
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0 1 2 3 4}"

for s in $SEEDS; do
  echo "=== seed $s ==="
  PYTHONPATH=src python3 -m awe.experiments.ablation_rule \
    --steps 3000 --seed "$s" --fig "results/rule_s${s}.png" 2>&1 | tee "results/rule_s${s}.log"
  PYTHONPATH=src python3 -m awe.experiments.ablation_reachp \
    --steps 4000 --seed "$s" --fig "results/reachp_s${s}.png" 2>&1 | tee "results/reachp_s${s}.log"
  PYTHONPATH=src python3 -m awe.experiments.bakeoff --task rule \
    --steps 3000 --seed "$s" --fig "results/bakeoff_rule_s${s}.png" 2>&1 | tee "results/bakeoff_rule_s${s}.log"
  PYTHONPATH=src python3 -m awe.experiments.bakeoff --task reachp \
    --steps 4000 --seed "$s" --fig "results/bakeoff_reachp_s${s}.png" 2>&1 | tee "results/bakeoff_reachp_s${s}.log"
done

python3 scripts/aggregate.py results
echo "Done. Remember to: git add -f results/*_s*.log results/*_s*.png && commit && push"
