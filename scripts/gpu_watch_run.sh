#!/usr/bin/env bash
# Polite shared-GPU runner: poll nvidia-smi every POLL_SEC (default 600 = 10min)
# and launch the experiment queue only when a GPU is actually idle, pinned to
# that one GPU so nothing else is touched.
#
# Usage:
#   bash scripts/gpu_watch_run.sh          # first idle GPU
#   bash scripts/gpu_watch_run.sh 1        # wait specifically for GPU 1
#
# Tunables (env):
#   POLL_SEC=600      poll interval (s)
#   MEM_MAX_MB=1000   consider idle if memory.used <= this
#   UTIL_MAX=10       ...and utilization.gpu <= this (%)
#   QUEUE="reachp2 sweep"   which stages to run, in order
#   SEEDS="0 1 2 3 4"       forwarded to sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

WANT="${1:-}"
POLL_SEC="${POLL_SEC:-600}"
MEM_MAX_MB="${MEM_MAX_MB:-1000}"
UTIL_MAX="${UTIL_MAX:-10}"
QUEUE="${QUEUE:-reachp2 sweep}"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found — run on the GPU server"; exit 1; }

find_idle_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
  while IFS=', ' read -r idx mem util; do
    [ -n "$WANT" ] && [ "$idx" != "$WANT" ] && continue
    if [ "$mem" -le "$MEM_MAX_MB" ] && [ "$util" -le "$UTIL_MAX" ]; then
      echo "$idx"
      return 0
    fi
  done | head -n1
}

echo "[gpu_watch] waiting for an idle GPU (mem<=${MEM_MAX_MB}MB, util<=${UTIL_MAX}%)," \
     "polling every ${POLL_SEC}s. queue: ${QUEUE}"
while true; do
  GPU="$(find_idle_gpu || true)"
  if [ -n "${GPU:-}" ]; then
    sleep 15                       # brief settle, then re-check (avoid races)
    GPU2="$(find_idle_gpu || true)"
    if [ "$GPU2" = "$GPU" ]; then
      break
    fi
  fi
  echo "[gpu_watch] $(date '+%F %T') no idle GPU — sleeping ${POLL_SEC}s"
  sleep "$POLL_SEC"
done

export CUDA_VISIBLE_DEVICES="$GPU"
echo "[gpu_watch] $(date '+%F %T') claiming GPU ${GPU}; running queue: ${QUEUE}"

for stage in $QUEUE; do
  case "$stage" in
    reachp2)
      PYTHONPATH=src python3 -m awe.experiments.ablation_reachp2 \
        --steps 8000 --fig results/reachp2_curve.png 2>&1 | tee results/reachp2_run.log
      ;;
    sweep)
      bash scripts/sweep.sh
      ;;
    depth_sanity)
      PYTHONPATH=src python3 -m awe.experiments.depth_sanity \
        --steps 3000 2>&1 | tee results/depth_sanity_run.log
      ;;
    *)
      echo "[gpu_watch] unknown stage '$stage' — skipping" ;;
  esac
done

echo "[gpu_watch] queue finished. Commit the results:"
echo "  git add -f results/*.log results/*.png && git commit -m 'Results: GPU runs' && git push"
