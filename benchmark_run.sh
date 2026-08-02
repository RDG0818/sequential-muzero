#!/usr/bin/env bash
# Framework-agnostic throughput/memory sampler.
# Wraps any training command, polls nvidia-smi + free every 1s while it runs,
# prints peak/avg GPU VRAM, GPU util%, host RAM. Works identically for JAX or PyTorch
# since it samples the driver/OS, not the process.
#
# Usage:
#   ./benchmark_run.sh <label> <command...>
# Examples:
#   ./benchmark_run.sh jax_smax python train.py model=smax mcts=default train.batch_size=256
#   ./benchmark_run.sh mazero_smac python main.py --opr train_sync --case smac --env_name 3m \
#       --num_simulations 100 --sampled_action_times 10 --batch_size 256 ...
#
# Ctrl-C stops both the sampler and the wrapped command cleanly and still prints the summary
# from whatever samples were collected.

set -uo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <label> <command...>" >&2
  exit 1
fi

LABEL="$1"; shift
OUTDIR="./bench_${LABEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
SAMPLE_LOG="$OUTDIR/samples.csv"
CMD_LOG="$OUTDIR/command.log"
SUMMARY="$OUTDIR/summary.txt"

echo "timestamp,gpu_mem_used_mb,gpu_util_pct,host_ram_used_mb" > "$SAMPLE_LOG"

sample() {
  while true; do
    ts=$(date +%s.%N)
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1)
    gpu_mem=$(echo "$gpu" | cut -d',' -f1 | tr -d ' ')
    gpu_util=$(echo "$gpu" | cut -d',' -f2 | tr -d ' ')
    ram=$(free -m | awk '/Mem:/{print $3}')
    echo "${ts},${gpu_mem:-NA},${gpu_util:-NA},${ram:-NA}" >> "$SAMPLE_LOG"
    sleep 1
  done
}

sample &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null' EXIT INT TERM

echo "[$(date)] label=$LABEL cmd: $*" | tee -a "$CMD_LOG"
START=$(date +%s.%N)

"$@" > >(tee -a "$CMD_LOG") 2> >(tee -a "$CMD_LOG" >&2)
STATUS=$?

END=$(date +%s.%N)
kill "$SAMPLER_PID" 2>/dev/null
trap - EXIT INT TERM

WALL=$(awk "BEGIN{printf \"%.2f\", ${END} - ${START}}")

tail -n +2 "$SAMPLE_LOG" | awk -F',' '
  NF==4 && $2 ~ /^[0-9]+$/ {
    n++; mem_sum+=$2; util_sum+=$3; ram_sum+=$4
    if ($2>mem_max) mem_max=$2
    if ($3>util_max) util_max=$3
    if ($4>ram_max) ram_max=$4
  }
  END {
    if (n>0) {
      printf "samples:          %d\n", n
      printf "gpu_mem_used_mb   peak=%d  avg=%.1f\n", mem_max, mem_sum/n
      printf "gpu_util_pct      peak=%d  avg=%.1f\n", util_max, util_sum/n
      printf "host_ram_used_mb  peak=%d  avg=%.1f\n", ram_max, ram_sum/n
    } else {
      print "no samples collected"
    }
  }' > "${SUMMARY}.tmp"

{
  echo "label:      $LABEL"
  echo "command:    $*"
  echo "exit code:  $STATUS"
  printf "wall time:  %.2f s\n" "$WALL"
  cat "${SUMMARY}.tmp"
} | tee "$SUMMARY"

rm -f "${SUMMARY}.tmp"

echo
echo "raw samples: $SAMPLE_LOG"
echo "command log: $CMD_LOG"
echo "steps/sec: not derived here (log format differs MAZero vs this repo)."
echo "  grep step counts + timestamps out of $CMD_LOG for a steady-state window (skip warmup),"
echo "  divide step delta by wall-clock delta between first/last chosen line."
