#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/tushurui/SPR-SC_20260907/06_REPRODUCTION_CODE
PY=/home/tushurui/miniconda3/envs/SPR-SC/bin/python
cd "$ROOT"

run_curve() {
  local dataset=$1
  local channel=$2
  local workers=$3
  shift 3
  local beta=()
  if [[ "$channel" == "rayleigh" ]]; then
    beta=(--channel-beta 1.5)
  fi
  "$PY" run_full_channel_benchmark.py \
    --workers "$workers" \
    --output "results_v1_1_optimal_${dataset}_fast_${channel}_0_20" \
    --channel "$channel" --datasets "$dataset" --snrs "$@" \
    --seeds 123 124 125 --train-channel-matched \
    --decision-threshold-mode validation --candidate-generator shyre_fast \
    --export-metrics "${beta[@]}"
}

# Stage 1: independent non-DBLP datasets. Foursquare is intentionally excluded.
run_curve m_fb15k awgn 1 0 2 4 6 8 10 12 14 16 18 20 & p1=$!
run_curve m_fb15k rayleigh 1 0 2 4 6 8 10 12 14 16 18 20 & p2=$!
run_curve wikipeople awgn 1 0 2 4 6 8 10 12 14 16 18 20 & p3=$!
run_curve wikipeople rayleigh 1 0 2 4 6 8 10 12 14 16 18 20 & p4=$!
wait "$p1" "$p2" "$p3" "$p4"

# Stage 2: AWGN and Rayleigh run in parallel; each channel remains single-worker.
run_curve dblp awgn 1 0 2 4 6 8 10 12 14 16 18 20 & p1=$!
run_curve dblp rayleigh 1 0 2 4 6 8 10 12 14 16 18 20 & p2=$!
wait "$p1" "$p2"

"$PY" plot_v1_1_optimal_results.py
printf 'COMPLETE %s\n' "$(date --iso-8601=seconds)" > v1_1_remaining_optimal_complete.txt
