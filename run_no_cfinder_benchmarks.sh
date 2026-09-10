#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec >> no_cfinder_benchmarks.stdout.log 2>&1

run_small_pair() {
  local channel="$1"
  echo "$(date --iso-8601=seconds) SMALL_${channel}_START"
  conda run --no-capture-output -n SPR-SC python run_full_channel_benchmark.py \
    --workers 2 --output results_soft_count_no_cfinder_small_0_20_step2 \
    --channel "$channel" --snrs 0 2 4 6 8 10 12 14 16 18 20 \
    --exclude-datasets dblp foursquare --features count --soft-reliability &
  local soft_pid=$!
  conda run --no-capture-output -n SPR-SC python run_full_channel_benchmark.py \
    --workers 2 --output results_count_no_cfinder_small_0_20_step2 \
    --channel "$channel" --snrs 0 2 4 6 8 10 12 14 16 18 20 \
    --exclude-datasets dblp foursquare --features count &
  local count_pid=$!
  wait "$soft_pid"
  wait "$count_pid"
  echo "$(date --iso-8601=seconds) SMALL_${channel}_COMPLETE"
}

run_dataset_lane() {
  local dataset="$1" snr_step="$2" soft_output="$3" count_output="$4"
  shift 4
  for channel in awgn rayleigh; do
    echo "$(date --iso-8601=seconds) ${dataset}_${channel}_SOFT_START"
    conda run --no-capture-output -n SPR-SC python run_full_channel_benchmark.py \
      --workers 1 --output "$soft_output" --channel "$channel" --snrs $snr_step \
      --exclude-datasets "$@" --features count --soft-reliability
    echo "$(date --iso-8601=seconds) ${dataset}_${channel}_COUNT_START"
    conda run --no-capture-output -n SPR-SC python run_full_channel_benchmark.py \
      --workers 1 --output "$count_output" --channel "$channel" --snrs $snr_step \
      --exclude-datasets "$@" --features count
  done
  echo "$(date --iso-8601=seconds) ${dataset}_COMPLETE"
}

# Stage A: six smaller datasets.  Two soft + two count workers = four main.py processes.
run_small_pair awgn
run_small_pair rayleigh

# Stage B: exactly two dataset lanes; each lane serially evaluates soft-count then count.
run_dataset_lane foursquare '0 2 4 6 8 10 12 14 16 18 20' \
  results_soft_count_no_cfinder_foursquare_0_20_step2 results_count_no_cfinder_foursquare_0_20_step2 \
  dblp enron hosts school school2 directors crime &
foursquare_pid=$!
run_dataset_lane dblp '0 5 10 15 20' \
  results_soft_count_no_cfinder_dblp_0_20_step5 results_count_no_cfinder_dblp_0_20_step5 \
  enron foursquare hosts school school2 directors crime &
dblp_pid=$!
wait "$foursquare_pid"
wait "$dblp_pid"
echo "$(date --iso-8601=seconds) ALL_COMPLETE"
