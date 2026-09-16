#!/usr/bin/env bash
# Resumable 0:2:20 dB Fig.1/2 rerun with matched channel-corrupted training.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 WORKERS DATASET... -- [shyre|shyre_fast]" >&2
  exit 2
fi
workers="$1"; shift
datasets=()
while [[ $# -gt 0 && "$1" != "--" ]]; do datasets+=("$1"); shift; done
shift
generator="$1"

for family in count soft_count; do
  soft=()
  [[ "$family" == "soft_count" ]] && soft=(--soft-reliability)
  for channel in awgn rayleigh; do
    output="results_fig1_2_matched_train_${family}_${channel}_0_20_step2"
    # A single directory can safely contain all listed datasets because the
    # runner keys resume cells by dataset/channel/SNR/seed.
    conda run --no-capture-output -n SPR-SC python run_full_channel_benchmark.py \
      --datasets "${datasets[@]}" --workers "$workers" --output "$output" \
      --channel "$channel" --snrs 0 2 4 6 8 10 12 14 16 18 20 --features count \
      --candidate-generator "$generator" --train-channel-matched --enable-runtime-metrics --export-metrics "${soft[@]}"
  done
done
