#!/usr/bin/env bash
# Resumable single-seed Fig. 1/2 grid for imported HYPER datasets.
# Exact SHyRe remains selected for FB-AUTO; large sources use the explicitly
# named shyre_fast sampler after their preflight, never a silent downsample.
set -euo pipefail

run_variant() {
  local dataset="$1" generator="$2" variant="$3" channel="$4"
  local output="results_fig1_2_${dataset}_${variant}_${channel}_0_20_step2"
  local soft=()
  if [[ "$variant" == "soft_count" ]]; then soft=(--soft-reliability); fi
  conda run --no-capture-output -n SPR-SC python run_full_channel_benchmark.py \
    --datasets "$dataset" --candidate-generator "$generator" --features count \
    --channel "$channel" --snrs 0 2 4 6 8 10 12 14 16 18 20 --workers 2 \
    --output "$output" --export-metrics --enable-runtime-metrics "${soft[@]}"
}

if [[ $# -ne 2 ]]; then
  echo "usage: $0 DATASET {shyre|shyre_fast}" >&2
  exit 2
fi
dataset="$1"; generator="$2"
for variant in count soft_count; do
  for channel in awgn rayleigh; do
    run_variant "$dataset" "$generator" "$variant" "$channel"
  done
done
