#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec >> fig34_fig910_60db_pilot.stdout.log 2>&1
EXCLUDE=(--exclude-datasets dblp foursquare)
LAST_PID=""

launch_fig34() {
  local channel="$1" variant="$2" generator="$3"
  local output="results_fig34_${channel}_60db_${variant}_${generator}"
  local cmd=(conda run --no-capture-output -n SPR-SC python run_full_channel_benchmark.py
    --workers 1 --output "$output" --channel "$channel" --snrs 60
    "${EXCLUDE[@]}" --features count --enable-candidate-metrics
    --enable-storage-metrics --enable-runtime-metrics --export-metrics --seed 123)
  [[ "$variant" == "soft" ]] && cmd+=(--soft-reliability)
  [[ "$generator" != "shyre" ]] && cmd+=(--candidate-generator "$generator")
  echo "$(date --iso-8601=seconds) FIG34_START channel=$channel variant=$variant generator=$generator"
  "${cmd[@]}" &
  LAST_PID=$!
}

for channel in awgn rayleigh; do
  pids=()
  for variant in count soft; do
    for generator in shyre strict_max_clique; do
      launch_fig34 "$channel" "$variant" "$generator"
      pids+=("$LAST_PID")
    done
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  echo "$(date --iso-8601=seconds) FIG34_${channel}_COMPLETE"
done

launch_alpha() {
  local variant="$1" alpha="$2"
  local tag="${alpha/./p}"
  local output="results_fig910_awgn_60db_${variant}_alpha${tag}"
  local cmd=(conda run --no-capture-output -n SPR-SC python run_full_channel_benchmark.py
    --workers 1 --output "$output" --channel awgn --snrs 60
    "${EXCLUDE[@]}" --features count --projection-retention "$alpha"
    --enable-storage-metrics --enable-runtime-metrics --export-metrics --seed 123)
  [[ "$variant" == "soft" ]] && cmd+=(--soft-reliability)
  echo "$(date --iso-8601=seconds) FIG910_START variant=$variant alpha=$alpha"
  "${cmd[@]}" &
  LAST_PID=$!
}

pids=()
for variant in count soft; do
  for alpha in 0.2 0.4 0.6 0.8 1.0; do
    launch_alpha "$variant" "$alpha"
    pids+=("$LAST_PID")
    if (( ${#pids[@]} >= 4 )); then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
    fi
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "$(date --iso-8601=seconds) ALL_COMPLETE"
