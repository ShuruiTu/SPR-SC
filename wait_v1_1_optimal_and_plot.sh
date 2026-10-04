#!/usr/bin/env bash
set -u

ROOT=/home/tushurui/SPR-SC_20260907/06_REPRODUCTION_CODE
PY=/home/tushurui/miniconda3/envs/SPR-SC/bin/python
OUTPUT="$ROOT/figures_v1_1_optimal"
STATUS="$ROOT/v1_1_optimal_auto_status.log"

directories=(
  results_v1_1_optimal_hosts_fast_awgn_0_20_step2
  results_v1_1_optimal_hosts_fast_rayleigh_0_20_step2
  results_v1_1_optimal_jf17k_fast_awgn_0_20_step2
  results_v1_1_optimal_jf17k_fast_rayleigh_0_20_step2
)

while true; do
  complete=1
  summary=""
  for directory in "${directories[@]}"; do
    csv="$ROOT/$directory/snr_baselines.csv"
    count=0
    if [[ -f "$csv" ]]; then
      count=$(awk -F, 'NR>1 && $7==0 && $5=="SHyRe" {n++} END{print n+0}' "$csv")
    fi
    summary+="$directory=$count/33 "
    if [[ "$count" -lt 33 ]]; then
      complete=0
    fi
  done
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$summary" > "$STATUS"
  if [[ "$complete" -eq 1 ]]; then
    break
  fi
  sleep 30
done

cd "$ROOT"
"$PY" plot_v1_1_optimal_results.py
printf 'COMPLETE %s\n' "$(date --iso-8601=seconds)" > "$OUTPUT/AUTO_COMPLETE.txt"
printf '%s plotting complete\n' "$(date --iso-8601=seconds)" >> "$STATUS"
