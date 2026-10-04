#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/tushurui/SPR-SC_20260907/06_REPRODUCTION_CODE
PY=/home/tushurui/miniconda3/envs/SPR-SC/bin/python
WORKERS=${ABLATION_WORKERS:-2}
SEEDS=(123 124 125)
STANDARD=(school school2 directors fb_auto)
FAST=(jf17k m_fb15k wikipeople)
STATUS="$ROOT/v1_1_ablation_all_status.log"

cd "$ROOT"

write_status() {
  "$PY" - <<'PY' > "$STATUS.tmp"
import json
from pathlib import Path
from datetime import datetime

root = Path('/home/tushurui/SPR-SC_20260907/06_REPRODUCTION_CODE')
configs = ('base', 'p1', 'p1_p2', 'p1_p3', 'p1_p2_p3', 'p1_p4', 'p1_p5_lr', 'p1_upsample')
datasets = ('enron', 'crime', 'school', 'school2', 'directors', 'fb_auto', 'jf17k', 'm_fb15k', 'wikipeople')
fast = {'jf17k', 'm_fb15k', 'wikipeople'}
print('更新时间:', datetime.now().astimezone().isoformat(timespec='seconds'))
print('协议: AWGN 20 dB，3 seeds；DBLP/Hosts-Virus/Foursquare 不在本轮消融中')
print('注意: fast 数据集没有 shyre_fast_channel_aware，P2 在其上记为 N/A/no-op')
print()
total = 0
for dataset in datasets:
    cells = 0
    for config in configs:
        directory = root / f'results_v1_1_ablation_awgn20_{config}'
        for path in directory.glob(f'{dataset}_awgn_20dB_seed*.metrics.jsonl'):
            try:
                payload = json.loads(path.read_text(encoding='utf-8').splitlines()[0])
                if 'SHyRe' in payload['outcome']['performance']:
                    cells += 1
            except Exception:
                pass
    total += cells
    expected = 18 if dataset in fast else 24
    print(f'{dataset:<12} {cells:>2}/{expected}')
print(f'\n总进度: {total}/198')
PY
  mv "$STATUS.tmp" "$STATUS"
}

monitor() {
  while true; do
    write_status
    sleep 30
  done
}

common=(--channel awgn --snrs 20 --seeds "${SEEDS[@]}" --workers "$WORKERS"
        --features count --train-channel-matched --enable-candidate-metrics
        --enable-runtime-metrics --export-metrics)

run_standard() {
  local config=$1 generator=$2
  shift 2
  local extra=("$@")
  local output="results_v1_1_ablation_awgn20_${config}"
  "$PY" run_full_channel_benchmark.py "${common[@]}" \
    --output "$output" --datasets "${STANDARD[@]}" \
    --candidate-generator "$generator" "${extra[@]}"
}

run_fast() {
  local config=$1
  shift
  local extra=("$@")
  local output="results_v1_1_ablation_awgn20_${config}"
  "$PY" run_full_channel_benchmark.py "${common[@]}" \
    --output "$output" --datasets "${FAST[@]}" \
    --candidate-generator shyre_fast "${extra[@]}"
}

monitor &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true; write_status' EXIT

# Finish every standard dataset first so the large imported datasets do not
# delay the immediately useful comparison.
run_standard base shyre
run_standard p1 shyre --decision-threshold-mode validation
run_standard p1_p2 shyre_channel_aware --decision-threshold-mode validation
run_standard p1_p3 shyre --decision-threshold-mode validation --train-snr-offsets -2 0 2
run_standard p1_p2_p3 shyre_channel_aware --decision-threshold-mode validation --train-snr-offsets -2 0 2
run_standard p1_p4 shyre --decision-threshold-mode validation \
  --soft-reliability --soft-reliability-mode distribution
run_standard p1_p5_lr shyre --decision-threshold-mode validation --model lr
run_standard p1_upsample shyre --decision-threshold-mode validation --class-balance upsample

# P2 is intentionally absent here: no shyre_fast_channel_aware implementation
# exists, so repeating P1 under a P2 label would not be a valid ablation.
run_fast base
run_fast p1 --decision-threshold-mode validation
run_fast p1_p3 --decision-threshold-mode validation --train-snr-offsets -2 0 2
run_fast p1_p4 --decision-threshold-mode validation \
  --soft-reliability --soft-reliability-mode distribution
run_fast p1_p5_lr --decision-threshold-mode validation --model lr
run_fast p1_upsample --decision-threshold-mode validation --class-balance upsample

write_status
"$PY" plot_v1_1_optimal_results.py
echo "$(date --iso-8601=seconds) COMPLETE" >> "$STATUS"
