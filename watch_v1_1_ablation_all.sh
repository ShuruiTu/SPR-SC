#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/tushurui/SPR-SC_20260907/06_REPRODUCTION_CODE
while true; do
  clear 2>/dev/null || true
  cat "$ROOT/v1_1_ablation_all_status.log" 2>/dev/null || echo "等待状态文件生成……"
  sleep 10
done
