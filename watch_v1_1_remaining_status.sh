#!/usr/bin/env bash
set -u

ROOT=/home/tushurui/SPR-SC_20260907/06_REPRODUCTION_CODE
STATUS="$ROOT/v1_1_remaining_optimal_status.log"

count_success() {
  local csv=$1
  if [[ ! -f "$csv" ]]; then
    printf '0'
    return
  fi
  awk -F, 'NR>1 && $7==0 && $5=="SHyRe" {n++} END{print n+0}' "$csv"
}

write_status() {
  local temporary="$STATUS.tmp"
  local total=0
  local expected=0
  {
    printf '更新时间: %s\n' "$(date --iso-8601=seconds)"
    printf '协议: SHyRe-fast，3 seeds，AWGN/Rayleigh；DBLP 与其他数据集均为 0:2:20 dB\n\n'
    printf '%-14s %-14s %-14s\n' '数据集' 'AWGN' 'Rayleigh'
    for dataset in m_fb15k wikipeople dblp; do
      awgn=$(count_success "$ROOT/results_v1_1_optimal_${dataset}_fast_awgn_0_20/snr_baselines.csv")
      rayleigh=$(count_success "$ROOT/results_v1_1_optimal_${dataset}_fast_rayleigh_0_20/snr_baselines.csv")
      printf '%-14s %2d/33          %2d/33\n' "$dataset" "$awgn" "$rayleigh"
      total=$((total + awgn + rayleigh))
      expected=$((expected + 66))
    done
    printf '\n总进度: %d/%d 实验单元（%.1f%%）\n' "$total" "$expected" "$(awk -v n="$total" -v d="$expected" 'BEGIN {print 100*n/d}')"
    if [[ -f "$ROOT/v1_1_remaining_optimal_complete.txt" ]]; then
      printf '状态: 全部完成\n'
      cat "$ROOT/v1_1_remaining_optimal_complete.txt"
    elif [[ "$total" -ge "$expected" ]]; then
      printf '状态: 实验已齐，等待自动绘图/完成标记\n'
    else
      printf '状态: 运行中或等待恢复\n'
    fi
    printf '\n内存:\n'
    free -h | sed -n '1,2p'
  } > "$temporary"
  mv "$temporary" "$STATUS"
}

while true; do
  write_status
  [[ -f "$ROOT/v1_1_remaining_optimal_complete.txt" ]] && break
  sleep 30
done
