#!/usr/bin/env bash
# Sweep kernel 7 tile sizes at one matrix size -> results/autotune.csv
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE=${SIZE:-4096}
OUT=results/autotune.csv

make -s
mkdir -p results
rm -f "$OUT"

tested=0
skipped=0
for BM in 64 128; do
  for BN in 64 128; do
    for BK in 8 16 32; do
      for TM in 4 8; do
        for TN in 4 8; do
          threads=$(( BM * BN / (TM * TN) ))
          smem_bytes=$(( (BM * BK + BK * BN) * 4 ))
          reason=""
          if (( threads < 64 || threads > 1024 )); then
            reason="threads=$threads"
          elif (( (BM * BK / 4) % threads != 0 )); then
            reason="A tile float4s do not split over $threads threads"
          elif (( (BK * BN / 4) % threads != 0 )); then
            reason="B tile float4s do not split over $threads threads"
          elif (( smem_bytes > 48 * 1024 )); then
            reason="smem=${smem_bytes}B"
          fi

          if [[ -n "$reason" ]]; then
            echo "skip  BM=$BM BN=$BN BK=$BK TM=$TM TN=$TN ($reason)"
            skipped=$(( skipped + 1 ))
            continue
          fi
          ./bin/sgemm_bench --autotune "$BM,$BN,$BK,$TM,$TN" --size "$SIZE" --csv "$OUT" \
            | grep -E "^7 |FAILED"
          tested=$(( tested + 1 ))
        done
      done
    done
  done
done

echo
echo "tested $tested configs, skipped $skipped -> $OUT"
echo "fastest configs:"
(head -n 1 "$OUT" && tail -n +2 "$OUT" | sort -t, -k9 -g -r | head -n 5) | column -s, -t
echo
echo "Set K7_BM..K7_TN in src/runner.cuh to the top row, then re-run scripts/run_all.sh."
