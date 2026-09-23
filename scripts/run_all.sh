#!/usr/bin/env bash
# Full benchmark sweep: every kernel at every size -> results/results.csv
set -euo pipefail
cd "$(dirname "$0")/.."

make -s
mkdir -p results
rm -f results/results.csv

./bin/sgemm_bench --kernel all --sweep --csv results/results.csv | tee results/run_all.log
echo "wrote results/results.csv"
