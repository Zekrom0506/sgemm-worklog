#!/usr/bin/env bash
# Profile one launch of every kernel at 4096 with Nsight Compute -> results/ncu/*.csv
# Also writes results/resource_usage.txt (registers / smem / spills from ptxas),
# which works even where ncu does not.
set -uo pipefail
cd "$(dirname "$0")/.."

SIZE=${SIZE:-4096}
ARCH=${ARCH:-sm_75}
mkdir -p results/ncu

make -s ARCH="$ARCH" || exit 1

# ---------------------------------------------------------------- ptxas report
NVCC=${NVCC:-nvcc}
$NVCC -O3 -std=c++17 -arch="$ARCH" -Xptxas -v -c src/main.cu -o /dev/null 2> results/ptxas_raw.txt
{
  echo "# ptxas resource usage, arch=$ARCH ($($NVCC --version | tail -n 1))"
  echo "kernel | registers | smem_bytes | spill_store_bytes"
  current=""
  while IFS= read -r line; do
    if [[ "$line" =~ Compiling\ entry\ function\ \'([^\']+)\' ]]; then
      current=$(echo "${BASH_REMATCH[1]}" | cu++filt 2>/dev/null || echo "${BASH_REMATCH[1]}")
      # "void sgemm_autotuned<(int)64, ...>(int, int, ...)" -> "sgemm_autotuned<64,...>"
      current=$(echo "$current" | sed -E 's/^void //; s/\(int\)//g; s/ //g; s/\(.*\)$//')
    elif [[ "$line" =~ spill ]]; then
      spills=$(echo "$line" | grep -oE '[0-9]+ bytes spill stores' | grep -oE '^[0-9]+')
    elif [[ "$line" =~ Used\ ([0-9]+)\ registers ]]; then
      regs=${BASH_REMATCH[1]}
      smem=$(echo "$line" | grep -oE '[0-9]+ bytes smem' | grep -oE '^[0-9]+' || echo 0)
      if [[ "$current" == sgemm_* ]]; then
        echo "$current | $regs | $smem | $spills"
      fi
    fi
  done < results/ptxas_raw.txt
} > results/resource_usage.txt
echo "wrote results/resource_usage.txt"

# ---------------------------------------------------------------- Nsight Compute
NCU=$(command -v ncu || true)
if [[ -z "$NCU" && -x /usr/local/cuda/bin/ncu ]]; then
  NCU=/usr/local/cuda/bin/ncu
fi
if [[ -z "$NCU" ]]; then
  echo "ncu not found; skipping Nsight Compute profiling."
  exit 0
fi

METRICS="gpu__time_duration.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
smsp__inst_executed.sum,\
launch__registers_per_thread,\
launch__occupancy_limit_registers"

# Kernel name filter for each kernel id. cuBLAS picks its own kernel, so match loosely.
kernel_filter() {
  case $1 in
    0) echo "regex:gemm" ;;
    1) echo "regex:sgemm_naive" ;;
    2) echo "regex:sgemm_coalesced" ;;
    3) echo "regex:sgemm_shared_mem" ;;
    4) echo "regex:sgemm_1d_blocktiling" ;;
    5) echo "regex:sgemm_2d_blocktiling" ;;
    6) echo "regex:sgemm_vectorized" ;;
    7) echo "regex:sgemm_autotuned" ;;
    8) echo "regex:sgemm_wmma_fp16" ;;
  esac
}
kernel_name() {
  case $1 in
    0) echo cublas ;; 1) echo naive ;; 2) echo coalesced ;; 3) echo shared_mem ;;
    4) echo 1d_blocktiling ;; 5) echo 2d_blocktiling ;; 6) echo vectorized ;;
    7) echo autotuned ;; 8) echo wmma_fp16 ;;
  esac
}

# Returns 0 if ncu works, 1 on a permissions error. Sets PREFIX to "" or "sudo".
PREFIX=""
probe_ncu() {
  local log=results/ncu/probe.log
  $1 "$NCU" --kernel-name regex:sgemm_naive --launch-count 1 --metrics gpu__time_duration.sum \
    ./bin/sgemm_bench --kernel 1 --size 256 > "$log" 2>&1
  ! grep -qE "ERR_NVGPUCTRPERM|permission|not have permission" "$log"
}

if probe_ncu ""; then
  PREFIX=""
elif command -v sudo > /dev/null && sudo -n true 2> /dev/null && probe_ncu "sudo -n"; then
  PREFIX="sudo -n"
else
  cat <<'EOF'

======================================================================
 Nsight Compute cannot read GPU performance counters here
 (ERR_NVGPUCTRPERM). This is common on Colab.

 Fallback: rent a Linux GPU for ~30 minutes (RunPod, Vast.ai, Lambda),
 ideally a T4, then:
     git clone <repo> && cd sgemm-worklog
     make ARCH=sm_75          # or the arch of the rented GPU
     sudo bash scripts/profile.sh
 and copy results/ncu/*.csv back into this repo.

 Everything else (timings, autotune, resource_usage.txt, plots other
 than the roofline) still works without ncu.
======================================================================
EOF
  exit 0
fi

echo "profiling with: ${PREFIX:+$PREFIX }$NCU"
for kernel in 0 1 2 3 4 5 6 7 8; do
  name=$(kernel_name "$kernel")
  report="results/ncu/k${kernel}_${name}"
  echo "  kernel $kernel ($name)"
  # The harness launches each kernel once for the correctness check and then
  # 3 warm-up runs; skip those and profile the first steady-state launch.
  # (For kernel 0 the first launch is the cuBLAS reference, so the count is the same.)
  $PREFIX "$NCU" --kernel-name "$(kernel_filter "$kernel")" \
    --launch-skip 4 --launch-count 1 \
    --metrics "$METRICS" --section WarpStateStats \
    --force-overwrite -o "$report" \
    ./bin/sgemm_bench --kernel "$kernel" --size "$SIZE" > "$report.log" 2>&1 \
    || echo "    ncu failed, see $report.log"
  if [[ -f "$report.ncu-rep" ]]; then
    $PREFIX "$NCU" --import "$report.ncu-rep" --csv --page raw --print-units base > "$report.csv"
  fi
done
echo "wrote results/ncu/*.csv"
