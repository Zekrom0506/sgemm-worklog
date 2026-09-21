# sgemm-worklog

Step-by-step FP32 matrix multiplication (SGEMM) in CUDA on a Tesla T4, from a
naive kernel towards cuBLAS performance. Work in progress.

Build and run (needs an NVIDIA GPU, e.g. Colab T4):

```bash
make ARCH=sm_75
./bin/sgemm_bench --kernel 0 --sweep
```

Plan: naive -> coalescing -> shared memory -> 1D/2D block tiling -> vectorized
loads -> autotuning, with a correctness check against cuBLAS at every step.
