#pragma once

#include "utils.cuh"
#include "kernels/k1_naive.cuh"
#include "kernels/k2_coalesced.cuh"
#include "kernels/k3_shared_mem.cuh"
#include "kernels/k4_1d_blocktiling.cuh"
#include "kernels/k5_2d_blocktiling.cuh"
#include "kernels/k6_vectorized.cuh"
#include "kernels/k7_autotuned.cuh"

const int NUM_FP32_KERNELS = 7;

void runCublas(cublasHandle_t handle, int M, int N, int K, float alpha,
               const float* A, const float* B, float beta, float* C) {
    // cuBLAS is column-major. A row-major matrix read as column-major is its
    // transpose, so we ask cuBLAS for C^T = B^T * A^T: pass B first, swap M and N,
    // and the column-major C^T it writes is exactly our row-major C.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha, B, N, A, K, &beta, C, N));
}

void runNaive(int M, int N, int K, float alpha,
              const float* A, const float* B, float beta, float* C) {
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

void runCoalesced(int M, int N, int K, float alpha,
                  const float* A, const float* B, float beta, float* C) {
    dim3 block(32 * 32);
    dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    sgemm_coalesced<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

void runSharedMem(int M, int N, int K, float alpha,
                  const float* A, const float* B, float beta, float* C) {
    dim3 block(32 * 32);
    dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    sgemm_shared_mem<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

void run1DBlocktiling(int M, int N, int K, float alpha,
                      const float* A, const float* B, float beta, float* C) {
    const int BM = 64, BN = 64, TM = 8;
    dim3 block((BM * BN) / TM);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_1d_blocktiling<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

void run2DBlocktiling(int M, int N, int K, float alpha,
                      const float* A, const float* B, float beta, float* C) {
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_2d_blocktiling<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

void runVectorized(int M, int N, int K, float alpha,
                   const float* A, const float* B, float beta, float* C) {
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_vectorized<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

template <int BM, int BN, int BK, int TM, int TN>
void launchAutotuned(int M, int N, int K, float alpha,
                     const float* A, const float* B, float beta, float* C) {
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_autotuned<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

typedef void (*LaunchFunction)(int, int, int, float, const float*, const float*, float, float*);

struct TileConfig {
    int BM, BN, BK, TM, TN;
    LaunchFunction launch;
};

// Mirrors the constraints listed in k7_autotuned.cuh.
bool isValidTileConfig(int BM, int BN, int BK, int TM, int TN) {
    int numThreads = (BM * BN) / (TM * TN);
    if (numThreads < 64 || numThreads > 1024) return false;
    if ((BM * BK / 4) % numThreads != 0) return false;
    if ((BK * BN / 4) % numThreads != 0) return false;
    if ((BM * BK + BK * BN) * 4 > 48 * 1024) return false;
    return true;
}

// Every valid point of the sweep BM,BN in {64,128}, BK in {8,16,32},
// TM,TN in {4,8}. Each entry is a separate compiled kernel.
const TileConfig AUTOTUNE_CONFIGS[] = {
    { 64,  64,  8, 4, 8, launchAutotuned<64, 64, 8, 4, 8>},
    { 64,  64,  8, 8, 4, launchAutotuned<64, 64, 8, 8, 4>},
    { 64,  64,  8, 8, 8, launchAutotuned<64, 64, 8, 8, 8>},
    { 64,  64, 16, 4, 4, launchAutotuned<64, 64, 16, 4, 4>},
    { 64,  64, 16, 4, 8, launchAutotuned<64, 64, 16, 4, 8>},
    { 64,  64, 16, 8, 4, launchAutotuned<64, 64, 16, 8, 4>},
    { 64,  64, 16, 8, 8, launchAutotuned<64, 64, 16, 8, 8>},
    { 64,  64, 32, 4, 4, launchAutotuned<64, 64, 32, 4, 4>},
    { 64,  64, 32, 4, 8, launchAutotuned<64, 64, 32, 4, 8>},
    { 64,  64, 32, 8, 4, launchAutotuned<64, 64, 32, 8, 4>},
    { 64,  64, 32, 8, 8, launchAutotuned<64, 64, 32, 8, 8>},
    { 64, 128,  8, 8, 8, launchAutotuned<64, 128, 8, 8, 8>},
    { 64, 128, 16, 4, 8, launchAutotuned<64, 128, 16, 4, 8>},
    { 64, 128, 16, 8, 4, launchAutotuned<64, 128, 16, 8, 4>},
    { 64, 128, 16, 8, 8, launchAutotuned<64, 128, 16, 8, 8>},
    { 64, 128, 32, 4, 4, launchAutotuned<64, 128, 32, 4, 4>},
    { 64, 128, 32, 4, 8, launchAutotuned<64, 128, 32, 4, 8>},
    { 64, 128, 32, 8, 4, launchAutotuned<64, 128, 32, 8, 4>},
    { 64, 128, 32, 8, 8, launchAutotuned<64, 128, 32, 8, 8>},
    {128,  64,  8, 8, 8, launchAutotuned<128, 64, 8, 8, 8>},
    {128,  64, 16, 4, 8, launchAutotuned<128, 64, 16, 4, 8>},
    {128,  64, 16, 8, 4, launchAutotuned<128, 64, 16, 8, 4>},
    {128,  64, 16, 8, 8, launchAutotuned<128, 64, 16, 8, 8>},
    {128,  64, 32, 4, 4, launchAutotuned<128, 64, 32, 4, 4>},
    {128,  64, 32, 4, 8, launchAutotuned<128, 64, 32, 4, 8>},
    {128,  64, 32, 8, 4, launchAutotuned<128, 64, 32, 8, 4>},
    {128,  64, 32, 8, 8, launchAutotuned<128, 64, 32, 8, 8>},
    {128, 128,  8, 8, 8, launchAutotuned<128, 128, 8, 8, 8>},
    {128, 128, 16, 4, 8, launchAutotuned<128, 128, 16, 4, 8>},
    {128, 128, 16, 8, 4, launchAutotuned<128, 128, 16, 8, 4>},
    {128, 128, 16, 8, 8, launchAutotuned<128, 128, 16, 8, 8>},
    {128, 128, 32, 4, 4, launchAutotuned<128, 128, 32, 4, 4>},
    {128, 128, 32, 4, 8, launchAutotuned<128, 128, 32, 4, 8>},
    {128, 128, 32, 8, 4, launchAutotuned<128, 128, 32, 8, 4>},
    {128, 128, 32, 8, 8, launchAutotuned<128, 128, 32, 8, 8>},
};
const int NUM_AUTOTUNE_CONFIGS = sizeof(AUTOTUNE_CONFIGS) / sizeof(AUTOTUNE_CONFIGS[0]);

// The official kernel 7. Set this to the fastest row of results/autotune.csv.
const int K7_BM = 128, K7_BN = 128, K7_BK = 8, K7_TM = 8, K7_TN = 8;

const TileConfig* findTileConfig(int BM, int BN, int BK, int TM, int TN) {
    for (int i = 0; i < NUM_AUTOTUNE_CONFIGS; i++) {
        const TileConfig& config = AUTOTUNE_CONFIGS[i];
        if (config.BM == BM && config.BN == BN && config.BK == BK &&
            config.TM == TM && config.TN == TN) {
            return &config;
        }
    }
    return nullptr;
}

void runAutotuned(int M, int N, int K, float alpha,
                  const float* A, const float* B, float beta, float* C) {
    const TileConfig* config = findTileConfig(K7_BM, K7_BN, K7_BK, K7_TM, K7_TN);
    if (config == nullptr) {
        fprintf(stderr, "K7 config %dx%dx%d/%dx%d is not in AUTOTUNE_CONFIGS\n",
                K7_BM, K7_BN, K7_BK, K7_TM, K7_TN);
        exit(EXIT_FAILURE);
    }
    config->launch(M, N, K, alpha, A, B, beta, C);
}

const char* kernelName(int kernel) {
    switch (kernel) {
        case 0: return "cublas";
        case 1: return "naive";
        case 2: return "coalesced";
        case 3: return "shared_mem";
        case 4: return "1d_blocktiling";
        case 5: return "2d_blocktiling";
        case 6: return "vectorized";
        case 7: return "autotuned";
        default: return "unknown";
    }
}

// Runs one of the FP32 kernels (0 = cuBLAS, 1-7 = ours).
void runKernel(int kernel, cublasHandle_t handle, int M, int N, int K, float alpha,
               const float* A, const float* B, float beta, float* C) {
    switch (kernel) {
        case 0: runCublas(handle, M, N, K, alpha, A, B, beta, C); break;
        case 1: runNaive(M, N, K, alpha, A, B, beta, C); break;
        case 2: runCoalesced(M, N, K, alpha, A, B, beta, C); break;
        case 3: runSharedMem(M, N, K, alpha, A, B, beta, C); break;
        case 4: run1DBlocktiling(M, N, K, alpha, A, B, beta, C); break;
        case 5: run2DBlocktiling(M, N, K, alpha, A, B, beta, C); break;
        case 6: runVectorized(M, N, K, alpha, A, B, beta, C); break;
        case 7: runAutotuned(M, N, K, alpha, A, B, beta, C); break;
        default:
            fprintf(stderr, "Unknown FP32 kernel %d\n", kernel);
            exit(EXIT_FAILURE);
    }
}
