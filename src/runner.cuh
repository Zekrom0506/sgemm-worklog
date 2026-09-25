#pragma once

#include "utils.cuh"
#include "kernels/k1_naive.cuh"

const int NUM_FP32_KERNELS = 1;

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

const char* kernelName(int kernel) {
    switch (kernel) {
        case 0: return "cublas";
        case 1: return "naive";
        default: return "unknown";
    }
}

// Runs one of the FP32 kernels (0 = cuBLAS, 1 = ours).
void runKernel(int kernel, cublasHandle_t handle, int M, int N, int K, float alpha,
               const float* A, const float* B, float beta, float* C) {
    switch (kernel) {
        case 0: runCublas(handle, M, N, K, alpha, A, B, beta, C); break;
        case 1: runNaive(M, N, K, alpha, A, B, beta, C); break;
        default:
            fprintf(stderr, "Unknown FP32 kernel %d\n", kernel);
            exit(EXIT_FAILURE);
    }
}
