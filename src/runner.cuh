#pragma once

#include "utils.cuh"

const int NUM_FP32_KERNELS = 0;

void runCublas(cublasHandle_t handle, int M, int N, int K, float alpha,
               const float* A, const float* B, float beta, float* C) {
    // cuBLAS is column-major. A row-major matrix read as column-major is its
    // transpose, so we ask cuBLAS for C^T = B^T * A^T: pass B first, swap M and N,
    // and the column-major C^T it writes is exactly our row-major C.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha, B, N, A, K, &beta, C, N));
}

const char* kernelName(int kernel) {
    switch (kernel) {
        case 0: return "cublas";
        default: return "unknown";
    }
}

// Runs one of the FP32 kernels (only cuBLAS so far).
void runKernel(int kernel, cublasHandle_t handle, int M, int N, int K, float alpha,
               const float* A, const float* B, float beta, float* C) {
    switch (kernel) {
        case 0: runCublas(handle, M, N, K, alpha, A, B, beta, C); break;
        default:
            fprintf(stderr, "Unknown FP32 kernel %d\n", kernel);
            exit(EXIT_FAILURE);
    }
}
