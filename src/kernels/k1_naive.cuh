#pragma once

__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float* A, const float* B,
                            float beta, float* C) {
    // threadIdx.x maps to row on purpose: the 32 threads of a warp read A and
    // write C at addresses a whole row apart (uncoalesced), which kernel 2 fixes.
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}
