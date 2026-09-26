#pragma once

// Launched with a 1D block of 1024 threads covering a 32x32 tile of C.
__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float* A, const float* B,
                                float beta, float* C) {
    const int TILE = 32;

    // A warp is 32 consecutive threadIdx.x values. Dividing by 32 gives every
    // thread in the warp the same row, and the remainder gives consecutive
    // columns, so the warp's B loads and C stores hit one contiguous 128-byte line.
    int row = blockIdx.x * TILE + threadIdx.x / TILE;
    int col = blockIdx.y * TILE + threadIdx.x % TILE;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}
