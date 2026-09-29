#pragma once

// 1D block of 1024 threads, one thread per element of a 32x32 tile of C.
__global__ void sgemm_shared_mem(int M, int N, int K, float alpha,
                                 const float* A, const float* B,
                                 float beta, float* C) {
    const int TILE = 32;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.x / TILE;
    int threadCol = threadIdx.x % TILE;

    int row = blockRow * TILE + threadRow;
    int col = blockCol * TILE + threadCol;

    float sum = 0.0f;
    for (int tileStart = 0; tileStart < K; tileStart += TILE) {
        // Each thread copies one element of each tile. A warp shares threadRow,
        // so both global reads are coalesced along a row.
        As[threadRow][threadCol] = A[row * K + tileStart + threadCol];
        Bs[threadRow][threadCol] = B[(tileStart + threadRow) * N + col];

        // Sync 1: every thread reads elements loaded by other threads, so the
        // whole tile must be in shared memory before anyone starts the dot product.
        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            // As[threadRow][k] is the same address for the whole warp (a broadcast);
            // Bs[k][threadCol] is 32 consecutive floats, one per bank.
            sum += As[threadRow][k] * Bs[k][threadCol];
        }

        // Sync 2: fast threads must not start overwriting the tiles for the next
        // iteration while slower threads are still reading the current ones.
        __syncthreads();
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}
