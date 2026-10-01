#pragma once

// BM=64, BN=64, BK=8, TM=8 -> (64*64)/8 = 512 threads per block.
// Each thread computes TM=8 outputs stacked vertically in one column of the tile.
__global__ void __launch_bounds__(512)
sgemm_1d_blocktiling(int M, int N, int K, float alpha,
                     const float* A, const float* B,
                     float beta, float* C) {
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // 64 threads per tile row: two warps share a threadRow and cover 64 columns.
    int threadCol = threadIdx.x % BN;
    int threadRow = threadIdx.x / BN;

    // The A tile is 64x8 and the B tile is 8x64: 512 elements each,
    // so every thread loads exactly one element of each.
    int aLoadRow = threadIdx.x / BK;
    int aLoadCol = threadIdx.x % BK;
    int bLoadRow = threadIdx.x / BN;
    int bLoadCol = threadIdx.x % BN;

    const float* aBlock = A + blockRow * BM * K;
    const float* bBlock = B + blockCol * BN;

    float threadResults[TM] = {0.0f};

    for (int tileStart = 0; tileStart < K; tileStart += BK) {
        As[aLoadRow * BK + aLoadCol] = aBlock[aLoadRow * K + tileStart + aLoadCol];
        Bs[bLoadRow * BN + bLoadCol] = bBlock[(tileStart + bLoadRow) * N + bLoadCol];
        __syncthreads();

        for (int k = 0; k < BK; k++) {
            // One B value is kept in a register and reused for TM multiply-adds,
            // so each thread now does 8 FMAs per 9 shared-memory loads instead of 1 per 2.
            float bValue = Bs[k * BN + threadCol];
            for (int i = 0; i < TM; i++) {
                threadResults[i] += As[(threadRow * TM + i) * BK + k] * bValue;
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        int row = blockRow * BM + threadRow * TM + i;
        int col = blockCol * BN + threadCol;
        C[row * N + col] = alpha * threadResults[i] + beta * C[row * N + col];
    }
}
