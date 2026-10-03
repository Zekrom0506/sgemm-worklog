#pragma once

// BM=128, BN=128, BK=8, TM=8, TN=8 -> (128*128)/(8*8) = 256 threads per block.
// Each thread computes an 8x8 sub-tile of C.
__global__ void __launch_bounds__(256)
sgemm_2d_blocktiling(int M, int N, int K, float alpha,
                     const float* A, const float* B,
                     float beta, float* C) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    const int numThreads = (BM * BN) / (TM * TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // 16 threads across, 16 down; each owns an 8x8 patch.
    int threadCol = threadIdx.x % (BN / TN);
    int threadRow = threadIdx.x / (BN / TN);

    // Each tile has 128*8 = 1024 elements but there are only 256 threads, so
    // each thread loads 4 elements, stepping down by `stride` rows each time.
    int aLoadRow = threadIdx.x / BK;
    int aLoadCol = threadIdx.x % BK;
    int aStride = numThreads / BK;
    int bLoadRow = threadIdx.x / BN;
    int bLoadCol = threadIdx.x % BN;
    int bStride = numThreads / BN;

    const float* aBlock = A + blockRow * BM * K;
    const float* bBlock = B + blockCol * BN;

    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int tileStart = 0; tileStart < K; tileStart += BK) {
        for (int offset = 0; offset < BM; offset += aStride) {
            As[(aLoadRow + offset) * BK + aLoadCol] =
                aBlock[(aLoadRow + offset) * K + tileStart + aLoadCol];
        }
        for (int offset = 0; offset < BK; offset += bStride) {
            Bs[(bLoadRow + offset) * BN + bLoadCol] =
                bBlock[(tileStart + bLoadRow + offset) * N + bLoadCol];
        }
        __syncthreads();

        for (int k = 0; k < BK; k++) {
            // Outer product: a column of 8 A values times a row of 8 B values gives
            // 64 FMAs for 16 shared-memory loads.
            for (int i = 0; i < TM; i++) {
                regM[i] = As[(threadRow * TM + i) * BK + k];
            }
            for (int j = 0; j < TN; j++) {
                regN[j] = Bs[k * BN + threadCol * TN + j];
            }
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    threadResults[i * TN + j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int row = blockRow * BM + threadRow * TM + i;
            int col = blockCol * BN + threadCol * TN + j;
            C[row * N + col] = alpha * threadResults[i * TN + j] + beta * C[row * N + col];
        }
    }
}
