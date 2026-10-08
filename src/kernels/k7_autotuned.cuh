#pragma once

// Kernel 6 with the tile sizes as template parameters, so scripts/autotune.sh
// can sweep them. Tile sizes must be compile-time constants: they size the
// shared-memory arrays and the per-thread register arrays, and they let the
// compiler fully unroll the inner loops.
//
// Requirements (checked on the host in runner.cuh before launching):
//   numThreads = BM*BN / (TM*TN) in [64, 1024]
//   (BM*BK/4) and (BK*BN/4) are multiples of numThreads (float4 loads split evenly)
//   (BM*BK + BK*BN) * 4 bytes <= 48 KB
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_autotuned(int M, int N, int K, float alpha,
                const float* A, const float* B,
                float beta, float* C) {
    const int numThreads = (BM * BN) / (TM * TN);

    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadCol = threadIdx.x % (BN / TN);
    int threadRow = threadIdx.x / (BN / TN);

    // Unlike kernel 6, a tile may hold more float4s than there are threads,
    // so each thread loads several, stepping down by `stride` rows.
    int aLoadRow = threadIdx.x / (BK / 4);
    int aLoadCol = threadIdx.x % (BK / 4);
    int aStride = numThreads / (BK / 4);
    int bLoadRow = threadIdx.x / (BN / 4);
    int bLoadCol = threadIdx.x % (BN / 4);
    int bStride = numThreads / (BN / 4);

    const float* aBlock = A + blockRow * BM * K;
    const float* bBlock = B + blockCol * BN;

    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int tileStart = 0; tileStart < K; tileStart += BK) {
        for (int offset = 0; offset < BM; offset += aStride) {
            float4 aVec = reinterpret_cast<const float4*>(
                &aBlock[(aLoadRow + offset) * K + tileStart + aLoadCol * 4])[0];
            As[aLoadCol * 4 + 0][aLoadRow + offset] = aVec.x;
            As[aLoadCol * 4 + 1][aLoadRow + offset] = aVec.y;
            As[aLoadCol * 4 + 2][aLoadRow + offset] = aVec.z;
            As[aLoadCol * 4 + 3][aLoadRow + offset] = aVec.w;
        }
        for (int offset = 0; offset < BK; offset += bStride) {
            reinterpret_cast<float4*>(&Bs[bLoadRow + offset][bLoadCol * 4])[0] =
                reinterpret_cast<const float4*>(
                    &bBlock[(tileStart + bLoadRow + offset) * N + bLoadCol * 4])[0];
        }
        __syncthreads();

        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i += 4) {
                float4 aValues = reinterpret_cast<const float4*>(&As[k][threadRow * TM + i])[0];
                regM[i + 0] = aValues.x;
                regM[i + 1] = aValues.y;
                regM[i + 2] = aValues.z;
                regM[i + 3] = aValues.w;
            }
            for (int j = 0; j < TN; j += 4) {
                float4 bValues = reinterpret_cast<const float4*>(&Bs[k][threadCol * TN + j])[0];
                regN[j + 0] = bValues.x;
                regN[j + 1] = bValues.y;
                regN[j + 2] = bValues.z;
                regN[j + 3] = bValues.w;
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
        int row = blockRow * BM + threadRow * TM + i;
        for (int j = 0; j < TN; j += 4) {
            int col = blockCol * BN + threadCol * TN + j;
            float4 cVec = reinterpret_cast<float4*>(&C[row * N + col])[0];
            cVec.x = alpha * threadResults[i * TN + j + 0] + beta * cVec.x;
            cVec.y = alpha * threadResults[i * TN + j + 1] + beta * cVec.y;
            cVec.z = alpha * threadResults[i * TN + j + 2] + beta * cVec.z;
            cVec.w = alpha * threadResults[i * TN + j + 3] + beta * cVec.w;
            reinterpret_cast<float4*>(&C[row * N + col])[0] = cVec;
        }
    }
}
