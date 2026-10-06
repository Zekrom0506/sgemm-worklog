#pragma once

// Same tiling as kernel 5 (BM=BN=128, BK=8, TM=TN=8, 256 threads), but global
// loads/stores use float4 and the A tile is stored transposed in shared memory.
__global__ void __launch_bounds__(256)
sgemm_vectorized(int M, int N, int K, float alpha,
                 const float* A, const float* B,
                 float beta, float* C) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    // As is stored [k][row] instead of [row][k]. In the inner loop a thread needs
    // the 8 A values of its rows for a fixed k; transposed, those 8 values are
    // contiguous, so they can be read with two 128-bit loads.
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadCol = threadIdx.x % (BN / TN);
    int threadRow = threadIdx.x / (BN / TN);

    // A tile: 128 rows x 2 float4 = 256 float4s, one per thread.
    int aLoadRow = threadIdx.x / (BK / 4);
    int aLoadCol = threadIdx.x % (BK / 4);
    // B tile: 8 rows x 32 float4 = 256 float4s, one per thread.
    int bLoadRow = threadIdx.x / (BN / 4);
    int bLoadCol = threadIdx.x % (BN / 4);

    const float* aBlock = A + blockRow * BM * K;
    const float* bBlock = B + blockCol * BN;

    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int tileStart = 0; tileStart < K; tileStart += BK) {
        float4 aVec = reinterpret_cast<const float4*>(
            &aBlock[aLoadRow * K + tileStart + aLoadCol * 4])[0];
        As[aLoadCol * 4 + 0][aLoadRow] = aVec.x;
        As[aLoadCol * 4 + 1][aLoadRow] = aVec.y;
        As[aLoadCol * 4 + 2][aLoadRow] = aVec.z;
        As[aLoadCol * 4 + 3][aLoadRow] = aVec.w;

        reinterpret_cast<float4*>(&Bs[bLoadRow][bLoadCol * 4])[0] =
            reinterpret_cast<const float4*>(
                &bBlock[(tileStart + bLoadRow) * N + bLoadCol * 4])[0];
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
