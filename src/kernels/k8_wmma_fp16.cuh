#pragma once

#include <cuda_fp16.h>
#include <mma.h>

// Tensor-core GEMM: A and B are FP16, accumulation and C are FP32.
// This is NOT comparable to kernels 1-7 (different input precision); it is
// benchmarked against cublasGemmEx with the same data types.
//
// Block = 4 warps (128 threads) arranged 2x2, block tile = 64x64.
// Each warp owns a 32x32 piece of C = 2x2 WMMA fragments of 16x16.
__global__ void __launch_bounds__(128)
sgemm_wmma_fp16(int M, int N, int K, float alpha,
                const half* A, const half* B,
                float beta, float* C) {
    using namespace nvcuda;
    const int WMMA_SIZE = 16;
    const int WARP_TILE = 32;
    const int BLOCK_TILE = 64;
    const int FRAGS = WARP_TILE / WMMA_SIZE;

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / 2;
    int warpCol = warpId % 2;
    int tileRow = blockIdx.y * BLOCK_TILE + warpRow * WARP_TILE;
    int tileCol = blockIdx.x * BLOCK_TILE + warpCol * WARP_TILE;

    // A fragment is distributed across the 32 threads of a warp; the layout of
    // which thread holds which element is opaque, so all loads/stores and the
    // multiply go through the warp-wide *_sync calls.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag[FRAGS];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag[FRAGS];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accFrag[FRAGS][FRAGS];

    for (int i = 0; i < FRAGS; i++) {
        for (int j = 0; j < FRAGS; j++) {
            wmma::fill_fragment(accFrag[i][j], 0.0f);
        }
    }

    for (int k = 0; k < K; k += WMMA_SIZE) {
        for (int i = 0; i < FRAGS; i++) {
            const half* aTile = A + (tileRow + i * WMMA_SIZE) * K + k;
            wmma::load_matrix_sync(aFrag[i], aTile, K);
        }
        for (int j = 0; j < FRAGS; j++) {
            const half* bTile = B + k * N + tileCol + j * WMMA_SIZE;
            wmma::load_matrix_sync(bFrag[j], bTile, N);
        }
        for (int i = 0; i < FRAGS; i++) {
            for (int j = 0; j < FRAGS; j++) {
                wmma::mma_sync(accFrag[i][j], aFrag[i], bFrag[j], accFrag[i][j]);
            }
        }
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;
    for (int i = 0; i < FRAGS; i++) {
        for (int j = 0; j < FRAGS; j++) {
            float* cTile = C + (tileRow + i * WMMA_SIZE) * N + tileCol + j * WMMA_SIZE;
            wmma::load_matrix_sync(cFrag, cTile, N, wmma::mem_row_major);
            // accFrag and cFrag have the same layout, so element t matches element t.
            for (int t = 0; t < cFrag.num_elements; t++) {
                cFrag.x[t] = alpha * accFrag[i][j].x[t] + beta * cFrag.x[t];
            }
            wmma::store_matrix_sync(cTile, cFrag, N, wmma::mem_row_major);
        }
    }
}
