#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                     \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = (call);                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)status,        \
                    __FILE__, __LINE__);                                      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

inline void fillRandom(std::vector<float>& data, unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = distribution(generator);
    }
}

inline float maxAbsError(const std::vector<float>& result,
                         const std::vector<float>& reference) {
    float maxError = 0.0f;
    for (size_t i = 0; i < result.size(); i++) {
        float error = std::fabs(result[i] - reference[i]);
        maxError = std::max(maxError, error);
    }
    return maxError;
}

inline float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return 0.5f * (values[middle - 1] + values[middle]);
    }
    return values[middle];
}

// Times each launch separately with CUDA events and returns the median in ms.
// The median is less sensitive than the mean to the T4's power throttling.
inline float timeKernelMs(const std::function<void()>& launch,
                          int warmupRuns = 3, int timedRuns = 20) {
    for (int i = 0; i < warmupRuns; i++) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> timesMs;
    for (int i = 0; i < timedRuns; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
        timesMs.push_back(elapsedMs);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return median(timesMs);
}

inline double gflops(int M, int N, int K, float milliseconds) {
    double flops = 2.0 * M * N * K;
    return flops / (milliseconds * 1e-3) / 1e9;
}

inline void printDeviceInfo() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    int runtimeVersion = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    printf("Device: %s (sm_%d%d), %d SMs, %.0f KB smem/block, CUDA runtime %d.%d\n",
           props.name, props.major, props.minor, props.multiProcessorCount,
           props.sharedMemPerBlock / 1024.0, runtimeVersion / 1000,
           (runtimeVersion % 1000) / 10);
}
