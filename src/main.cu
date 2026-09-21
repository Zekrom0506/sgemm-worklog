#include <cstring>
#include <string>
#include <vector>

#include "runner.cuh"

const unsigned int SEED_A = 1234;
const unsigned int SEED_B = 5678;

struct Options {
    std::vector<int> kernels;
    std::vector<int> sizes;
    std::string csvPath;
};

void printUsage(const char* program) {
    printf("Usage: %s [--kernel N|all] [--size S | --sweep] [--csv path]\n", program);
    printf("  kernels: 0 = cuBLAS, 1-7 = FP32 kernels\n");
}

Options parseArgs(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        bool hasValue = (i + 1 < argc);
        if (arg == "--kernel" && hasValue) {
            std::string value = argv[++i];
            if (value == "all") {
                for (int k = 0; k <= NUM_FP32_KERNELS; k++) {
                    options.kernels.push_back(k);
                }
            } else {
                options.kernels.push_back(std::stoi(value));
            }
        } else if (arg == "--size" && hasValue) {
            options.sizes.push_back(std::stoi(argv[++i]));
        } else if (arg == "--sweep") {
            options.sizes = {128, 256, 512, 1024, 2048, 4096};
        } else if (arg == "--csv" && hasValue) {
            options.csvPath = argv[++i];
        } else {
            printUsage(argv[0]);
            exit(arg == "--help" ? EXIT_SUCCESS : EXIT_FAILURE);
        }
    }
    if (options.kernels.empty()) options.kernels.push_back(0);
    if (options.sizes.empty()) options.sizes.push_back(4096);
    return options;
}

FILE* openCsv(const std::string& path, const char* header) {
    if (path.empty()) return nullptr;
    FILE* file = fopen(path.c_str(), "a+");
    if (file == nullptr) {
        fprintf(stderr, "Could not open %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    if (ftell(file) == 0) {
        fprintf(file, "%s\n", header);
    }
    return file;
}

void printRow(int kernel, const char* name, int size, float ms, double gflopsValue,
              double pctCublas, float error) {
    printf("%-3d %-20s %6d %11.3f %10.1f %9.1f%% %12.2e\n",
           kernel, name, size, ms, gflopsValue, pctCublas, error);
}

void writeRow(FILE* csv, int kernel, const char* name, int size, float ms,
              double gflopsValue, double pctCublas, float error) {
    if (csv == nullptr) return;
    fprintf(csv, "%d,%s,%d,%.4f,%.2f,%.2f,%.3e\n",
            kernel, name, size, ms, gflopsValue, pctCublas, error);
    fflush(csv);
}

int main(int argc, char** argv) {
    Options options = parseArgs(argc, argv);
    printDeviceInfo();

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    FILE* csv = openCsv(options.csvPath, "kernel,name,size,median_ms,gflops,pct_cublas,max_abs_error");

    printf("\n%-3s %-20s %6s %11s %10s %10s %12s\n",
           "id", "kernel", "size", "median_ms", "GFLOPS", "% cuBLAS", "max_abs_err");

    for (int size : options.sizes) {
        if (size % 128 != 0) {
            fprintf(stderr, "Size %d is not a multiple of 128; the kernels assume it is.\n", size);
            return EXIT_FAILURE;
        }
        int M = size, N = size, K = size;
        float alpha = 1.0f, beta = 0.0f;
        size_t bytes = (size_t)size * size * sizeof(float);

        std::vector<float> hostA(size * size), hostB(size * size);
        fillRandom(hostA, SEED_A);
        fillRandom(hostB, SEED_B);

        float *deviceA, *deviceB, *deviceC;
        CUDA_CHECK(cudaMalloc(&deviceA, bytes));
        CUDA_CHECK(cudaMalloc(&deviceB, bytes));
        CUDA_CHECK(cudaMalloc(&deviceC, bytes));
        CUDA_CHECK(cudaMemcpy(deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice));

        float cublasMs = timeKernelMs([&]() {
            runCublas(handle, M, N, K, alpha, deviceA, deviceB, beta, deviceC);
        });
        double cublasGflops = gflops(M, N, K, cublasMs);

        for (int kernel : options.kernels) {
            if (kernel == 0) {
                printRow(0, kernelName(0), size, cublasMs, cublasGflops, 100.0, 0.0f);
                writeRow(csv, 0, kernelName(0), size, cublasMs, cublasGflops, 100.0, 0.0f);
                continue;
            }

            printf("%-3d %-20s  not implemented yet\n", kernel, kernelName(kernel));
        }

        CUDA_CHECK(cudaFree(deviceA));
        CUDA_CHECK(cudaFree(deviceB));
        CUDA_CHECK(cudaFree(deviceC));
    }

    if (csv != nullptr) fclose(csv);
    CUBLAS_CHECK(cublasDestroy(handle));

    return EXIT_SUCCESS;
}
