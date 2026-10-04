ARCH ?= sm_75
NVCC ?= nvcc

NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo
LDLIBS := -lcublas

SOURCES := src/main.cu $(wildcard src/*.cuh) $(wildcard src/kernels/*.cuh)

.PHONY: all clean resources

all: bin/sgemm_bench

bin/sgemm_bench: $(SOURCES)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) -o $@ src/main.cu $(LDLIBS)

# Registers / shared memory / spills per kernel, straight from ptxas.
resources:
	@mkdir -p results
	$(NVCC) $(NVCCFLAGS) -Xptxas -v -c src/main.cu -o /dev/null 2> results/ptxas_raw.txt
	@echo "wrote results/ptxas_raw.txt"

clean:
	rm -rf bin
