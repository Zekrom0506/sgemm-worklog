ARCH ?= sm_75
NVCC ?= nvcc

NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo
LDLIBS := -lcublas

SOURCES := src/main.cu $(wildcard src/*.cuh) $(wildcard src/kernels/*.cuh)

.PHONY: all clean

all: bin/sgemm_bench

bin/sgemm_bench: $(SOURCES)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) -o $@ src/main.cu $(LDLIBS)

clean:
	rm -rf bin
