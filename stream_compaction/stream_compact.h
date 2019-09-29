#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define blockSize 512

// for reducing bank conflicts
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

inline int ilog2(int x) {
	int lg = 0;
	while (x >>= 1) {
		++lg;
	}
	return lg;
}

inline int ilog2ceil(int x) {
	return x == 1 ? 0 : ilog2(x - 1) + 1;
}

__global__ void kernMapToBoolean(int n, int *bools, const int *idata);

__global__ void kernScatter(int n, int *odata, const int *idata, const int *bools, const int *indices);

__global__ void kernScanDataShared(int n, int* in, int* out, int* sums);

__global__ void kernScanDataUpSweep(int n, int offset1, int offset2, int* buff);
__global__ void kernScanDataDownSweep(int n, int offset1, int offset2, int* buff);

__global__ void kernScanDataShared(int n, int* in, int* out, int* sums);

__global__ void kernStitch(int n, int* in, int* sums);

void gpuScan(int n, int* dev_out, int* dev_in);
int intCompact(int n, int* dev_out, int* dev_in, int *dev_in2);

void scanTest();