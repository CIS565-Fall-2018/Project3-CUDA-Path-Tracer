#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "device_launch_parameters.h"

#define blockSize 128


namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		__global__ void naiveParallelScan(int n, int k, int *odata, int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;
			if (index >= k) {
				odata[index] = idata[index] + idata[index - k];
			}
			else {
				odata[index] = idata[index];
			}
		}
		__global__ void toExclusive(int n, int *odata, int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;
			if (index > 0) {
				odata[index] = idata[index - 1];
			}
			else {
				odata[index] = 0;
			}
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int* inclusive;
			int* exclusive;
			cudaMalloc((int**)&inclusive, n * sizeof(int));
			cudaMalloc((int**)&exclusive, n * sizeof(int));
			cudaMemcpy(inclusive, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            // TODO
			for (int d = 0; d <= ilog2ceil(n); d++) {
				naiveParallelScan<<< fullBlocksPerGrid, blockSize >>>(n, pow(2.0, d), exclusive, inclusive);
				// ping-pong
				std::swap(exclusive, inclusive);
			}
			toExclusive<<< fullBlocksPerGrid, blockSize >>>(n, exclusive, inclusive);
            timer().endGpuTimer();
			cudaMemcpy(odata, exclusive, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(inclusive);
			cudaFree(exclusive);
        }
    }
}
