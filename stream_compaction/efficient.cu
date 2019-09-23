#define GLM_FORCE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernUpSweep(int n, int off1, int off2, int* a) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx % off1 != 0 || idx >= n - 1) {
				return;
			}
			a[idx + off1 - 1] += a[idx + off2 - 1];
		}	

		__global__ void kernDownSweep(int n, int off1, int off2, int* a) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx % off1 != 0 || idx >= n - 1) {
				return;
			}
			int t = a[idx + off2 - 1];
			a[idx + off2 - 1] = a[idx + off1 - 1];
			a[idx + off1 - 1] += t;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int npot, int *odata, const int *idata, int *cudaA) {

			dim3 fullBlocksPerGrid((n + blocksize - 1) / blocksize);

			// UPSWEEP	
			int dmax = ilog2ceil(n) - 1;

			timer().startGpuTimer();

			for (int d = 0; d <= dmax; ++d) {
				int off1 = (int)pow(2, d + 1);
				int off2 = (int)pow(2, d);
				kernUpSweep<<<fullBlocksPerGrid, blocksize>>>(n, off1, off2, cudaA);
			}
			int temp[1] = { 0 };
			cudaMemcpy(cudaA + (n - 1), temp, 1 * sizeof(int), cudaMemcpyHostToDevice);
			// DOWNSWEEP
			for (int d = dmax; d >= 0; d--) {
				int off1 = (int)pow(2, d + 1);
				int off2 = (int)pow(2, d);
				kernDownSweep<<<fullBlocksPerGrid, blocksize>>>(n, off1, off2, cudaA);
			}

			timer().endGpuTimer();

			cudaMemcpy(odata, cudaA, npot * sizeof(int), cudaMemcpyDeviceToHost);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
		__global__ void kernConvertToBinary(int n, int *devBinary) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= n) {
				return;
			}
			if (devBinary[idx] != 0) {
				devBinary[idx] = 1;
			}
			else {
				devBinary[idx] = 0;
			}
		}

		__global__ void kernScatter(int n, int *devBinary, int *devBinaryCopy, int *devCopy, int *devResult) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= n) {
				return;
			}
			if (devBinary[idx] == 1) {
				devResult[devBinaryCopy[idx]] = devCopy[idx];
			}
		}

        int compact(int n, int npot, int *odata, const int *idata, int *devCopy, int *devBinary) {
			
			dim3 fullBlocksPerGrid((n + blocksize - 1) / blocksize);

			int *devBinaryCopy;
			int *devResult;

			cudaMalloc((void**)&devBinaryCopy, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc devBinaryCopy");
			cudaMalloc((void**)&devResult, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc devResult");

            timer().startGpuTimer();
			
			// preparing binary array
			kernConvertToBinary<<<fullBlocksPerGrid, blocksize>>>(n, devBinary);
			cudaMemcpy(devBinaryCopy, devBinary, n * sizeof(int), cudaMemcpyDeviceToDevice);

			// running scan
			// UPSWEEP	
			int dmax = ilog2ceil(n) - 1;

			for (int d = 0; d <= dmax; ++d) {
				int off1 = (int)pow(2, d + 1);
				int off2 = (int)pow(2, d);
				kernUpSweep<<<fullBlocksPerGrid, blocksize>>>(n, off1, off2, devBinaryCopy);
			}
			int temp[1] = { 0 };
			cudaMemcpy(devBinaryCopy + (n - 1), temp, 1 * sizeof(int), cudaMemcpyHostToDevice);
			// DOWNSWEEP
			for (int d = dmax; d >= 0; d--) {
				int off1 = (int)pow(2, d + 1);
				int off2 = (int)pow(2, d);
				kernDownSweep<<<fullBlocksPerGrid, blocksize>>>(n, off1, off2, devBinaryCopy);
			}

			// populating compact return array
			kernScatter<<<fullBlocksPerGrid, blocksize>>>(n, devBinary, devBinaryCopy, devCopy, devResult);
			
			timer().endGpuTimer();

			int tempResult[1] = { 0 };
			int offset = npot == n ? n - 1 : npot;
			cudaMemcpy(&tempResult, devBinaryCopy + offset, 1 * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(odata, devResult, npot * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(devBinaryCopy);
			cudaFree(devResult);

            return tempResult[0];
        }
    }
}
