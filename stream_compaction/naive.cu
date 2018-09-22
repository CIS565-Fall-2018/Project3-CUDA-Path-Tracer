#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		__global__ void kernNaiveScan(int n, int bound, int *odata, const int *idata) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			if (index >= bound) {
				odata[index] = idata[index - bound] + idata[index];
			}
			else {
				odata[index] = idata[index];
			}
		}

		__global__ void kernInclusiveToExclusive(int n, int *odata, const int *idata) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			odata[index] = index == 0 ? 0 : idata[index - 1];
		}


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
			int blockSize = 1024;
			dim3 threadsPerBlock(blockSize);
			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

			// allocate memory
			int *dev_idata;
			int *dev_odata;

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_idata failed!");
			timer().startGpuTimer();
			int depth = ilog2ceil(n);
			int bound = 1;
			for (int d = 1; d <= depth; d++) {
				kernNaiveScan << <blocksPerGrid, threadsPerBlock >> > (n, bound, dev_odata, dev_idata);
				checkCUDAError("kernNaiveScan failed!");
				std::swap(dev_odata, dev_idata);
				bound *= 2;
			}
			kernInclusiveToExclusive << <blocksPerGrid, threadsPerBlock >> > (n, dev_odata, dev_idata);
			checkCUDAError("kernInclusiveToExclusive failed!");
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_odata failed!");
			timer().endGpuTimer();
			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaDeviceSynchronize();
            
        }

    }
}
