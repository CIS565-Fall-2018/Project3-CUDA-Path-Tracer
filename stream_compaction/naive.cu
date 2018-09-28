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

		__global__ void kernScanNaive(int n, int d, int *odata, int *idata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= n) {
				return;
			}

			int power = 1 << (d - 1);
			if (k >= power) {
				odata[k] = idata[k - power] + idata[k];
			}
			else {
				odata[k] = idata[k];
			}
		}

		__global__ void kernShiftRight(int n, int *odata, int *idata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= n) {
				return;
			}

			odata[k] = (k == 0) ? 0 : idata[k - 1];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int blockSize = 128;
			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
			dim3 threadsPerBlock(blockSize);

			int *dev_odata, *dev_idata;
			int size = n * sizeof(int);

			// allocate the buffers and copy the data
			cudaMalloc((void**)&dev_odata, size);
			checkCUDAError("ERROR: cudaMalloc of dev_odata", __LINE__);
			cudaMalloc((void**)&dev_idata, size);
			checkCUDAError("ERROR: cudaMalloc of dev_idata", __LINE__);
			cudaMemcpy(dev_idata, idata, size, cudaMemcpyHostToDevice);
			checkCUDAError("ERROR: cudaMemcpy idata failed", __LINE__);
			
            timer().startGpuTimer();
			int its = ilog2ceil(n);
			for (int i = 1; i <= its; i++) {
				kernScanNaive << <blocksPerGrid, threadsPerBlock >> > (n, i, dev_odata, dev_idata);
				checkCUDAError("ERROR: naive scan", __LINE__);

				std::swap(dev_odata, dev_idata);
			}

			// convert from inclusive to exclusive
			kernShiftRight << <blocksPerGrid, threadsPerBlock >> > (n, dev_odata, dev_idata);
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, size, cudaMemcpyDeviceToHost);
			checkCUDAError("ERROR: cudaMemcpy of output data", __LINE__);

			cudaFree(dev_odata);
			cudaFree(dev_idata);
        }
    }
}
