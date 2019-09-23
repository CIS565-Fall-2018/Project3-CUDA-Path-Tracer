#define GLM_FORCE_CUDA

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
        // TODO:
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
        */

		__global__ void kernAdvanceScan(int n, int offset, int* a, int* b) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= n) {
				return;
			}

			if (idx >= offset) {
				b[idx] = a[idx - offset] + a[idx];
			}
			else {
				b[idx] = a[idx];
			}
		}

        void scan(int n, int *odata, const int *idata, int *cudaA, int *cudaB) {

			dim3 fullBlocksPerGrid((n + blocksize - 1) / blocksize);

			int kmax = ilog2ceil(n);

			timer().startGpuTimer();

			for (int k = 1; k <= kmax; ++k) {
				// invoke kernel
				int offset = (int)pow(2, k - 1);
				kernAdvanceScan<<<fullBlocksPerGrid, blocksize>>>(n - 1, offset, cudaA, cudaB);
				// pointer swap
				int *temp = cudaA;
				cudaA = cudaB;
				cudaB = temp;
			}

			timer().endGpuTimer();

			cudaMemcpy(odata + 1, cudaA, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
			odata[0] = 0;
        }
    }
}
