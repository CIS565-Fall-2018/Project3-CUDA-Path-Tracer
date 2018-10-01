#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "device_launch_parameters.h"
#include "../src/scene.h"
#define blockSize 128


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
		__global__ void upSweep(int n, int k, int *dev) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			if ((index % (2 * k) == 0) && (index + (2 * k) <= n))
				dev[index + (2 * k) - 1] += dev[index + k - 1];
		}

		__global__ void downSweep(int n, int k, int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;
			// need to check boundary
			if ((index % (2 * k) == 0) && (index + (2 * k) <= n)) {
				int temp = idata[index + k - 1];
				idata[index + k - 1] = idata[index + (2 * k) - 1];
				idata[index + (2 * k) - 1] += temp;
			}
		}

        void scan(int n, int *odata, const int *idata) {
			int *exclusive;
			int length = pow(2, ilog2ceil(n));
			cudaMalloc((void**)&exclusive, length * sizeof(int));
			cudaMemset(exclusive, 0, length * sizeof(int));
			cudaMemcpy(exclusive, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			dim3 fullBlocksPerGrid((length + blockSize - 1) / blockSize);
			//timer().startGpuTimer();
			// TODO
			// up-sweep
			for (int d = 1; d < length; d *= 2) {
				upSweep<<< fullBlocksPerGrid, blockSize >>>(length, d, exclusive);
			}
			cudaMemset(exclusive + length - 1, 0, sizeof(int));
			// down-sweep
			for (int d = length / 2; d >= 1; d /= 2) {
				downSweep<<< fullBlocksPerGrid, blockSize >>>(length, d, exclusive);
			}
            //timer().endGpuTimer();
			cudaMemcpy(odata, exclusive, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(exclusive);
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
        int compact(int n, PathSegment *odata,  PathSegment *idata) {
			int *bools;
			int *indices;
			int *i_aug;
			int *o_aug;
			int length = pow(2, ilog2ceil(n));
			cudaMalloc((void**)&bools, length * sizeof(int));
			cudaMalloc((void**)&indices, length * sizeof(int));
			cudaMalloc((void**)&i_aug, n * sizeof(int));
			cudaMalloc((void**)&o_aug, n * sizeof(int));

			cudaMemset(bools, 0, length * sizeof(int));
			cudaMemset(indices, 0, length * sizeof(int));

			cudaMemcpy(i_aug, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			dim3 fullBlocksPerGrid((length + blockSize - 1) / blockSize);

			//timer().startGpuTimer();
            // TODO
			StreamCompaction::Common::kernMapToBoolean<<< fullBlocksPerGrid, blockSize >>>(n, bools, i_aug);
			scan(n, indices, bools);
			StreamCompaction::Common::kernScatter <<< fullBlocksPerGrid, blockSize >>>(n, o_aug, i_aug, bools, indices);
            //timer().endGpuTimer();
			cudaMemcpy(odata, o_aug, n * sizeof(int), cudaMemcpyDeviceToHost);

			int num1 = 0;
			cudaMemcpy(&num1, &bools[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
			int num2 = 0;
			cudaMemcpy(&num2, &indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
			
			cudaFree(bools);
			cudaFree(indices);
			cudaFree(i_aug);
			cudaFree(o_aug);

            return num1 + num2;
        }
    }
}
