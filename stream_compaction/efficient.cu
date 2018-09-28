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

		__global__ void kernUpward(int n, int d, int *data) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= n) {
				return;
			}

			int power_d = 1 << d;
			int power_d1 = 1 << (d + 1);
			
			if (k % power_d1 == 0) {
				int idx = k + power_d - 1;
				int idx_1 = k + power_d1 - 1;
				data[idx_1] += data[idx];
			}
			
		}

		__global__ void kernDownward(int n, int d, int *data) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= n) {
				return;
			}

			int power_d = 1 << d;
			int power_d1 = 1 << (d + 1);
			
			if (k % power_d1 == 0) {
				int idx = k + power_d - 1;
				int idx_1 = k + power_d1 - 1;
				int t = data[idx];
				data[idx] = data[idx_1];
				data[idx_1] += t;
			}
		}

		void scanEfficient(int n, int *dev_data) {
			int logn = ilog2ceil(n);
			int length = 1 << ilog2ceil(n);
			int blockSize = 128;
			dim3 blocksPerGrid((length + blockSize - 1) / blockSize);
			dim3 threadsPerBlock(blockSize);

			for (int i = 0; i < logn; i++) {
				kernUpward << <blocksPerGrid, threadsPerBlock >> > (length, i, dev_data);
				checkCUDAError("kernUpwardSweep failed", __LINE__);
			}
			int zero = 0;
			cudaMemcpy(dev_data + length - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);
			for (int i = logn - 1; i >= 0; i--) {
				kernDownward << <blocksPerGrid, threadsPerBlock >> > (length, i, dev_data);
				checkCUDAError("kernDownwardSweep failed", __LINE__);
			}
		}
		
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int length = 1 << ilog2ceil(n);
			int size = length * sizeof(int);
			int *idata_padded = new int[length];

			// pad the array if the length is not a power of 2
			for (int i = 0; i < n; i++) {
				idata_padded[i] = idata[i];
			}
			for (int i = n; i < length; i++) {
				idata_padded[i] = 0;
			}

			// copy the padded data to the device 
			int *dev_data;
			cudaMalloc((void**)&dev_data, size);
			checkCUDAError("ERROR: cudaMalloc dev_dat", __LINE__);
			cudaMemcpy(dev_data, idata_padded, size, cudaMemcpyHostToDevice);
			checkCUDAError("ERROR: cudaMemcpy to device", __LINE__);
            
			// perform the scan
			bool end = true;
			try {
				timer().startGpuTimer();
			}
			catch (std::exception) {
				end = false;
			}
			scanEfficient(n, dev_data);
			if (end) {
				timer().endGpuTimer();
			}

			// copy the results back to host and free data
			cudaMemcpy(idata_padded, dev_data, size, cudaMemcpyDeviceToHost);
			checkCUDAError("ERROR: cudaMemcpy to host", __LINE__);
			cudaFree(dev_data);

			// copy to output data and free padded array
			for (int i = 0; i < n; i++) {
				odata[i] = idata_padded[i];
			}

			free(idata_padded);
			
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
		__global__ void kernCreateMask(int n, int *dev_odata, int *dev_idata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= n) {
				return;
			}

			dev_odata[k] = dev_idata[k] == 0 ? 0 : 1;
		}

		__global__ void kernCompact(int n, int *tmp, int* scanRes, int *dev_odata, int *dev_idata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= n) {
				return;
			}
			if (tmp[k] == 1) {
				dev_odata[scanRes[k]] = dev_idata[k];
			}
		}

        int compact(int n, int *odata, const int *idata) {
			int blockSize = 128;
			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
			dim3 threadsPerBlock(blockSize);

			int length = 1 << ilog2ceil(n);
			int size = length * sizeof(int);
			int *idata_padded = new int[length];

			// pad the array if the length is not a power of 2
			for (int i = 0; i < n; i++) {
				idata_padded[i] = idata[i];
			}
			for (int i = n; i < length; i++) {
				idata_padded[i] = 0;
			}

			int *dev_odata, *dev_idata, *dev_tmp, *dev_scanRes;
			int *scanRes = new int[length];

			// allocate the buffers and copy the data
			cudaMalloc((void**)&dev_odata, size);
			checkCUDAError("ERROR: cudaMalloc of dev_odata");
			cudaMalloc((void**)&dev_idata, size);
			checkCUDAError("ERROR: cudaMalloc of dev_idata");
			cudaMalloc((void**)&dev_tmp, size);
			checkCUDAError("ERROR: cudaMalloc of tmp");
			cudaMalloc((void**)&dev_scanRes, size);
			checkCUDAError("ERROR: cudaMalloc of scanRes");
			cudaMemcpy(dev_idata, idata_padded, size, cudaMemcpyHostToDevice);
			checkCUDAError("ERROR: cudaMemcpy idata failed");

			timer().startGpuTimer();
			kernCreateMask << <blocksPerGrid, threadsPerBlock >> > (n, dev_tmp, dev_idata);
			cudaMemcpy(dev_scanRes, dev_tmp, size, cudaMemcpyDeviceToDevice);
			scanEfficient(n, dev_scanRes);
			kernCompact << <blocksPerGrid, threadsPerBlock >> > (n, dev_tmp, dev_scanRes, dev_odata, dev_idata);
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_odata, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(scanRes, dev_scanRes, size, cudaMemcpyDeviceToHost);
			int result = scanRes[length - 1];
			checkCUDAError("ERROR: cudaMemcpy of output data");
			
			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaFree(dev_tmp);
			cudaFree(dev_scanRes);
			free(scanRes);
			
			return result;
        }
    }
}
