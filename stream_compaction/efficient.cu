#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "thrust.h"

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#define BLOCKSIZE_SHARED 64

namespace StreamCompaction {
	namespace Efficient {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		__global__ void kernUpSweep(int n, int d, int *odata) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			int step = 1 << (d + 1);
			if (index % step == 0) {
				odata[index + step - 1] += odata[index + (1 << d) - 1];
			}
		}

		__global__ void kernDownSweep(int n, int d, int *odata) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			int step = 1 << (d + 1);
			if (index % step == 0) {
				int t = odata[index + (1 << d) - 1];
				odata[index + (1 << d) - 1] = odata[index + step - 1];
				odata[index + step - 1] += t;
			}
		}



		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void nonoptscan(int n, int *odata, const int *idata) {
			// TODO
			int upLimit = ilog2ceil(n);
			int len = 1 << upLimit;

			int blockSize = 1024;
			dim3 threadsPerBlock(blockSize);
			dim3 blocksPerGrid((len + blockSize - 1) / blockSize);

			int *dev_data;
			cudaMalloc((void**)&dev_data, len * sizeof(int));
			checkCUDAError("cudaMalloc dev_data failed!");
			cudaMemcpy(dev_data, idata, len * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_data failed!");



			timer().startGpuTimer();
			for (int d = 0; d <= upLimit - 1; d++) {
				kernUpSweep << <blocksPerGrid, threadsPerBlock >> > (len, d, dev_data);
				checkCUDAError("kernUpSweep failed!");
			}

			cudaMemset(&dev_data[len - 1], 0, sizeof(int));
			checkCUDAError("cudaMemcpy set last one to be zero failed!");

			for (int d = upLimit - 1; d >= 0; d--) {
				kernDownSweep << <blocksPerGrid, threadsPerBlock >> > (len, d, dev_data);
				checkCUDAError("kernDownSweep failed!");
			}
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_data, len * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_data failed!");
			cudaFree(dev_data);
			cudaDeviceSynchronize();
		}

		__global__ void kernOptUpSweep(int n, int d, int *odata) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			int step = 1 << (d + 1);
			odata[index * step + step - 1] += odata[index * step + (1 << d) - 1];
		}

		__global__ void kernOptDownSweep(int n, int d, int *odata) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			int step = 1 << (d + 1);
			int t = odata[index*step + (1 << d) - 1];
			odata[index * step + (1 << d) - 1] = odata[index * step + step - 1];
			odata[index * step + step - 1] += t;
		}

		void scan(int n, int *odata, const int *idata) {
			int upLimit = ilog2ceil(n);
			int len = 1 << upLimit;

			int blockSize = 1024;
			dim3 threadsPerBlock(blockSize);
			dim3 blocksPerGrid((len + blockSize - 1) / blockSize);

			int *dev_data;
			cudaMalloc((void**)&dev_data, len * sizeof(int));
			checkCUDAError("cudaMalloc dev_data failed!");
			cudaMemcpy(dev_data, idata, len * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_data failed!");



			timer().startGpuTimer();
			for (int d = 0; d <= upLimit - 1; d++) {
				int step = 1 << (d + 1);
				int tempLen = len / step;
				blocksPerGrid = dim3((tempLen + blockSize) / blockSize);
				kernOptUpSweep << <blocksPerGrid, threadsPerBlock >> > (tempLen, d, dev_data);
				checkCUDAError("kernUpSweep failed!");
			}

			cudaMemset(&dev_data[len - 1], 0, sizeof(int));
			checkCUDAError("cudaMemcpy set last one to be zero failed!");

			for (int d = upLimit - 1; d >= 0; d--) {
				int step = 1 << (d + 1);
				int tempLen = len / step;
				blocksPerGrid = dim3((tempLen + blockSize) / blockSize);
				kernOptDownSweep << <blocksPerGrid, threadsPerBlock >> > (tempLen, d, dev_data);
				checkCUDAError("kernDownSweep failed!");
			}
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_data, len * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_data failed!");
			cudaFree(dev_data);
			cudaDeviceSynchronize();
		}

		void gpuScan(int n, int *data) {
			int upLimit = ilog2ceil(n);
			int blockSize = 1024;
			dim3 threadsPerBlock(blockSize);
			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

			for (int d = 0; d <= upLimit - 1; d++) {
				kernUpSweep << <blocksPerGrid, threadsPerBlock >> > (n, d, data);
				checkCUDAError("kernUpSweep failed!");
			}

			cudaMemset(&data[n - 1], 0, sizeof(int));
			checkCUDAError("cudaMemcpy set last one to be zero failed!");

			for (int d = upLimit - 1; d >= 0; d--) {
				kernDownSweep << <blocksPerGrid, threadsPerBlock >> > (n, d, data);
				checkCUDAError("kernDownSweep failed!");
			}
		}

		__global__ void kernSharedMemoryScan(int n, int *odata, const int *idata, int *blockData) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			extern __shared__ int temp[BLOCKSIZE_SHARED + NUM_BANKS];
			int thid = threadIdx.x;
			int offset = 1;
			int stride = blockIdx.x * blockDim.x;
			int ai = thid << 1;
			int bi = ai + 1;
			int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
			int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
			temp[ai + bankOffsetA] = idata[ai + stride];
			temp[bi + bankOffsetB] = idata[bi + stride];
			for (int d = BLOCKSIZE_SHARED >> 1; d > 0; d >>= 1)
			{
				__syncthreads();
				if (thid < d)
				{
					int ai = offset*(2 * thid + 1) - 1;
					int bi = offset*(2 * thid + 2) - 1;
					ai += CONFLICT_FREE_OFFSET(ai);
					bi += CONFLICT_FREE_OFFSET(bi);
					temp[bi] += temp[ai];
				}
				offset *= 2;
			}
			__syncthreads();
			if (thid != 0) {
				blockData[blockIdx.x] = temp[BLOCKSIZE_SHARED - 1 + CONFLICT_FREE_OFFSET(BLOCKSIZE_SHARED - 1)];
				temp[BLOCKSIZE_SHARED - 1 + CONFLICT_FREE_OFFSET(BLOCKSIZE_SHARED - 1)] = 0;
			}
			for (int d = 1; d < BLOCKSIZE_SHARED; d *= 2) // traverse down tree & build scan
			{
				offset >>= 1;
				__syncthreads();
				if (thid < d)
				{
					int ai = offset*(2 * thid + 1) - 1;
					int bi = offset*(2 * thid + 2) - 1;
					ai += CONFLICT_FREE_OFFSET(ai);
					bi += CONFLICT_FREE_OFFSET(bi);
					float t = temp[ai];
					temp[ai] = temp[bi];
					temp[bi] += t;
				}
			}
			ai = thid << 1;
			bi = ai + 1;
			odata[ai + stride] = temp[ai + bankOffsetA];
			odata[bi + stride] = temp[bi + bankOffsetB];
		}

		__global__ void kernAddBlockData(int n, const int *idata, int *odata, const int *blockData) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			int bid = blockIdx.x;
			if (bid > 0) {
				odata[index] = idata[index] + blockData[index];
			}
			else {
				odata[index] = idata[index];
			}
		}

		void sharedMemoryScan(int n, int *odata, const int *idata) {
			int upLimit = ilog2ceil(n);
			int len = 1 << upLimit;

			int blockSize = 64;
			dim3 threadsPerBlock(BLOCKSIZE_SHARED / 2);
			dim3 blocksPerGrid((n + BLOCKSIZE_SHARED - 1) / blockSize);


			int alen = ((n + BLOCKSIZE_SHARED - 1) / BLOCKSIZE_SHARED);
			dim3 threadsPerBlock_Block(BLOCKSIZE_SHARED / 2);
			dim3 blocksPerGrid_Block((alen + BLOCKSIZE_SHARED - 1) / BLOCKSIZE_SHARED);
			int *dev_idata;
			int *dev_odata;
			int *dev_blockData;
			int *dev_blockSum;
			cudaMalloc((void**)&dev_idata, len * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, len * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&dev_blockData, alen * sizeof(int));
			checkCUDAError("cudaMalloc dev_blockData failed!");
			cudaMalloc((void**)&dev_blockSum, alen * sizeof(int));
			checkCUDAError("cudaMalloc dev_blockSum failed!");
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_idata failed!");

			int *host_aux, *host_aux_sum;
			cudaMallocHost((void**)&host_aux, alen * sizeof(int)); // Use Pin-Memory to ensure the maximum memory speed
			cudaMallocHost((void**)&host_aux_sum, alen * sizeof(int));

			timer().startGpuTimer();
			kernSharedMemoryScan << <blocksPerGrid, threadsPerBlock >> > (len, dev_odata, dev_idata, dev_blockData);
			checkCUDAError("kernSharedMemoryScan failed!");
			cudaMemcpy(host_aux, dev_blockData, alen * sizeof(int), cudaMemcpyDeviceToHost);
			thrust::inclusive_scan(host_aux, host_aux + alen, host_aux_sum);
			cudaMemcpy(dev_blockSum, host_aux_sum, alen * sizeof(int), cudaMemcpyHostToDevice);
			kernAddBlockData << <blocksPerGrid, BLOCKSIZE_SHARED >> > (n, dev_odata, dev_idata, dev_blockSum);
			checkCUDAError("kernSharedMemoryScan failed!");
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_odata failed!");

			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_blockData);
			cudaFree(dev_blockSum);
			cudaDeviceSynchronize();
			cudaFreeHost(host_aux);
			cudaFreeHost(host_aux_sum);
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
		int compact(int n, int *odata, const int *idata) {
			int upLimit = ilog2ceil(n);
			int len = 1 << upLimit;
			int blockSize = 1024;
			dim3 threadsPerBlock(blockSize);
			dim3 blocksPerGrid((len + blockSize - 1) / blockSize);
			int *dev_odata;
			int *dev_idata;
			int *dev_bools;
			int *dev_indices;

			cudaMalloc((void**)&dev_odata, sizeof(int) * len);
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&dev_idata, sizeof(int) * len);
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_bools, sizeof(int) * len);
			checkCUDAError("cudaMalloc dev_bools failed!");
			cudaMalloc((void**)&dev_indices, sizeof(int) * len);
			checkCUDAError("cudaMalloc dev_indices failed!");
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_idata failed!");

			timer().startGpuTimer();

			StreamCompaction::Common::kernMapToBoolean << <blocksPerGrid, threadsPerBlock >> > (len, dev_bools, dev_idata);
			checkCUDAError("kernMaptoBoolean failed!");

			cudaMemcpy(dev_indices, dev_bools, sizeof(int) * len, cudaMemcpyDeviceToDevice);
			cudaMemset(&dev_odata[n], 0, sizeof(int) * (len - n));
			gpuScan(len, dev_indices);

			StreamCompaction::Common::kernScatter << <blocksPerGrid, threadsPerBlock >> > (len, dev_odata, dev_idata, dev_bools, dev_indices);
			checkCUDAError("kernScatter failed!");

			timer().endGpuTimer();
			cudaMemcpy(odata, dev_odata, sizeof(int) * len, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy odata failed!");

			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaFree(dev_indices);
			cudaFree(dev_bools);
			int count = 0;
			while (odata[count] != 0) {
				count++;
			}
			return count;
		}
	}
}
