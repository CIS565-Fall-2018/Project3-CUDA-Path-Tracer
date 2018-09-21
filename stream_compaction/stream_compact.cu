#include "stream_compact.h"


__global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (index >= n) return;
	if (idata[index] != 0) bools[index] = 1;
	else bools[index] = 0;
}

__global__ void kernScatter(int n, int *odata,
	const int *idata, const int *bools, const int *indices) {
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (index >= n) return;
	if (bools[index] == 1) odata[indices[index]] = idata[index];
}

__global__ void kernScanDataUpSweep(int n, int offset1, int offset2, int* buff) {
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	int access = index * offset2 - 1;
	if (access >= n || n < 1 || access < 0) return;

	buff[access] += buff[access - offset1];
}


__global__ void kernScanDataDownSweep(int n, int offset1, int offset2, int* buff) {
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	int access = index * offset2 - 1;
	if (access >= n || n < 1 || access < 0) return;

	int temp = buff[access - offset1];
	buff[access - offset1] = buff[access];
	buff[access] += temp;
}

__global__ void kernScanDataShared(int n, int* in, int* out, int* sums) {
	// init shared mem for block, could improve latency
	__shared__ int sBuf[blockSize];

	int tx = threadIdx.x;
	int index = (blockDim.x * blockIdx.x) + tx;

	int off_tx = tx + CONFLICT_FREE_OFFSET(tx);

	// copy used vals to shared mem
	sBuf[off_tx] = (index < n) ? in[index] : 0;

	__syncthreads(); // avoid mem issues

	int offset; // step size
	int access; // shared buffer access index
	int a2;

	// Upsweep
	for (offset = 1; offset < blockSize; offset *= 2) {
		access = (2 * offset * (tx + 1)) - 1;
		a2 = access - offset;
		a2 += CONFLICT_FREE_OFFSET(a2);
		access += CONFLICT_FREE_OFFSET(access);
		if (access < blockSize) sBuf[access] += sBuf[a2];
		__syncthreads(); // avoid mem issues
	}

	// prepare array for downsweep
	if (tx == blockSize - 1 + CONFLICT_FREE_OFFSET(blockSize - 1)) {
		if (sums != NULL) sums[blockIdx.x] = sBuf[off_tx];
		sBuf[off_tx] = 0;
	}
	__syncthreads();
	if (index >= n - 1) sBuf[off_tx] = 0;
	__syncthreads(); // avoid mem issues

	// Downsweep (inclusive)
	// do exclusive downsweep
	int temp;

	for (offset = blockSize; offset >= 1; offset /= 2) {
		access = (2 * offset * (tx + 1)) - 1;
		a2 = access - offset;
		a2 += CONFLICT_FREE_OFFSET(a2);
		access += CONFLICT_FREE_OFFSET(access);
		if (access < blockSize) {
			temp = sBuf[a2]; // store left child
			sBuf[a2] = sBuf[access]; // swap
			sBuf[access] += temp; // add
		}
		__syncthreads(); // avoid mem issues
	}

	// write to dev memory
	if (index < n) {
		out[index] = sBuf[off_tx];
	}
}

__global__ void kernStitch(int n, int* in, int* sums) {
	int bx = blockIdx.x;
	int index = (blockDim.x * bx) + threadIdx.x;;

	if (bx == 0) return;
	if (index >= n) return;
	in[index] += sums[bx];
}

void gpuScan(int n, int* dev_out, int* dev_in) {

	// set up shared mem scan
	int num_blocks = 1 + (n - 1) / blockSize; // number of blocks n elements fit into
	int limit = ilog2ceil(num_blocks);
	int sum_size = pow(2, limit); // size of block sum array for scanning

	int num_threads;

	dim3 fullBlocksPerGrid(num_blocks); // blocks in grid to start with

	int* dev_sums; // sums, from first blockwise scan
	cudaMalloc((void**)&dev_sums, sum_size * sizeof(int));

	cudaMemset(dev_out, 0, n * sizeof(int)); //initialize output buffer
	checkCUDAError("initializing shared mem scan data buff fail!");

	// shared mem scan blocks of data
	kernScanDataShared << <fullBlocksPerGrid, blockSize >> >(n, dev_in, dev_out, dev_sums);
	checkCUDAError("shared mem scan fail!");

	
	if (num_blocks > 1) {
		// scan sums
		if (num_blocks <= blockSize) {
			// can use faster shared mem scan
			fullBlocksPerGrid.x = 1 + (num_blocks - 1) / blockSize;
			kernScanDataShared << <fullBlocksPerGrid, blockSize >> >(num_blocks, dev_sums, dev_sums, NULL);
			checkCUDAError("sum shared scan fail!");
		}

		else {
			// use global memory scan (easier)

			int d;
			int offset1;
			int offset2;

			// UpSweep
			for (d = 1; d <= limit; d++) {
				offset1 = pow(2, d - 1);
				offset2 = pow(2, d);
				num_threads = sum_size / offset2;
				fullBlocksPerGrid.x = 1 + num_threads / blockSize;
				kernScanDataUpSweep << <fullBlocksPerGrid, blockSize >> > (sum_size, offset1, offset2, dev_sums);
				checkCUDAError("upsweep fail!");
			}

			// DownSweep
			cudaMemset(dev_sums + num_blocks - 1, 0, (sum_size - num_blocks + 1) * sizeof(int));
			for (d = limit; d >= 1; d--) {
				offset1 = pow(2, d - 1);
				offset2 = pow(2, d);
				num_threads = sum_size / offset2;
				fullBlocksPerGrid.x = 1 + num_threads / blockSize;
				kernScanDataDownSweep << <fullBlocksPerGrid, blockSize >> > (sum_size, offset1, offset2, dev_sums);
				checkCUDAError("downsweep fail!");
			}
		}

		// stitch together blocks
		fullBlocksPerGrid.x = num_blocks;
		kernStitch << <fullBlocksPerGrid, blockSize >> >(n, dev_out, dev_sums);
		checkCUDAError("shared mem scan stitch fail!");
	}

	cudaFree(dev_sums);
}

// for pathtrace:
// dev_in1 is the # of bounces remaining array
// dev_in2 is the pixel index
// dev_out is then the sorted pixel indices, 
// return is # members w/ bounces remaining
int intCompact(int n, int* dev_out, int* dev_in1, int *dev_in2) { 
	int* dev_map;
	int* dev_scan;

	int num_blocks = 1 + (n - 1) / blockSize;
	dim3 fullBlocksPerGrid(num_blocks);

	cudaMalloc((void**)&dev_map, n * sizeof(int));
	cudaMalloc((void**)&dev_scan, n * sizeof(int));
	checkCUDAError("compact malloc fail!");

	// map
	kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(n, dev_map, dev_in1);
	checkCUDAError("shared mem compact bool mapping fail!");

	// scan
	gpuScan(n, dev_scan, dev_map);

	// scatter
	kernScatter << <fullBlocksPerGrid, blockSize >> >(n, dev_out, dev_in2, dev_map, dev_scan);
	checkCUDAError("shared mem compact scatter fail!");

	// calc. num of elements
	int r1;
	int r2;

	cudaMemcpy(&r1, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&r2, dev_map + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(dev_map);
	cudaFree(dev_scan);

	return (r1 + r2);
}

#define TEST_SIZE 15
void scanTest() {
	int test_in[TEST_SIZE] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14};
	int test_out[TEST_SIZE] = { 0 };

	int* dev_in;
	int* dev_out;

	cudaMalloc((void**)&dev_in, TEST_SIZE * sizeof(int));
	cudaMalloc((void**)&dev_out, TEST_SIZE * sizeof(int));
	checkCUDAError("test malloc fail!");

	cudaMemcpy(dev_in, test_in, TEST_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("test in copy fail!");

	gpuScan(TEST_SIZE, dev_out, dev_in);

	cudaMemcpy(test_out, dev_out, TEST_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	checkCUDAError("test out copy fail!");

	printf("\nTest scan: { ");
	for (int i = 0; i < TEST_SIZE; i++) {
		printf("%i ", test_out[i]);
	}
	printf(" }\n");

	cudaFree(dev_in);
	cudaFree(dev_out);
	checkCUDAError("test free fail!");
}
