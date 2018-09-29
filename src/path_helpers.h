#pragma once

#define COMPACT_BLOCK 512

#define CACHE_FIRST_BOUNCE 1
#define MATERIAL_SORT 1
#define DEPTH_OF_FIELD 0
#define SUBSURFACE 1

struct isBouncy
{
	isBouncy() {};
	__host__ __device__
		bool operator()(const PathSegment& path)
	{
		return (path.remainingBounces > 0);
	}
};

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

__global__ void kernSCsetup(int n, int* dev_val, PathSegment* paths) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= n) return;
	if ((paths[index]).remainingBounces > 0) dev_val[index] = 1;
}

__global__ void kernComputeFalseIdx(int n, int totTrue, int* f_arr, int* t_arr) {
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (index >= n) return;

	f_arr[index] = index - t_arr[index] + totTrue;
}

__global__ void kernSortRays(int n, int* bools, int* t_idx, int* f_idx, PathSegment* input, PathSegment* output) {
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (index >= n) return;

	if (bools[index] == 1) output[t_idx[index]] = input[index];
	else output[f_idx[index]] = input[index];
}

int compactRays(int n, PathSegment *dev_paths) {
	int num_blocks = (n + COMPACT_BLOCK - 1) / COMPACT_BLOCK;
	dim3 fullBlocksPerGrid(num_blocks);

	int *dev_map;
	int *dev_scan;
	int *dev_false;

	PathSegment* dev_out;

	cudaMalloc((void**)&dev_map, n * sizeof(int));
	cudaMalloc((void**)&dev_scan, n * sizeof(int));
	cudaMalloc((void**)&dev_false, n * sizeof(int));
	cudaMalloc((void**)&dev_out, n * sizeof(PathSegment));
	checkCUDAError("compact malloc fail!");

	cudaMemset(dev_out, -1, n * sizeof(PathSegment));

	// map
	kernSCsetup << <fullBlocksPerGrid, COMPACT_BLOCK >> >(n, dev_map, dev_paths);
	checkCUDAError("compact setup fail!");

	// scan
	gpuScan(n, dev_scan, dev_map);
	checkCUDAError("scanning fail!");

	// calc. num of elements
	int r1;
	int r2;

	cudaMemcpy(&r1, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&r2, dev_map + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy fail!");

	// get false indices (kind of like radix sort)
	kernComputeFalseIdx << < fullBlocksPerGrid, COMPACT_BLOCK >> >(n, r1 + r2, dev_false, dev_scan);

	// scatter
	kernSortRays << < fullBlocksPerGrid, COMPACT_BLOCK >> > (n, dev_map, dev_scan, dev_false, dev_paths, dev_out);
	checkCUDAError("compact scatter fail!");

	// copy output
	cudaMemcpy(dev_paths, dev_out, n * sizeof(PathSegment), cudaMemcpyDeviceToDevice);

	// free memory
	cudaFree(dev_map);
	cudaFree(dev_scan);
	cudaFree(dev_false);
	cudaFree(dev_out);
	checkCUDAError("free fail!");

	return (r1 + r2);

}

__global__ void kernMaterialMap(int n, ShadeableIntersection* intersects, int* materials, int* idx) {
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (index >= n) return;

	idx[index] = index;
	materials[index] = (intersects[index]).materialId;
}

__global__ void kernSortMaterial(int n, int* indices, PathSegment* paths, ShadeableIntersection* intersects, PathSegment* path_sort, ShadeableIntersection* inter_sort) {
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (index >= n) return;

	int idx = indices[index];

	inter_sort[index] = intersects[idx];
	path_sort[index] = paths[idx];
}

void sortRaysMaterial(int n, PathSegment* paths, ShadeableIntersection* intersects) {
	int num_blocks = (n + COMPACT_BLOCK - 1) / COMPACT_BLOCK;
	dim3 fullBlocksPerGrid(num_blocks);

	int* dev_mat;
	int* dev_idx;

	ShadeableIntersection* inter_sort;
	PathSegment* path_sort;

	cudaMalloc((void**)&dev_mat, n * sizeof(int));
	cudaMalloc((void**)&dev_idx, n * sizeof(int));
	cudaMalloc((void**)&path_sort, n * sizeof(PathSegment));
	cudaMalloc((void**)&inter_sort, n * sizeof(ShadeableIntersection));

	// create material map
	kernMaterialMap<<<fullBlocksPerGrid, COMPACT_BLOCK >>>(n, intersects, dev_mat, dev_idx);

	// use as keys for sort
	thrust::device_ptr<int> dev_thrust_keys = thrust::device_ptr<int>(dev_mat);

	thrust::device_ptr<int> dev_thrust_vals = thrust::device_ptr<int>(dev_idx);

	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + n, dev_thrust_vals);

	// sort
	kernSortMaterial << <fullBlocksPerGrid, COMPACT_BLOCK >> > (n, dev_idx, paths, intersects, path_sort, inter_sort);

	// copy sorted vals
	cudaMemcpy(paths, path_sort, n * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	cudaMemcpy(intersects, inter_sort, n * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);


	cudaFree(dev_mat);
	cudaFree(dev_idx);
	cudaFree(path_sort);
	cudaFree(inter_sort);

}

__global__ void kernDOF(int num_paths, int iter, Camera cam, float focalLength, PathSegment* paths) {
	// Calc. Thread Index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > num_paths) return;

	Ray & ray = paths[idx].ray;

	glm::vec3 aim = ray.direction * focalLength + cam.position;

	// Set up RNG
	thrust::default_random_engine rngX = makeSeededRandomEngine(iter, num_paths - idx, 0);
	thrust::default_random_engine rngY = makeSeededRandomEngine(iter, idx, 1);
	thrust::uniform_real_distribution<float> u(-0.5, 0.5);

	glm::vec3 dif = glm::normalize(aim - ray.origin) / focalLength;
	ray.direction.x += u(rngX) * dif.x;
	ray.direction.y += u(rngY) * dif.y;

}