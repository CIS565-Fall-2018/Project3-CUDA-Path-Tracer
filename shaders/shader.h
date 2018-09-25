#pragma once

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

__global__ void kernSCsetup(int n, int* dev_val, PathSegment* paths) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= n) return;

	dev_val[index] = (paths[index]).remainingBounces;
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
	int num_blocks = (n + blockSize - 1) / blockSize;
	dim3 fullBlocksPerGrid(num_blocks);

	int *dev_map;
	int *dev_scan;
	int *dev_false;

	int* dev_bounce;

	PathSegment* dev_out;

	cudaMalloc((void**)&dev_map, n * sizeof(int));
	cudaMalloc((void**)&dev_scan, n * sizeof(int));
	cudaMalloc((void**)&dev_false, n * sizeof(int));
	cudaMalloc((void**)&dev_bounce, n * sizeof(int));
	cudaMalloc((void**)&dev_out, n * sizeof(PathSegment));
	checkCUDAError("compact malloc fail!");

	cudaMemset(dev_out, -1, n * sizeof(PathSegment));

	kernSCsetup<<<fullBlocksPerGrid, blockSize>>>(n, dev_bounce, dev_paths);
	checkCUDAError("compact setup fail!");
	
	// map
	kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(n, dev_map, dev_bounce);
	checkCUDAError("bool mapping fail!");

	// scan
	gpuScan(n, dev_scan, dev_map);
	checkCUDAError("scanning fail!");

	// calc. num of elements
	int r1;
	int r2;

	cudaMemcpy(&r1, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&r2, dev_map + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy fail!");

	// get false indices
	kernComputeFalseIdx<<< fullBlocksPerGrid, blockSize>>>(n, r1 + r2, dev_false, dev_scan);

	// scatter
	kernSortRays << < fullBlocksPerGrid, blockSize >> > (n, dev_map, dev_scan, dev_false, dev_paths, dev_out);
	checkCUDAError("compact scatter fail!");

	// copy output
	cudaMemcpy(dev_paths, dev_out, n * sizeof(PathSegment), cudaMemcpyDeviceToDevice);

	// free memory
	cudaFree(dev_map);
	cudaFree(dev_scan);
	cudaFree(dev_bounce);
	cudaFree(dev_false);
	cudaFree(dev_out);
	checkCUDAError("free fail!");

	return (r1 + r2);

}

// Basic naive shading/ray gen
__global__ void kernShadeGeneric(int iter, int num_paths, int depth, ShadeableIntersection *shadeableIntersections, PathSegment *pathSegments, Material *materials) {
	// Calc. Thread Index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > num_paths) return;

	PathSegment path = pathSegments[idx]; // the ray we are checking

	if (path.remainingBounces < 1) return;

	Material material; // material of object intersecting ray (if exists)
	glm::vec3 intersectPoint; // calculated intersection point

	// Set up RNG
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
	//thrust::uniform_real_distribution<float> u01(0, 1);

	// check for intersection
	ShadeableIntersection intersect = shadeableIntersections[idx]; // the calculated intersection of this ray with the scene objects
	if (intersect.t == -1) {
		// if no intersection, black
		path.color = glm::vec3(0.0f);
		
		// terminate path
		path.remainingBounces = 0;
	}
	else {
		material = materials[intersect.materialId]; // load material
		
		//bsdf stuff
		intersectPoint = getPointOnRay(path.ray, intersect.t);
		scatterRay(path, intersectPoint, intersect.surfaceNormal, material, rng);
		
	}

	pathSegments[idx] = path; // update path ray

}