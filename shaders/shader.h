#pragma once

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

// Basic naive shading/ray gen
__global__ void kernShadeGeneric(int iter, int num_paths, ShadeableIntersection *shadeableIntersections, PathSegment *pathSegments, Material *materials, int* dev_active) {
	// Calc. Thread Index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > num_paths) return;

	int active = dev_active[idx];

	PathSegment ray = pathSegments[active]; // the ray we are checking

	if (ray.remainingBounces < 1) return;

	Material material; // material of object intersecting ray (if exists)
	glm::vec3 intersectPoint; // calculated intersection point

	// Set up RNG
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
	//thrust::uniform_real_distribution<float> u01(0, 1);

	// check for intersection
	ShadeableIntersection intersect = shadeableIntersections[active]; // the calculated intersection of this ray with the scene objects
	if (intersect.t <= 0.0f) {
		// if no intersection, black
		ray.color = glm::vec3(0.0f);
		
		// terminate path
		ray.remainingBounces = 0;
	}
	else {
		material = materials[intersect.materialId]; // load material
		ray.color *= material.color; // the color attributed to the ray is adjusted by the material it interacts with

		//bsdf stuff
		intersectPoint = ray.ray.origin + intersect.t * ray.ray.direction;
		scatterRay(ray, intersectPoint, intersect.surfaceNormal, material, rng);

		ray.remainingBounces--;

		// if has emmitance (light) add in light factor
		if (material.emittance > 0.0f) {
			ray.color *= material.emittance;

			// terminate path
			ray.remainingBounces = 0;
		}
	}
	pathSegments[active] = ray; // update path ray

}