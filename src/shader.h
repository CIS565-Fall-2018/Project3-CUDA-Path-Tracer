#pragma once


__global__ void kernShadeMaterials(int iter, int num_paths, int depth, ShadeableIntersection *shadeableIntersections, PathSegment *pathSegments, Material *materials) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > num_paths) return;

	PathSegment & path = pathSegments[idx]; // the ray we are checking
	if (path.remainingBounces < 1) return; // abort if no need to shade

	Material material; // material of object intersecting ray (if exists)

	// Set up RNG
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);

	ShadeableIntersection intersect = shadeableIntersections[idx]; // the calculated intersection of this ray with the scene objects
	glm::vec3 intersectPoint;
	glm::vec3 normal;
	if (intersect.t == -1) {
		// if no intersection, black
		path.color = glm::vec3(0.0f);

		// terminate path
		path.remainingBounces = 0;
	}
	else {
		intersectPoint = getPointOnRay(path.ray, intersect.t);
		material = materials[intersect.materialId]; // load material
		normal = intersect.surfaceNormal;

		scatterRay(path, intersectPoint, normal, material, rng);

	}
	
}


// Basic naive shading/ray gen
__global__ void kernShadeGeneric(int iter, int num_paths, int depth, ShadeableIntersection *shadeableIntersections, PathSegment *pathSegments, Material *materials) {
	// Calc. Thread Index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > num_paths) return;

	PathSegment & path = pathSegments[idx]; // the ray we are checking

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
		//path.ray.direction = calculateRandomDirectionInHemisphere(intersect.surfaceNormal, rng);
		
	}

	//pathSegments[idx] = path; // update path ray

}