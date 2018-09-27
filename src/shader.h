#pragma once
__device__ void shadeDiffuse(PathSegment & path, thrust::default_random_engine rng, glm::vec3 normal) {
	// select if diffuse material
	path.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
}

__device__ void shadeReflective(PathSegment & path, glm::vec3 normal) {
	// select if reflective material
	path.ray.direction = calculateIdealReflect(normal, path.ray.direction);

}

__device__ float schlickApprox(float n, glm::vec3 normal, glm::vec3 incident) {
	float dot = fabs(glm::dot(normal, incident));
	float r0 = pow((1 - n) / (1 + n), 2);
	float R = r0 + (1 - r0) * pow((1 - dot), 5);
	return R;
}

__device__ void shadeRefractive(PathSegment & path, Material material, glm::vec3 intersect, glm::vec3 normal, thrust::default_random_engine rng) {
	float n = material.indexOfRefraction;
	if (path.inside == true) n = 1 / n;
	float R = schlickApprox(n, normal, path.ray.direction);
	float T = 1 - R;
	glm::vec3 incident = path.ray.direction;

	thrust::uniform_real_distribution<float> u01(0, 1);
	if (u01(rng) > T) {
		path.ray.direction = calculateIdealReflect(normal, incident);
	}
	else {
		//if (path.inside) normal = -1.0f * (normal);
		path.ray.direction = calculateIdealRefract(normal, incident, n);
		path.ray.origin = getPointOnRay(path.ray, 0.1f);
		path.inside = !path.inside;
	}
}

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

		if (material.hasReflective) {
			//reflective shader
			scatterRay(path, intersectPoint, material);
			shadeReflective(path, normal);

		}
		else if (material.hasRefractive) {
			scatterRay(path, intersectPoint, material);
			shadeRefractive(path, material, intersectPoint, normal, rng);
		}
		else {
			// generic diffuse shading
			scatterRay(path, intersectPoint, material);
			shadeDiffuse(path, rng, normal);
		}


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
		scatterRay(path, intersectPoint, material);
		path.ray.direction = calculateRandomDirectionInHemisphere(intersect.surfaceNormal, rng);
		
	}

	//pathSegments[idx] = path; // update path ray

}