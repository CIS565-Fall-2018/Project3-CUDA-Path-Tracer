#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	
	
	pathSegment.ray.origin = intersect;
	if (m.hasReflective)
	{
		glm::vec3 reflectedRay = glm::reflect(pathSegment.ray.direction, normal);		
		pathSegment.color = pathSegment.color * m.specular.color * float(0.5) +  pathSegment.color * m.color * float(0.5);		
		//pathSegment.color *= (m.specular.color * glm::abs(glm::dot(reflectedRay, normal)));
		// Because with more refleciton, the color needs to multiply more colors and will be darker,
		// so we do not need mulatiply glm::dot(ray, normal) to it?
		pathSegment.ray.direction = reflectedRay;
		pathSegment.remainingBounces--;
	}
	else
	{
		glm::vec3 randomRay = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.color *= m.color;		
		pathSegment.ray.direction = randomRay;
		pathSegment.remainingBounces--;
	}
	
}




//__global__ void shadeFakeMaterial(
//	int iter
//	, int num_paths
//	, ShadeableIntersection * shadeableIntersections
//	, PathSegment * pathSegments
//	, Material * materials
//)
//{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (idx < num_paths)
//	{
//		ShadeableIntersection intersection = shadeableIntersections[idx];
//		if (intersection.t > 0.0f) { // if the intersection exists...
//									 // Set up the RNG
//									 // LOOK: this is how you use thrust's RNG! Please look at
//									 // makeSeededRandomEngine as well.
//			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
//			thrust::uniform_real_distribution<float> u01(0, 1);
//
//			Material material = materials[intersection.materialId];
//			glm::vec3 materialColor = material.color;
//
//			// If the material indicates that the object was a light, "light" the ray
//			if (material.emittance > 0.0f) {
//				pathSegments[idx].color *= (materialColor * material.emittance);
//			}
//			// Otherwise, do some pseudo-lighting computation. This is actually more
//			// like what you would expect from shading in a rasterizer like OpenGL.
//			// TODO: replace this! you should be able to start with basically a one-liner
//			else {
//				float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
//				pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
//				pathSegments[idx].color *= u01(rng); // apply some noise because why not
//			}
//			// If there was no intersection, color the ray black.
//			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
//			// used for opacity, in which case they can indicate "no opacity".
//			// This can be useful for post-processing and image compositing.
//		}
//		else {
//			pathSegments[idx].color = glm::vec3(0.0f);
//		}
//	}
//}
