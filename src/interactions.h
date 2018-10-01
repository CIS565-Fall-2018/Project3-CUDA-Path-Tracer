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

float shift = 0.001f;
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


	if (m.emittance > 0.0f) {
		// emittance
		pathSegment.remainingBounces = 0;
		pathSegment.color *= (m.color * m.emittance);
		return;
	}
	//// diffuse reflection
	//pathSegment.color *= m.color;
	//pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
	//pathSegment.ray.origin = intersect + normal * 0.001f;
	//pathSegment.remainingBounces--;

	//if (m.hasReflective == 0.f && m.hasRefractive == 0.f) {
	//	pathSegment.color *= m.color;
	//	pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
	//	pathSegment.ray.origin = intersect + normal * 0.001f;
	//	pathSegment.remainingBounces--;
	//	return;
	//}
	//int types = 1;
	//if (m.hasReflective > 0.f) {
	//	types++;
	//}
	//if (m.hasRefractive > 0.f) {
	//	types++;
	//}
	//thrust::uniform_int_distribution<int> uTypes(0, types - 1);
	//int randomType = uTypes(rng);
	//// diffusion
	//if (randomType == 0) {
	//	pathSegment.color *= m.color;
	//	pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
	//	pathSegment.ray.origin = intersect + normal * 0.001f;
	//	pathSegment.remainingBounces--;
	//	return;
	//}
	//// reflection
	//if ((randomType == 1 && types == 2 && m.hasReflective) || (randomType == 1 && types == 3)) {
	//	pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
	//	pathSegment.color *= m.specular.color;
	//	pathSegment.color *= glm::abs(glm::dot(pathSegment.ray.direction, normal)) * m.color;
	//	pathSegment.ray.origin = intersect + normal * 0.001f;
	//	pathSegment.remainingBounces--;
	//}
	//// refraction
	//else {
	//	glm::vec3 ray = pathSegment.ray.direction;
	//	float cosTheta = glm::dot(ray, normal);
	//	float etaIn, etaOut;
	//	if (cosTheta < 0) {
	//		etaIn = 1.f;
	//		etaOut = m.indexOfRefraction;
	//	}
	//	else {
	//		etaIn = m.indexOfRefraction;
	//		etaOut = 1.f;
	//	}
	//	//cosTheta = std::abs(cosTheta);
	//	float R0 = (etaIn - etaOut) / (etaIn + etaOut);
	//	R0 *= R0;
	//	float R = R0 + (1 - R0) * pow(1 - cosTheta, 5);
	//	thrust::uniform_real_distribution<float> uRefr(0, 1);
	//	float refr = uRefr(rng);
	//	if (refr < 1) {
	//		pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, normal, etaOut / etaIn);
	//		pathSegment.color *= m.color;
	//		pathSegment.ray.origin = intersect + normal * 0.001f;
	//		pathSegment.remainingBounces--;
	//	}
	//	else {
	//		pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
	//		pathSegment.color *= m.specular.color;
	//		pathSegment.color *= m.color;
	//		pathSegment.ray.origin = intersect + normal * 0.001f;
	//		pathSegment.remainingBounces--;
	//	}
	//}
	
	if (m.hasReflective > 0.f) {
		thrust::uniform_real_distribution<float> u01(0, 1);
		float probability = u01(rng);
		if (probability > 0.5) {
			pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
			pathSegment.color *= m.specular.color;
			pathSegment.color *=  m.color;
			pathSegment.ray.origin = intersect + normal * 0.001f;
			pathSegment.remainingBounces--;
		}
		else {
			pathSegment.color *= m.color;			
			pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
			pathSegment.ray.origin = intersect + normal * 0.001f;
			pathSegment.remainingBounces--;
		}
	}
	else if (m.hasRefractive > 0.f) {
		glm::vec3 ray = pathSegment.ray.direction;
		float cosTheta = glm::dot(normal, ray);
		float etaIn, etaOut;
		if (cosTheta > 0) {
			etaIn = m.indexOfRefraction;
			etaOut = 1.f;
		}
		else {
			etaIn = 1.f;
			etaOut = m.indexOfRefraction;
		}
		cosTheta = abs(cosTheta);
		float R0 = (etaIn - etaOut) / (etaIn + etaOut);
		R0 *= R0;
		float R = R0 + (1 - R0) * pow(1 - cosTheta, 5);
		thrust::uniform_real_distribution<float> uRefr(0, 1);
		float refr = uRefr(rng);
		if (refr > R) {
			pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, normal, etaIn / etaOut);
			pathSegment.color *= m.color;
			pathSegment.ray.origin = intersect + normal * 0.001f;
			pathSegment.remainingBounces--;
		}
		else {
			pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
			pathSegment.color *= m.specular.color;
			pathSegment.color *= m.color;
			pathSegment.ray.origin = intersect + normal * 0.001f;
			pathSegment.remainingBounces--;
		}
	}
	else {
			pathSegment.color *= m.color;
			pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
			pathSegment.ray.origin = intersect + normal * 0.001f;
			pathSegment.remainingBounces--;
			return;
	}
}


