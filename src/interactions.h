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

#define USE_REFRACT 1
#define USE_FRESNEL 1
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
	

#if USE_REFRACT
	thrust::uniform_real_distribution<float> u01(0, 1);
	float prob = u01(rng);
	if (prob < m.hasReflective)
	{
		// reflective
		glm::vec3 reflectedRay = glm::reflect(pathSegment.ray.direction, normal);
		pathSegment.ray.direction = reflectedRay;
		pathSegment.ray.origin = intersect;
		pathSegment.color *= m.specular.color;
		pathSegment.remainingBounces--;


	}
	else if (prob < m.hasReflective + m.hasRefractive)
	{
#if USE_FRESNEL
		// refractive
		float NI = glm::dot(pathSegment.ray.direction, normal);
		//float NI = glm::dot(normal, pathSegment.ray.direction);
		float ratio = m.indexOfRefraction;
		if (NI < 0)
		{
			ratio = 1.0f / ratio;
		}
		glm::vec3 refractedRay = glm::refract(pathSegment.ray.direction, normal, ratio);
		pathSegment.color *= m.specular.color;
		pathSegment.ray.origin = intersect + refractedRay * 1e-3f;
		pathSegment.ray.direction = refractedRay;
		pathSegment.remainingBounces--;
		//pathSegment.remainingBounces = 0;


#else
		float ratio = m.indexOfRefraction;
		float NI = glm::dot(pathSegment.ray.direction, normal);
		if (NI < 0)
		{
			ratio = 1.0f / ratio;
		}
		float r = (1 - ratio) * (1 - ratio) / ((1 + ratio) * (1 + ratio));
		float xxx = 1.0f + NI;
		float f = r + (1 - r) * xxx * xxx * xxx * xxx * xxx;
		glm::vec3 myRay;
		if (u01(rng) < f)
		{
			myRay = glm::reflect(pathSegment.ray.direction, normal);
		}
		else
		{
			myRay = glm::refract(pathSegment.ray.direction, normal, ratio);
		}

		pathSegment.ray.origin = intersect + myRay * 1e-3f;
		pathSegment.ray.direction = myRay;
		pathSegment.color *= m.specular.color;
		pathSegment.remainingBounces--;
#endif
 	}
	else

	{
		// diffuse
		glm::vec3 randomRay = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.color *= m.color;
		pathSegment.ray.direction = randomRay;
		pathSegment.ray.origin = intersect;
		pathSegment.remainingBounces--;
	}
#else
	pathSegment.ray.origin = intersect;
	if (m.hasReflective)
	{
		thrust::uniform_real_distribution<float> u01(0, 1);
		float temp = u01(rng);
		glm::vec3 reflectedRay = glm::reflect(pathSegment.ray.direction, normal);		
		pathSegment.color = pathSegment.color * m.specular.color * temp + pathSegment.color * m.color * (1 - temp);

		//pathSegment.color = pathSegment.color * m.specular.color * float(0.5) +  pathSegment.color * m.color * float(0.5);		
		//pathSegment.color *= (m.specular.color * glm::abs(glm::dot(reflectedRay, normal)));
		// Because with more refleciton, the color needs to multiply more colors and will be darker,
		// so we do not need mulatiply glm::dot(ray, normal) to it?
		pathSegment.ray.origin = intersect;
		pathSegment.ray.direction = reflectedRay;
		pathSegment.remainingBounces--;
	}
	else
	{
		glm::vec3 randomRay = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.color *= m.color;
		pathSegment.ray.direction = randomRay;
		pathSegment.ray.origin = intersect;
		pathSegment.remainingBounces--;
	}
#endif	
}


