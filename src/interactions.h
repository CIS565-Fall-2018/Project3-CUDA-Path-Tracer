#pragma once

#include "intersections.h"

/**
 * Maps a (u,v) in [0, 1)^2 to a 2D unit disk centered at (0,0). Based on PBRT
 */
__host__ __device__
glm::vec2 calculateConcentricSampleDisk(float u, float v) {
	glm::vec2 uOffset = 2.0f * glm::vec2(u, v) - glm::vec2(1, 1);
	if (uOffset.x == 0 && uOffset.y == 0) {
		return glm::vec2(0.0f);
	}

	float theta, r;
	if (glm::abs(uOffset.x) > glm::abs(uOffset.y)) {
		r = uOffset.x;
		theta = PI / 4 * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = (PI / 2) - (PI / 4 * (uOffset.x / uOffset.y));
	}
	return r * glm::vec2(glm::cos(theta), glm::sin(theta));
}
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
	thrust::default_random_engine &rng)
{
	glm::vec3 newDir(0);
	thrust::uniform_real_distribution<float> u01(0, 1);
	float p = u01(rng);

	if (m.hasRefractive > p) {
		// adjust eta & normal according to direction of ray (inside or outside mat)
		bool inside = glm::dot(pathSegment.ray.direction, normal) > 0.f;
		glm::vec3 tempNormal = normal * (inside ? -1.0f : 1.0f);
		float eta = inside ? m.indexOfRefraction : (1.0f / m.indexOfRefraction);

		// normal refraction
		newDir = glm::refract(pathSegment.ray.direction, tempNormal, eta);

		if (glm::length(newDir) < 0.01f) {
			// total reflection
			pathSegment.color *= 0;
			newDir = glm::reflect(pathSegment.ray.direction, normal);
		}
		else {
			float schlick_coef = powf(1 - max(0.0f, glm::dot(pathSegment.ray.direction, normal)), 5);
			pathSegment.color *= glm::mix(m.specular.color, glm::vec3(1.0f), schlick_coef);
		}	
	}
	else if (m.hasReflective > p) {
		// reflection
		newDir = glm::reflect(pathSegment.ray.direction, normal);
		pathSegment.color *= m.specular.color * m.hasReflective;
	}
	else {
		// diffuse
		newDir = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.color *= m.color * (1 - (m.hasReflective + m.hasRefractive));
	}

	pathSegment.ray.direction = newDir;
	pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.01f;
}
