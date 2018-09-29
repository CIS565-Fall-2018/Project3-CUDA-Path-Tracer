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

__host__ __device__ glm::vec3 refract(glm::vec3 i, glm::vec3 n, float idxFrom, float idxTo)
{
	i = glm::normalize(i);
	n = glm::normalize(n);
	float coso = glm::dot(n, i);
	if (coso < 0) {
		n = -n;
		coso = glm::dot(n, i);
	}
	//glm::vec3 reflect = i + 2 * coso * n;

	float ratio = (idxFrom / idxTo);
	glm::vec3 refract = ratio * i +
		((ratio * coso) - glm::sqrt(1 - glm::pow(ratio, 2) * (1 - glm::pow(coso, 2)))) * n;
	return refract;
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
	bool rayOutside,
	glm::vec3 normal,
	const Material &m,
	thrust::default_random_engine &rng)
{
	glm::vec3 newDir(0);

	thrust::uniform_real_distribution<float> u01(0, 1);
	float p = u01(rng);

	if (m.hasRefractive > p) {
		// refraction
		float eta = rayOutside ? 1 / m.indexOfRefraction : m.indexOfRefraction;
		glm::vec3 n = normal;// rayOutside ? normal : -normal;

							 //newDir = glm::refract(pathSegment.ray.direction, n, eta);

		float from = 1;
		float to = m.indexOfRefraction;
		if (!rayOutside) {
			from = m.indexOfRefraction;
			to = 1;
		}
		newDir = refract(pathSegment.ray.direction, normal, from, to);
		/*if (rayOutside) {
		newDir = glm::refract(pathSegment.ray.direction, normal, 1/to);
		}
		else {
		newDir = glm::refract(pathSegment.ray.direction, -normal, to);
		}*/

		pathSegment.color *= m.specular.color * m.hasRefractive;
	}
	else if (m.hasReflective > p) {
		// reflection
		newDir = glm::reflect(pathSegment.ray.direction, normal);
		pathSegment.color *= m.specular.color * m.hasReflective;
	}
	else {
		//diffuse
		newDir = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.color *= m.color * (1 - (m.hasReflective + m.hasRefractive));
	}

	pathSegment.ray.direction = newDir;
	pathSegment.ray.origin = intersect;
}
