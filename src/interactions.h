#pragma once

#include "intersections.h"

#define THRESH_INTERNAL_REFLECTION .01f

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

__host__ __device__
void reflect(PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &material,
        glm::vec3 direction) {
	pathSegment.color *= material.color;
	pathSegment.ray.direction = glm::reflect(direction, normal);
	pathSegment.ray.origin = intersect + .0001f * pathSegment.ray.direction;
}

__host__ __device__
void refract(PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &material,
        glm::vec3 direction) {
	float refractive_index = material.indexOfRefraction;
	// Not sure why this fixes the bug...
	if (glm::dot(direction, normal) < 0) {
		refractive_index = 1.0f / refractive_index;
	} else {
		normal = -normal;
	}

	if (glm::length(pathSegment.ray.direction) > THRESH_INTERNAL_REFLECTION) {
		pathSegment.ray.direction = glm::refract(direction, normal, refractive_index);
	} else {
		pathSegment.ray.direction = glm::reflect(direction, normal);
	}

	pathSegment.color *= material.color;
	pathSegment.ray.origin = intersect + .001f * pathSegment.ray.direction;
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
        const Material &material,
        thrust::default_random_engine &rng) {
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	glm::vec3 direction = glm::normalize(pathSegment.ray.direction);

	if (material.hasReflective && material.hasRefractive) {
		thrust::uniform_real_distribution<float> u01(0,1);

		float cosAngle = glm::dot(direction, normal);
		if (cosAngle > 1) {
			cosAngle = 1;
		} else if (cosAngle < -1) {
			cosAngle = -1;
		}

		float refractive_index_before = 1;
		float refractive_index_after;

		if (cosAngle > 0) {
			std::swap(refractive_index_before, refractive_index_after);
		} else {
			cosAngle *= -1;
			pathSegment.ray.origin = intersect + .001f * direction;
		}

		float refractive_index = refractive_index_before / refractive_index_after;
		float sinAngleBefore = 1 - std::pow(cosAngle, 2);
		if (sinAngleBefore > 0) {
			sinAngleBefore = std::sqrt(sinAngleBefore);
		} else {
			sinAngleBefore = 0;
		}
		float sinAngleAfter = refractive_index * sinAngleBefore;

		if (sinAngleAfter > 1) {
			pathSegment.color *= material.specular.color;
			pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
		} else {
			//Shlick
			//R = R_0 + (1 - R_0) * (1 - cos(theta))^5
			//R_0 = [(n_1 - n_2) / (n_1 + n_2)]^2

			float r0 = (refractive_index_before - refractive_index_after) / (refractive_index_before + refractive_index_after);
			r0 *= r0;
			float cosAngleRaised = std::pow(1 - cosAngle, 5);
			float R = r0 + (1 - r0) * cosAngleRaised;

			//Randomize
			if (R < u01(rng)) {
				refract(pathSegment, intersect, normal, material, direction);
			} else {
				reflect(pathSegment, intersect, normal, material, direction);
			}
		}

	} else if (material.hasReflective) {
		reflect(pathSegment, intersect, normal, material, direction);
	} else if (material.hasRefractive) {
		refract(pathSegment, intersect, normal, material, direction);
	} else {
		//Diffuse
		pathSegment.color *= material.color;
		pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.ray.origin = intersect + .001f * pathSegment.ray.direction;
	}

}
