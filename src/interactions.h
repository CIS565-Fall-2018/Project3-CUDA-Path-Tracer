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

__host__ __device__
glm::vec3 diffuse(const glm::vec3 &normal, thrust::default_random_engine &rng) {
	return calculateRandomDirectionInHemisphere(normal, rng);
}

__host__ __device__
glm::vec3 reflect(const glm::vec3 &normal, const glm::vec3 &inDir, thrust::default_random_engine &rng) {
	return glm::reflect(inDir, normal);
}

__host__ __device__
glm::vec3 refract(const glm::vec3 &normal, const glm::vec3 &inDir, const float ior, thrust::default_random_engine &rng) {
	bool out = glm::dot(normal, inDir) >= 0;
	float index = out ? ior : 1.f / ior;
	glm::vec3 n = out ? -normal : normal;
	return glm::refract(inDir, n, index);
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
	
	thrust::uniform_real_distribution<float> u01(0, 1);
	float sample = u01(rng);

	glm::vec3 inDir = pathSegment.ray.direction;
	glm::vec3 newDir; glm::vec3 newDirReflect;
	float reflectionCoefficient, cosTheta, R_o;
	
	if (sample >= 0 && sample < m.hasDiffuse) { // diffuse strength is between 0 and 1
		newDir = diffuse(normal, rng);
		pathSegment.color *= m.color;
	}
	else { // material has some reflectivity and/or refractivity
		if (m.hasReflective > 0 && m.hasRefractive < EPSILON) { // pure reflective
			newDir = reflect(normal, inDir, rng);
		}
		else { // refractive will have some reflection
			float sample2 = u01(rng);
			newDirReflect = reflect(normal, inDir, rng);
			// applying Shlick's approximation
			cosTheta = glm::dot(normal, newDirReflect) / (glm::length(normal) * glm::length(newDirReflect));
			R_o = ((1 - m.indexOfRefraction) / (1 + m.indexOfRefraction)) * ((1 - m.indexOfRefraction) / (1 + m.indexOfRefraction));
			reflectionCoefficient = R_o + (1 - R_o) * (1 - cosTheta) * (1 - cosTheta) * (1 - cosTheta) * (1 - cosTheta) * (1 - cosTheta);

			if (sample2 <= reflectionCoefficient) { // reflect
				newDir = newDirReflect;
			}
			else { // refract
				newDir = refract(normal, inDir, m.indexOfRefraction, rng);
			}
		}
	}

	pathSegment.remainingBounces -= 1;
	pathSegment.ray.direction = glm::normalize(newDir);
	pathSegment.ray.origin = intersect + newDir * 0.001f;
}
