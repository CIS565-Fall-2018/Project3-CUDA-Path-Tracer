#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__device__
glm::vec3 calculateIdealReflect(glm::vec3 normal, glm::vec3 incident) {
	glm::vec3 reflected = incident - 2 * glm::dot(incident, normal) * normal;
	return reflected;
}

__device__
glm::vec3 calculateIdealRefract(glm::vec3 normal, glm::vec3 incident, float n) {
	float cos_i = fabs(glm::dot(incident, normal));
	float sin_i2 = 1 - cos_i * cos_i;
	float cos_t = sqrt(1 - sin_i2 / (n * n));
	glm::vec3 refracted = (incident / n) + ((cos_i / n) - cos_t) * normal;
	return refracted;
}

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

__device__ void shadeDiffuse(PathSegment & path, Material m, thrust::default_random_engine rng, glm::vec3 normal) {
	// select if diffuse material
	path.color *= m.color;
	path.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
}

__device__ void shadeReflective(PathSegment & path, Material m, glm::vec3 normal) {
	// select if reflective material
	path.color *= m.specular.color;
	path.ray.direction = calculateIdealReflect(normal, path.ray.direction);

}

__device__ float schlickApprox(float n, glm::vec3 normal, glm::vec3 incident) {
	float dot = fabs(glm::dot(normal, incident));
	float r0 = pow((1 - n) / (1 + n), 2);
	float R = r0 + (1 - r0) * pow((1 - dot), 5);
	return R;
}

__device__ void shadeReflectRefract(PathSegment & path, Material material, glm::vec3 intersect, glm::vec3 normal, thrust::default_random_engine rng) {
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
	path.color *= material.specular.color * R + material.color * T;
}

__device__ void shadeRefractive(PathSegment & path, Material material, glm::vec3 intersect, glm::vec3 normal, thrust::default_random_engine rng) {
	float n = material.indexOfRefraction;
	if (path.inside == true) n = 1 / n;
	glm::vec3 incident = path.ray.direction;

	path.ray.direction = calculateIdealRefract(normal, incident, n);
	path.ray.origin = getPointOnRay(path.ray, 0.01f);
	path.inside = !path.inside;

	path.color *= material.color;
}


__device__
void scatterRay(
		PathSegment & path,
        glm::vec3 intersect,
		glm::vec3 normal,
        const Material &m,
		thrust::default_random_engine rng
		) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	thrust::uniform_real_distribution<float> u01(0, 1);
	path.ray.origin = intersect;

	
	path.remainingBounces--;

	// if has emmitance (light) add in light factor
	if (m.emittance > 0.0f) {

		path.color *= m.color * m.emittance;

		// terminate path
		path.remainingBounces = 0;
		return;
	}

	
	if (m.hasRefractive && m.hasReflective) {
		shadeReflectRefract(path, m, intersect, normal, rng);
	}
	else if (m.hasRefractive) {
		shadeRefractive(path, m, intersect, normal, rng);
	}
	else if (m.hasReflective) {
		//reflective shader
		shadeReflective(path, m, normal);
	}
	else {
		// generic diffuse shading
		shadeDiffuse(path, m, rng, normal);
	}

}
