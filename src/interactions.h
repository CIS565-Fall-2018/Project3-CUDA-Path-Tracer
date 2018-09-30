#pragma once

#include "intersections.h"

#define SUBSURFACE 1
#define PENETRATE_DEPTH 0.0002f
#define SCATTER_LENGTH 0.1f // average length between scatters

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
	// cos of angle between incident vector and normal
	float cos_i = fabs(glm::dot(incident, normal));
	// sine of angle squared
	float sin_i2 = 1 - cos_i * cos_i;

	// cosine of transmitted angle
	float cos_t = sqrt(1 - sin_i2 / (n * n));
	// transmitted vector
	glm::vec3 refracted = (incident / n) + ((cos_i / n) - cos_t) * normal;
	return refracted;
}

__device__
glm::vec3 calculateRandomDirectionInSphere(thrust::default_random_engine &rng) {
	thrust::minstd_rand rng2;
	thrust::uniform_real_distribution<float> u01(0, 1);
	float phi = u01(rng) * TWO_PI;

	float a = u01(rng) * 2 - 1; // make distribution -1 to 1
	float costheta;
	if (a < 0) {
		costheta = -sqrt(-a);
	}
	else costheta = sqrt(a);

	float sintheta = sqrt(1 - costheta * costheta); // sin(theta)

	glm::vec3 rand_direction;

	rand_direction.x = sintheta * cos(phi);
	rand_direction.y = sintheta * sin(phi);
	rand_direction.z = costheta;

	return rand_direction;
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
	Ray & ray = path.ray;
	
	float length = glm::length(intersect - ray.origin);
	ray.origin = intersect;

	float n = material.indexOfRefraction;
	if (path.inside == true) {
		n = 1 / n;
		path.color *= exp(-length * material.specular.exponent);
	}
	float R = schlickApprox(n, normal, ray.direction);
	float T = 1 - R;
	glm::vec3 incident = ray.direction;

	thrust::uniform_real_distribution<float> u01(0, 1);
	if (u01(rng) > T) {
		ray.direction = calculateIdealReflect(normal, incident);
	}
	else {
		ray.origin += (glm::normalize(ray.direction) - glm::normalize(normal)) * PENETRATE_DEPTH; // march ray through surface
		ray.direction = calculateIdealRefract(normal, incident, n); // get new refracted direction
	}
	path.color *= material.specular.color * R + material.color * T;
}

__device__ void shadeRefractive(PathSegment & path, Material material, glm::vec3 intersect, glm::vec3 normal, thrust::default_random_engine rng) {
	Ray & ray = path.ray;

	float n = material.indexOfRefraction;
	if (path.inside == true) n = 1 / n;
	glm::vec3 incident = ray.direction;

	float length = glm::length(intersect - ray.origin);
	ray.origin = intersect;

	ray.origin += (glm::normalize(ray.direction) - glm::normalize(normal)) * PENETRATE_DEPTH; // march ray through surface
	ray.direction = calculateIdealRefract(normal, incident, n); // get new refracted direction

	// exponent accounts for absorbancy of material
	path.color *= material.color * exp(-length * material.specular.exponent);
}

__device__ void shadeSubsurface(PathSegment & path, Material material, glm::vec3 intersect, glm::vec3 normal, thrust::default_random_engine rng) {
	Ray & ray = path.ray;

	// enter material if not inside
	if (!path.inside) {
		ray.origin = intersect;
		ray.origin += (glm::normalize(ray.direction) - glm::normalize(normal)) * PENETRATE_DEPTH; // march ray through surface
		ray.direction = calculateRandomDirectionInHemisphere(-normal, rng); // hemisphere diffuse around neg. normal
		path.color *= material.color; // mix material color into ray
	}
	// else we need bounce the ray after some scattering distance
	else {
		float length = glm::length(intersect - ray.origin);
		thrust::random::normal_distribution<float> dist(SCATTER_LENGTH, SCATTER_LENGTH / 3.0f);
		thrust::minstd_rand rng2;
		float scatter = dist(rng2); // distance along ray we do scatter
		if (scatter <= 0 || scatter >= length) {
			// if we don't scatter we go out of the material and diffuse
			ray.origin = intersect;
			ray.origin += (glm::normalize(ray.direction) - glm::normalize(normal)) * PENETRATE_DEPTH; // march ray through surface
			ray.direction = calculateRandomDirectionInHemisphere(-normal, rng); // hemisphere diffuse around neg. normal
			path.color *= material.color * exp(-length * material.specular.exponent); // mix material color into ray w/ some absorbancy
		}
		else {
			// scatter in sphere around scatter point
			ray.origin = getPointOnRay(ray, scatter); // point ray travels to before scatter
			// generate new direction
			ray.direction = calculateRandomDirectionInSphere(rng);
			// color ray
			path.color *= material.color * exp(-scatter * material.specular.exponent); // mix material color into ray w/ some absorbancy
		}


	}

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

	//thrust::uniform_real_distribution<float> u01(0, 1);
	//path.ray.origin = intersect;

	
	path.remainingBounces--;

	// if has emmitance (light) add in light factor
	if (m.emittance > 0.0f) {

		path.color *= m.color * m.emittance;

		// terminate path
		path.remainingBounces = 0;
		return;
	}


	if (m.hasRefractive && m.hasReflective) {
		// shader where both reflective and refraction elements accounted for
		shadeReflectRefract(path, m, intersect, normal, rng);
	}
	else if (m.hasRefractive) {
		// pure refractive shader
		shadeRefractive(path, m, intersect, normal, rng);
	}
	else if (m.hasReflective) {
		path.ray.origin = intersect;
		//reflective shader
		shadeReflective(path, m, normal);
	}
	else if (SUBSURFACE && m.specular.exponent > 0) {
		// subsurface scattering
		shadeSubsurface(path, m, intersect, normal, rng);
	}
	else {
		path.ray.origin = intersect;
		// generic diffuse shading
		shadeDiffuse(path, m, rng, normal);
	}

}
