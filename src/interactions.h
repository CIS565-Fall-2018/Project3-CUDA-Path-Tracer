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

__forceinline__
__host__ __device__
bool Refract(const glm::vec3 &wi, const glm::vec3 &n, float eta,
					glm::vec3 *wt) {
	// Compute cos theta using Snell's law
	float cosThetaI = glm::dot(n, wi);
	float sin2ThetaI = glm::max(float(0), float(1 - cosThetaI * cosThetaI));
	float sin2ThetaT = eta * eta * sin2ThetaI;

	// Handle total internal reflection for transmission
	if (sin2ThetaT >= 1) return false;
	float cosThetaT = glm::sqrt(1 - sin2ThetaT);
	*wt = eta * -wi + (eta * cosThetaI - cosThetaT) * glm::vec3(n);
	return true;
}

__forceinline__
__host__ __device__
void SpecularBTDF(PathSegment & pathSegment, glm::vec3 intersect,
				  glm::vec3 normal, const Material &m, thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	glm::vec3 wo = -pathSegment.ray.direction;
	float VdotN = glm::dot(wo, normal);
	bool leaving = VdotN < 0.f;
	glm::vec3 n = normal * (leaving ? -1.f : 1.f);
	float eta = leaving ? m.indexOfRefraction : (1.f / m.indexOfRefraction);
	float d = m.dispersion;
	Refract(wo, n, eta, &pathSegment.ray.direction);
	pathSegment.ray.origin = intersect + (.001f) * pathSegment.ray.direction;
	pathSegment.color *= m.color;
}

__forceinline__
__host__ __device__
void LambertBRDF(PathSegment &pathSegment, glm::vec3 intersect,
				 glm::vec3 normal, const Material &m, thrust::default_random_engine &rng) {
	pathSegment.ray.direction = calculateRandomDirectionInHemisphere(glm::normalize(normal), rng);
	glm::vec3 f = m.color * INV_PI;
	float absDot = glm::abs(glm::dot(glm::normalize(-pathSegment.ray.direction), normal));
	float pdf = absDot * INV_PI;
	pathSegment.color *= f * absDot / pdf;
	pathSegment.ray.origin = intersect + (.0001f) * pathSegment.ray.direction;
}

__forceinline__
__host__ __device__
void SpecularBRDF(PathSegment &pathSegment, glm::vec3 intersect,
				  glm::vec3 normal, const Material &m, thrust::default_random_engine &rng) {
	pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
	glm::vec3 f = m.specular.color;
	float pdf = 1.f;
	pathSegment.color *= f;
	pathSegment.ray.origin = intersect + (.0001f) * pathSegment.ray.direction;
}

__forceinline__
__host__ __device__
void Fresnel(PathSegment &pathSegment, glm::vec3 intersect,
			 glm::vec3 normal, const Material &m, thrust::default_random_engine &rng) {
	float cosi = glm::clamp(-1.f, 1.f, glm::dot(pathSegment.ray.direction, normal));
	float etai = 1.f, etat = m.indexOfRefraction;
	etai = cosi > 0 ? etat : etai;
	etat = cosi > 0 ? etai : etat;
	float sint = etai / etat * glm::sqrt(glm::max(0.f, 1.f - cosi * cosi));
	float kr = 0;
	if (sint >= 1) {
		// total internal reflection
		SpecularBRDF(pathSegment, intersect, normal, m, rng);
	} else {
		float cost = glm::sqrt(glm::max(0.f, 1.f - sint * sint));
		cosi = glm::abs(cosi);
		float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
		float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
		kr = (Rs * Rs + Rp * Rp) / 2;
	}
	thrust::uniform_real_distribution<float> u01(0, 1);
	if (u01(rng) < kr) {
		SpecularBRDF(pathSegment, intersect, normal, m, rng);
	} else {
		SpecularBTDF(pathSegment, intersect, normal, m, rng);
	}
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
        thrust::default_random_engine &rng,
		glm::vec3 texColor = glm::vec3(-1)) {
	if (m.hasReflective && m.hasRefractive) {
		// Fresnel
		Fresnel(pathSegment, intersect, normal, m, rng);
	} else if (m.hasReflective) {
		SpecularBRDF(pathSegment, intersect, normal, m, rng);
	} else if (m.hasRefractive) {
		SpecularBTDF(pathSegment, intersect, normal, m, rng);
	} else {
		LambertBRDF(pathSegment, intersect, normal, m, rng);
	}
}
