#pragma once

#include "intersections.h"
#include "globals.h"

#define fresnel 1

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

__host__ __device__ void spawnRay(PathSegment &path, glm::vec3 intersection, glm::vec3 normal, glm::vec3 wi)
{
	path.ray.origin = intersection + RayEpsilon * normal;
	path.ray.direction = wi;
}

#ifdef BSDF
//********************************* lambertian bsdf ************************************
__host__ __device__ glm::vec3 diffuseF(const Material &mat)
{
	return mat.color * InvPi;
}

__host__ __device__ float diffusePDF(const glm::vec3 &wi, const glm::vec3 &wo)
{
	return SameHemisphere(wi, wo) ? AbsCosTheta(wi) * InvPi : 0.0f;
}

__host__ __device__ glm::vec3 diffuseSampleF(
	glm::vec3 wo,
	glm::vec3 *wi,
	glm::vec3 normal,
	float *pdf,
	const Material &m,
	thrust::default_random_engine &rng)
{
	*wi = calculateRandomDirectionInHemisphere(normal, rng);
	if (wo.z < 0.0f) wi->z *= -1;
	*pdf = diffusePDF(*wi, wo);
	return diffuseF(m);
}

__host__ __device__ void diffuse(
	PathSegment &path,
	glm::vec3 intersection,
	glm::vec3 normal,
	const Material &m,
	thrust::default_random_engine &rng)
{

	glm::vec3 wi = calculateRandomDirectionInHemisphere(normal, rng);
	wi = glm::normalize(wi);
	path.color *= m.color;

	spawnRay(path, intersection, normal, wi);
}
#endif

//__host__ __device__ glm::vec3 fresnelDielectric(glm::vec3 normal, glm::vec3 direction, float ei, float et) {
//	bool isEnter = glm::dot(normal, direction) > 0;
//	if (!isEnter) {
//		std::swap(ei, et);
//		normal = -normal;
//	}
//	float eta = ei / et;
//	float cosThetaI = glm::clamp(glm::dot(glm::normalize(normal), glm::normalize(direction)), -1.0f, 1.0f);
//	float sinThetaI = std::sqrt(std::max(0.0f, 1.0f - cosThetaI * cosThetaI));
//	float sinThetaT = eta * sinThetaI;
//
//	if (sinThetaT >= 1.0f) return glm::vec3(1.0f);
//	float cosThetaT = std::sqrt(std::max(0.0f, 1.0f - sinThetaT * sinThetaT));
//	float rpar = (et * cosThetaI - ei * cosThetaT) / (et * cosThetaI + ei * cosThetaT);
//	float rper = (ei * cosThetaI - et * cosThetaT) / (ei * cosThetaI + et * cosThetaT);
//
//	return glm::vec3((rpar * rpar + rper * rper) / 2.0f);
//}



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
		int idx,
		PathSegment & path,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	thrust::uniform_real_distribution<float> u0(0, 1);
	float prob = u0(rng);

	glm::vec3 wi;

	if (prob < m.hasReflective) {

		wi = glm::reflect(path.ray.direction, normal);
		wi = glm::normalize(wi);
		path.color *= m.specular.color;

	} 
	else if(prob < m.hasReflective + m.hasRefractive)
	{

		bool isEnter = glm::dot(normal, -path.ray.direction) > 0;
		float etaI = 1.0f, etaO = m.indexOfRefraction;
		glm::vec3 n = normal;
		if (!isEnter) {
			std::swap(etaI, etaO);
			n = -n;
		}
		float eta = etaI / etaO;

		float R0 = (1.0f - eta) / (1.0f + eta);
		R0 *= R0;
		float mc = 1 - glm::dot(glm::normalize(normal), glm::normalize(-path.ray.direction));
		float R = R0 + (1.0f - R0) * glm::pow(mc, 5);
		if (u0(rng) < R) {
			wi = glm::reflect(path.ray.direction, normal);
		}
		else {
			wi = glm::refract(path.ray.direction, normal, eta);
		}
		wi = glm::normalize(wi);
		path.color *= m.specular.color;
		normal = -n;

	}
	else
	{
		wi = calculateRandomDirectionInHemisphere(normal, rng);
		wi = glm::normalize(wi);
		path.color *= m.color;
	}

	spawnRay(path, intersect, normal, wi);

}
