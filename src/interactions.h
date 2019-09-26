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
void reflective(
    PathSegment &pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    glm::vec3 dir = glm::normalize(pathSegment.ray.direction);
    glm::vec3 nor = glm::normalize(normal);
    pathSegment.ray.direction = glm::reflect(dir, nor);
    pathSegment.color *= m.color;
    pathSegment.ray.origin = intersect + (.001f) * pathSegment.ray.direction;
}

__forceinline__
__host__ __device__
void refractive(
    PathSegment &pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    glm::vec3 dir = glm::normalize(pathSegment.ray.direction);
    glm::vec3 nor = glm::normalize(normal);
    float ior = m.indexOfRefraction;

    // if ray is in the same as normal direction, then ray is inside object
        // surface normal must be flipped so that surface is facing ray
    if (glm::dot(dir, nor) > 0)
    {
        nor = -nor;
    }
    else
    {
        ior = 1.0f / ior;
    }

    // check for total internal reflection

    if (glm::length(pathSegment.ray.direction) < 0.01f) {
        pathSegment.ray.direction = glm::reflect(dir, nor);
    }
    else
    {
        pathSegment.ray.direction = glm::refract(dir, nor, ior);
    }
    
    pathSegment.color *= m.color;
    pathSegment.ray.origin = intersect + (.001f) * pathSegment.ray.direction;
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
		PathSegment &pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) 
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    glm::vec3 dir = glm::normalize(pathSegment.ray.direction);
    glm::vec3 nor = normal;

    if (m.hasReflective && m.hasRefractive)
    {
        thrust::uniform_real_distribution<float> u01(0, 1);

        float cosThetaI = glm::clamp(glm::dot(dir, nor), -1.0f, 1.0f);
        float etaI = 1.f;
        float etaT = m.indexOfRefraction;

        // outside object
        if (cosThetaI < 0.f) {
            cosThetaI = -cosThetaI;
            pathSegment.ray.origin = intersect + (.001f) * dir;
        }
        // inside object
        else 
        {
            float temp = etaI; 
            etaI = etaT; 
            etaT = temp;
            nor = -nor;
            pathSegment.ray.origin = intersect + (.001f) * dir;
        }

        float sinThetaI = std::sqrt(std::max(0.0f, 1.0f - cosThetaI * cosThetaI));
        float sinThetaT = etaI / etaT * sinThetaI;

        // total internal reflection
        if (sinThetaT >= 1.0f) { 
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, nor);
            pathSegment.color *= m.specular.color;
        }
        // use fresnel schlick's approximation
        else {

            // Schlick's Approx
            float r0 = glm::pow((etaI - etaT) / (etaI + etaT), 2.f);
            float fresnel = r0 + (1.0f - r0) * pow(1.f - glm::abs(glm::dot(normal, pathSegment.ray.direction)), 5.f);

            if (fresnel < u01(rng)) {
                refractive(pathSegment, intersect, normal, m, rng);
            }
            else {
                reflective(pathSegment, intersect, normal, m, rng);
            }
        }
        
    }
    // reflective case
    else if (m.hasReflective)
    {
        reflective(pathSegment, intersect, normal, m, rng);
    }
    // refractive case
    else if (m.hasRefractive)
    {
        refractive(pathSegment, intersect, normal, m, rng);
    }
    // diffuse case
    else
    {
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(nor, rng);
        pathSegment.color *= m.color;
        pathSegment.ray.origin = intersect + (.001f) * pathSegment.ray.direction;
    }

}
