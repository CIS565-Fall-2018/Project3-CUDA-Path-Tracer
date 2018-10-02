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
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) 
{
    glm::vec3 refraction_ray;

    //fresnel
    float r_result = 0.0f;

    const glm::vec3 path_bounce_vec = [&]()
    {
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
        const float reflect_refract_diffuse_rand = u01(rng);

        const float reflective_percentage_range = m.hasReflective;
        const float refractive_percentage_range = m.hasRefractive + reflective_percentage_range;
        const float diffuse_percentage_range = 1.0f - refractive_percentage_range;

        //both reflective/refractive
        if(m.hasReflective && m.hasRefractive && reflect_refract_diffuse_rand < refractive_percentage_range)
        {
            //compute fresnel
            float dot = glm::dot(pathSegment.ray.direction, normal);
            float ior = m.indexOfRefraction;
            glm::vec3 n = normal;
            if(dot >= 0.0f)
            {
                ior = 1.0f / ior;
            }
            else
            {
                n = -n;
            }

            // Schlick Approximation
            float n1 = ior;
            float n2 = 1.0f / ior;
            float r0 = (n1 - n2) / (n1 + n2) * (n1 - n2) / (n1 + n2);
            float r_result = r0 + (1 - r0) * std::pow((1 - std::cos(ior)), 5.0f);

            const float randomness = u01(rng);

            //split based on randomness
            if(randomness < reflective_percentage_range)
            {
                refraction_ray = m.hasReflective * glm::reflect(pathSegment.ray.direction, normal);
                return refraction_ray;
            }
            else
            {
               refraction_ray = m.hasRefractive * glm::refract(pathSegment.ray.direction, n, ior);
            }

            return refraction_ray;
        }

        if (reflect_refract_diffuse_rand < reflective_percentage_range)
        {
            //reflective
            return glm::reflect(pathSegment.ray.direction, normal);
        }
        if (reflect_refract_diffuse_rand > reflective_percentage_range 
            && reflect_refract_diffuse_rand < refractive_percentage_range)
        {
            float dot = glm::dot(-pathSegment.ray.direction, normal);
            float ior = m.indexOfRefraction;
            glm::vec3 n = normal;
            if(dot >= 0.0f)
            {
                ior = 1.0f / ior;
            }
            else
            {
                n = -n;
            }
            refraction_ray = glm::refract(pathSegment.ray.direction, n, ior);

            //total internal reflection
            if (refraction_ray.length() <= 0.1f)
            {
                //reflect instead
                return glm::reflect(pathSegment.ray.direction, normal);
            }
            return refraction_ray;
        }

        //diffuse
        return calculateRandomDirectionInHemisphere(normal, rng);
    }();

    const glm::vec3 color = [&]()
    {
        glm::vec3 result{};
        const float reflective_percentage = m.hasReflective;
        const float refractive_percentage = m.hasRefractive;
        const float diffuse_percentage = 1.0f - reflective_percentage - refractive_percentage;

        //both reflective/refractive
        if(r_result)
        {
            //calculate from both reflective + refractive
            return (1.0f - r_result) * m.color + r_result * m.specular.color;
        }

        if (m.hasReflective)
        {
            //reflective
            result += reflective_percentage * m.specular.color;
        }
        if (m.hasRefractive)
        {
            //total internal reflection
            if (refraction_ray.length() <= 0.1f)
            {
                result += glm::vec3(0.0f);
            } 
            else
            {
                //refractive
                result += refractive_percentage * m.color;
            }
        }

        //diffuse
        result += diffuse_percentage * m.color;
        //printf("%lf\n", diffuse_percentage);
        return result;
    }();

    pathSegment.ray.origin = std::move(intersect) + path_bounce_vec * 0.01f;
    pathSegment.ray.direction = std::move(path_bounce_vec);
    thrust::uniform_real_distribution<float> u01(0, 1);
    float random_between_0_and_1 = u01(rng);
    
    pathSegment.color = pathSegment.color * std::move(color);

    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
}
