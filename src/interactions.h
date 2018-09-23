#pragma once

#include "intersections.h"

/*************************************************/
/********** BEGIN: SCATTERING FUNCTIONS **********/
/*************************************************/

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
glm::vec3 squareToDiskConcentric(const glm::vec2 &sample) {
  // sample is made of x and y locs on the square

  float phi, r, u, v;
  float a = 2 * sample.x - 1;
  float b = 2 * sample.y - 1;
  if (a > -b) {
    if (a > b) {
      r = a;
      phi = (PI / 4.f) * (b / a);
    } else { // region 2, also |b| > |a|
      r = b;
      phi = (PI / 4.f) * (2.f - (a / b));
    }
  } else {// region 3 or 4
    if (a < b) {// region 3, also |a| >= |b|, a != 0
      r = -a;
      phi = (PI / 4.f) * (4.f + (b / a));
    } else { // region 4, |b| >= |a|, but a==0 and b==0 could occur.
      r = -b;
      if (b != 0) {
        phi = (PI / 4.f) * (6.f - (a / b));
      } else {
        phi = 0;
      }
    }
  }
  u = r * std::cos(phi);
  v = r * std::sin(phi);

  return glm::vec3(u, v, 0.f);
}

__host__ __device__
glm::vec3 squareToHemisphereCosine(const glm::vec2& sample) {
  glm::vec3 d = squareToDiskConcentric(sample);
  float z = std::sqrt(std::max(0.f, 1 - d.x*d.x - d.y*d.y));

  return glm::vec3(d.x, d.y, z);
}

__host__ __device__
float squareToDiskPDF(const glm::vec3 &sample) {
  return 1.f / (PI);
}

/********** END: SCATTERING FUNCTIONS **********/

/******************************************************************/
/**************** ADDITIONAL OPERATIONAL FUNCTIONS ****************/
/******************************************************************/
// from my CIS 561: Advanced Computer Graphics CPU Pathtracer Implementation

__host__ __device__
float AbsDot(const glm::vec3& a, const glm::vec3& b) {
  return glm::abs(glm::dot(a, b));
}

__host__ __device__
bool SameHemisphere(const glm::vec3 &w, const glm::vec3 &wp) {
  return w.z * wp.z > 0;
}

__host__ __device__
glm::vec3 Faceforward(const glm::vec3 &n, const glm::vec3 &v) {
  return (glm::dot(n, v) < 0.f) ? -n : n;
}

/**************** ADDITIONAL OPERATIONAL FUNCTIONS ****************/

/***********************************************/
/********** BEGIN: RAY MANIPULATIONS ***********/
/***********************************************/

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
  PathSegment &pathSegment, glm::vec3 intersect, glm::vec3 normal,
  const Material &m, thrust::default_random_engine &rng) {

  thrust::uniform_real_distribution<float> u01(0, 1);
  float probability = u01(rng);

  /*float diffuse_probability = 1.f - m.hasReflective - m.hasRefractive;

  if (probability < m.hasReflective) {
    pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
    pathSegment.color *= m.specular.color * m.hasReflective;
  } else if (probability < m.hasRefractive) {
    bool entering_shape = glm::dot(pathSegment.ray.direction, normal) <= 0;
    float eta = 1.f / m.indexOfRefraction;
    glm::vec3 using_normal = normal;
    if (entering_shape) {
      eta = 1.f / eta;
      using_normal *= -1;
    }

    pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, using_normal, eta);
    pathSegment.color *= m.specular.color * m.hasRefractive;
  } else {
    // pure diffuse
    pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
    pathSegment.color *= m.color * diffuse_probability;
  }*/

  pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
  pathSegment.color *= m.color;
  pathSegment.ray.origin = intersect + EPSILON * pathSegment.ray.direction;
}

/********** END: RAY MANIPULATIONS **********/