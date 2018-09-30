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

// from my CIS 561: Advanced Computer Graphics CPU Pathtracer Implementation

__host__ __device__
glm::vec3 squareToSphereUniform(const glm::vec2 &sample) {
  // we start with:
  // x = sin(theta)cos(phi)
  // y = sin(theta)sin(phi)
  // z = 1 − 2cos(theta)
  // then we set  Let’s take theta = 2ρξ2 and phi = cos-1(ξ1)
  // so we get the following:
  //      x = cos(2ρξ2)√(1 − z^2)
  //      y = sin(2ρξ2)√(1 − z^2)
  //      z = 1 − 2ξ1

  float z = 1.0f - 2.0f * sample[0];
  float y = glm::sin(2 * PI * sample[1]) * glm::sqrt(1.0f - glm::pow(z, 2.0f));
  float x = glm::cos(2 * PI * sample[1]) * glm::sqrt(1.0f - glm::pow(z, 2.0f));
  return glm::vec3(x, y, z);
}

__host__ __device__
void Refract(const glm::vec3 wi, const glm::vec3 &normal, float eta, glm::vec3 *wt) {
  // Compute cos theta using Snell's law
  float cosThetaI = glm::dot(normal, wi);
  float sin2ThetaI = glm::max(float(0), float(1 - cosThetaI * cosThetaI));
  float sin2ThetaT = eta * eta * sin2ThetaI;
  float cosThetaT = glm::sqrt(1 - sin2ThetaT);

  // Handle total internal reflection for transmission
  if (sin2ThetaT >= 1) {
    *wt = glm::reflect(wi, normal);
  }
  else {
    *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * normal;
  }
}

__host__ __device__
glm::vec3 squareToDiskConcentric(const glm::vec2 &sample) {
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
  }
  else {// region 3 or 4
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

// An additional sampling technique for this assignment

__host__ __device__
glm::vec3 squareToBonneProjection(const glm::vec2 &sample) {
  // basically square to heart mapping
  // Based on the Bonne Map Projection technique
  // phi = latitude
  // phi1 = defined by user for final shape manipulation
  // lambda0 = meridian
  // lambda = longitude

  glm::vec3 sphere_sampling = squareToSphereUniform(sample);

  // convert x,y,z [0->1] sphere locations to longitude & latitude
  float latitude = glm::acos(sphere_sampling.y); // theta
  float longitude = glm::atan(sphere_sampling.x / sphere_sampling.z); // phi

  const float LAMBDA_0 = 0;
  const float PHI_1 = 45.f * PI / 180.f;

  // latitude [-90 to 90] and longitude [-180 to 180] but in radians
  float phi = latitude * 90.f * PI / 180.f;
  float lambda = longitude * 180.f * PI / 180.f;

  float tangent_PHI_1 = glm::tan(PHI_1);
  float cotangent_PHI_1 = (tangent_PHI_1 < EPSILON) ? 0 : 1 / tangent_PHI_1;

  float rho = cotangent_PHI_1 + PHI_1 - phi;
  float E = ((lambda - LAMBDA_0)* glm::cos(phi)) / rho;

  return glm::vec3(rho * glm::sin(E), cotangent_PHI_1 - rho * glm::cos(E), 0.f);
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
  return (n[0] * v[0] + n[1] * v[1] + n[2] * v[2]  < 0.f) ? -n : n;
}

/**************** END: ADDITIONAL OPERATIONAL FUNCTIONS ****************/

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
  glm::vec3 original_direction = pathSegment.ray.direction;

  if (probability < m.hasRefractive) {
    // refractive - transmissive

    float comparing_incident_direction = glm::dot(pathSegment.ray.direction, normal);
    bool entering_shape = comparing_incident_direction > 0;

    // flip based on surface direction
    float eta = glm::mix(1.f / m.indexOfRefraction, m.indexOfRefraction, entering_shape);

    // Schlick's Approximation for transmissive surfaces
    float r0 = powf((1.f - eta) / (1.f + eta), 2.f);
    float rTheta = r0 + (1 - r0) * powf(1 - glm::abs(comparing_incident_direction), 5.f);

    // bounce based on internal reflection or refracting through surface
    glm::vec3 refract = glm::normalize(glm::refract(pathSegment.ray.direction, normal, eta));
    glm::vec3 reflect = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
    bool flip = rTheta < u01(rng);
    pathSegment.ray.direction = glm::mix(reflect, refract, flip);
    pathSegment.ray.origin = intersect + EPSILON * glm::mix(normal, -normal, flip);

    pathSegment.color *= m.specular.color;
  } else if (probability < m.hasReflective) {
    // reflective

    pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
    pathSegment.ray.origin = intersect + EPSILON * normal;
    pathSegment.color *= m.specular.color;
  } else {
    // pure diffuse
    pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
    pathSegment.ray.origin = intersect + EPSILON * normal;
  }
  

  pathSegment.color *= m.color;//* AbsDot(normal, original_direction);
  pathSegment.remainingBounces--;
}

/********** END: RAY MANIPULATIONS **********/