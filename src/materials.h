#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"
#include "warpfunctions.h"

namespace Materials {

} // namespace Materials

namespace BRDF {
  namespace Lambert {
    __host__ __device__ inline Color3f f(const Color3f& albedo, const Vector3f &wo, const Vector3f &wi)
    {
      return albedo * float(InvPi);
    }

    __host__ __device__ inline float Pdf(const Vector3f &wo, const Vector3f &wi)
    {
      return SameHemisphere(wo, wi) ? Warp::SquareToHemisphereCosinePDF(wi) : 0;
    }

    __host__ __device__ inline Color3f Sample_f(const Color3f& albedo, const Vector3f &wo, Vector3f *wi, Float *pdf, float rngX, float rngY)
    {
      *wi = Warp::SquareToHemisphereCosine(rngX, rngY);
      if (wo.z < 0.0) {
        wi->z *= -1;
      }

      *pdf = Pdf(wo, *wi);
      return f(albedo, wo, *wi);
    }
  }
} // namespace Materials
