#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"
#include "warpfunctions.h"

namespace Materials
{
} // namespace Materials

enum FresnelType
{
  FRESNEL_NOOP,
  FRESNEL_NOREFLECT,
  FRESNEL_DIELECTRIC,
  FRESNEL_CONDUCTOR
};

enum MicrofacetType
{
  TROWBRIDGE_REITZ
};

namespace BRDF
{
  __host__ __device__ inline float CosTheta(const Vector3f& w) { return w.z; }
  __host__ __device__ inline float Cos2Theta(const Vector3f& w) { return w.z * w.z; }
  __host__ __device__ inline float AbsCosTheta(const Vector3f& w) { return std::abs(w.z); }

  __host__ __device__ inline float Sin2Theta(const Vector3f& w)
  {
    return glm::max((float)0, (float)1 - Cos2Theta(w));
  }

  __host__ __device__ inline float SinTheta(const Vector3f& w) { return std::sqrt(Sin2Theta(w)); }

  __host__ __device__ inline float TanTheta(const Vector3f& w) { return SinTheta(w) / CosTheta(w); }

  __host__ __device__ inline float Tan2Theta(const Vector3f& w)
  {
    return Sin2Theta(w) / Cos2Theta(w);
  }


  __host__ __device__ inline float CosPhi(const Vector3f& w)
  {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
  }

  __host__ __device__ inline float SinPhi(const Vector3f& w)
  {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
  }

  __host__ __device__ inline float Cos2Phi(const Vector3f& w) { return CosPhi(w) * CosPhi(w); }

  __host__ __device__ inline float Sin2Phi(const Vector3f& w) { return SinPhi(w) * SinPhi(w); }


  __host__ __device__ inline  bool Refract(const Vector3f &wi, const Normal3f &n, float eta,
    Vector3f *wt) {
    // Compute cos theta using Snell's law
    const float cosThetaI = glm::dot(n, wi);
    const float sin2ThetaI = glm::max(float(0), float(1 - cosThetaI * cosThetaI));
    const float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return false;
    const float cosThetaT = std::sqrt(1 - sin2ThetaT);
    *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * Vector3f(n);
    return true;
  }

  __host__ __device__ inline  Normal3f Faceforward(const Normal3f &n, const Vector3f &v) {
    return (glm::dot(n, v) < 0.f) ? -n : n;
  }

  namespace Fresnel
  {
    __host__ __device__ inline Color3f Dielectric(float cosThetaI, float etaI, float etaT)
    {
      cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

      float etaA = etaI;
      float etaB = etaT;

      bool isEntering = cosThetaI > 0.0f;
      if (!isEntering)
      {
        float temp = etaA;
        etaA = etaB;
        etaB = temp;
        cosThetaI = std::fabs(cosThetaI);
      }

      float sinThetaI = std::sqrt(glm::max(0.0f, 1.0f - cosThetaI * cosThetaI));
      float sinThetaT = etaA / etaB * sinThetaI;

      // TIR
      if (sinThetaT >= 1.0f)
      {
        return Color3f(1.0f);
      }

      float costThetaT = std::sqrt(glm::max(0.0f, 1.0f - sinThetaT * sinThetaT));

      float Rparl = ((etaB * cosThetaI) - (etaA * costThetaT)) /
        ((etaB * cosThetaI) + (etaA * costThetaT));

      float Rperp = ((etaA * cosThetaI) - (etaB * costThetaT)) /
        ((etaA * cosThetaI) + (etaB * costThetaT));

      return Color3f((Rparl * Rparl + Rperp * Rperp) / 2.0f);
    }

    __host__ __device__ inline Color3f Evaluate(FresnelType type, float cosI)
    {
      if (type == FRESNEL_NOOP)
      {
        return Color3f(1.0f);
      }

      return Color3f(0.0f);
    }
  }

  namespace Lambert
  {
    __host__ __device__ inline Color3f f(const Color3f& albedo, const Vector3f& wo, const Vector3f& wi)
    {
      return albedo * float(InvPi);
    }

    __host__ __device__ inline float Pdf(const Vector3f& wo, const Vector3f& wi)
    {
      return SameHemisphere(wo, wi) ? Warp::SquareToHemisphereCosinePDF(wi) : 0;
    }

    __host__ __device__ inline Color3f Sample_f(const Color3f& albedo, const Vector3f& wo, Vector3f* wi, Float* pdf,
                                                float rngX, float rngY)
    {
      *wi = Warp::SquareToHemisphereCosine(rngX, rngY);
      if (wo.z < 0.0)
      {
        wi->z *= -1;
      }

      *pdf = Pdf(wo, *wi);
      return f(albedo, wo, *wi);
    }
  }

  namespace Specular
  {
    __host__ __device__ inline Color3f f(const Vector3f& wo, const Vector3f& wi)
    {
      return Color3f(0.f);
    }

    __host__ __device__ inline float Pdf(const Vector3f& wo, const Vector3f& wi)
    {
      return 0.f;
    }

    __host__ __device__ inline Color3f Sample_f(const Color3f& albedo, const Vector3f& wo, Vector3f* wi, Float* pdf,
                                                FresnelType fresnel)
    {
      *wi = Vector3f(-wo.x, -wo.y, wo.z);

      *pdf = 1; // Since we are reflecting

      if (fresnel == FRESNEL_NOOP)
      {
        return albedo / AbsCosTheta(*wi);
      }

      // Un-implemented fresnel or NoReflect
      return Color3f(0.0f);
    }
  }

  namespace SpecularBTDF
  {
    __host__ __device__ inline Color3f f(const Vector3f& wo, const Vector3f& wi)
    {
      return Color3f(0.f);
    }

    __host__ __device__ inline float Pdf(const Vector3f& wo, const Vector3f& wi)
    {
      return 0.f;
    }

    __host__ __device__ inline Color3f Sample_f(const Color3f& albedo, const Vector3f& wo, Vector3f* wi, Float* pdf,
      FresnelType fresnel, float etaA, float etaB)
    {
      const bool isEnteringMedium = wo.z > 0.0f;
      const float etaI = isEnteringMedium ? etaA : etaB;
      const float etaT = isEnteringMedium ? etaB : etaA;

      const Vector3f correctedNormal = Faceforward(Vector3f(0,0,1), wo);
      const float eta = etaI/etaT;

      if (!Refract(wo, correctedNormal, eta, wi)) {
        return Color3f(0.0f);
      }

      *pdf = 1;

      if (fresnel == FRESNEL_NOREFLECT)
      {
        return albedo / AbsCosTheta(*wi);
      }

      // Un-implemented fresnel or Noop
      return Color3f(0.0f);
    }
  }

  namespace MicrofacetDistribution
  {
    namespace TrowbridgeReitz
    {
      __host__ __device__ inline float D(const Vector3f& wh, float alphax, float alphay)
      {
        float tan2Theta = Tan2Theta(wh);
        if (glm::isinf(tan2Theta)) return 0.f;

        const float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

        float e =
          (Cos2Phi(wh) / (alphax * alphax) + Sin2Phi(wh) / (alphay * alphay)) *
          tan2Theta;
        return 1 / (Pi * alphax * alphay * cos4Theta * (1 + e) * (1 + e));
      }

      __host__ __device__ inline float Lambda(const Vector3f& w, float alphax, float alphay)
      {
        float absTanTheta = std::abs(TanTheta(w));
        if (glm::isinf(absTanTheta)) return 0.;

        // Compute alpha for direction w
        float alpha =
          std::sqrt(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
        float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
        return (-1 + std::sqrt(1.f + alpha2Tan2Theta)) / 2;
      }

      __host__ __device__ inline float G(const Vector3f& wo, const Vector3f& wi, float alphax, float alphay)
      {
        return 1 / (1 + Lambda(wo, alphax, alphay) + Lambda(wi, alphax, alphay));
      }

      __host__ __device__ float Pdf(const Vector3f& wo, const Vector3f& wh, float alphax, float alphay)
      {
        return D(wh, alphax, alphay) * AbsCosTheta(wh);
      }

      __host__ __device__ inline Vector3f Sample_wh(const Vector3f& wo, float rngX, float rngY, float alphax,
                                                    float alphay)
      {
        Point2f xi(rngX, rngY);
        Vector3f wh;
        float cosTheta = 0, phi = (2 * Pi) * xi[1];
        // if (alphax == alphay)
        // {
          float tanTheta2 = alphax * alphax * xi[0] / (1.0f - xi[0]);
          cosTheta = 1 / std::sqrt(1 + tanTheta2);
        // }
        // else
        // {
        //   phi = std::atan(alphay / alphax * std::tan(2 * Pi * xi[1] + .5f * Pi));
        //   if (xi[1] > .5f) { phi += Pi; }
        //   float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
        //   const float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
        //   const float alpha2 = 1 / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
        //   float tanTheta2 = alpha2 * xi[0] / (1 - xi[0]);
        //   cosTheta = 1 / std::sqrt(1 + tanTheta2);
        // }

        float sinTheta = std::sqrt(glm::max(0.0f, 1.0f - cosTheta * cosTheta));

        wh = Vector3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
        if (!SameHemisphere(wo, wh)) wh = -wh;

        return wh;
      }
    }
  }

  namespace Microfacet
  {
    __host__ __device__ inline Color3f f(const Color3f& albedo, const Vector3f& wo, const Vector3f& wi,
                                         FresnelType fresnel, float alphax, float alphay)
    {
      Float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
      Vector3f wh = wi + wo;

      // Edge cases
      if (cosThetaI < 0.0001 || cosThetaO < 0.0001) return Color3f(0.);
      if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Color3f(0.);

      wh = glm::normalize(wh);

      Color3f F = Fresnel::Evaluate(fresnel, glm::dot(wi, wh));

      return albedo * MicrofacetDistribution::TrowbridgeReitz::D(wh, alphax, alphay) * MicrofacetDistribution::
        TrowbridgeReitz::G(wo, wi, alphax, alphay) * F / (4.0f * cosThetaI * cosThetaO);
    }

    __host__ __device__ inline float Pdf(const Vector3f& wo, const Vector3f& wi, float alphax, float alphay)
    {
      if (!SameHemisphere(wo, wi))
      {
        return 0;
      }

      Vector3f wh = glm::normalize(wo + wi);
      return MicrofacetDistribution::TrowbridgeReitz::Pdf(wo, wh, alphax, alphay) / (4.0f * glm::dot(wo, wh));
    }

    __host__ __device__ inline Color3f Sample_f(const Color3f& albedo, const Vector3f& wo, Vector3f* wi, Float* pdf,
                                                FresnelType fresnel, float rngX, float rngY, float alphax, float alphay)
    {
      Vector3f wh = MicrofacetDistribution::TrowbridgeReitz::Sample_wh(wo, rngX, rngY, alphax, alphay);
      
      *wi = glm::normalize(glm::reflect(-wo, wh));
      
      if (!SameHemisphere(wo, *wi)) { return Color3f(0.f); }

      *pdf = MicrofacetDistribution::TrowbridgeReitz::Pdf(wo, wh, alphax, alphay) / (4.0f * glm::dot(wo, wh));

      return f(albedo, wo, *wi, fresnel, alphax, alphay);
    }
  }
} // namespace Materials
