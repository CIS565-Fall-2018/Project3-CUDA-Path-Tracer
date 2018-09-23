#pragma once

#include <glm/glm.hpp>
#include "utilities.h"
#include "interactions.h"
#include "sceneStructs.h"

namespace Lights
{
  namespace Arealight
  {
    __host__ __device__ inline Color3f L(const Color3f& color, const Normal3f& isectNormal, const Vector3f& w, bool isTwoSided = true)
    {
      const float dot = glm::dot(isectNormal, w);
      if (!isTwoSided && dot <= 0)
      {
        return Color3f(0.0);
      }

      return color;
    }

    __host__ __device__ inline Color3f Sample_Li(const Color3f& color, const Point3f& isectPoint, const float rngX, const float rngY, Geom* geometry,
      Vector3f *wi, Float *pdf, Intersection* intr, bool isTwoSided = true) {
      
      const Intersection sIntr = Shapes::Sample(geometry, isectPoint, rngX, rngY, pdf);
      const float diff = glm::length2(isectPoint - sIntr.point);

      if (intr != nullptr) {
        *intr = sIntr;
      }

      if (*pdf < 0.00001f || diff < 0.00001f) {
        return Color3f(0.0);
      }

      *wi = glm::normalize(sIntr.point -  isectPoint);

      return L(color, sIntr.normal, -*wi, isTwoSided);
    }

    __host__ __device__ inline float Pdf_Li(const Point3f& isectPoint, const Normal3f& isectNormal, const Vector3f &wi, Geom* shape) {

      Ray shapeRay = Intersections::SpawnRay(isectPoint, isectNormal, wi);
      Vector3f normal, bitangent, tangent;
      Vector2f uv;
      Point3f intrPoint;

      float t = Shapes::SquarePlane::Intersect(*shape, shapeRay, intrPoint, normal, bitangent, tangent, uv);

      if (t < EPSILON) {
        return 0.0f;
      }

      float dist = glm::length(isectPoint - intrPoint);

      float dot = glm::abs(glm::dot(normal, -wi));

      if (dot < 0.0001) {
        return 0.0f;
      }

      float pdf = 1.0f / Shapes::SquarePlane::Area(shape);
      pdf = (dist * dist) / (dot / pdf);

      return pdf;
    }
  }
}
