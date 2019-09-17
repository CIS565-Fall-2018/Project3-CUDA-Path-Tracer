#pragma once

#include <glm/glm.hpp>
#include "utilities.h"
#include "interactions.h"
#include "sceneStructs.h"

namespace Lights
{
  namespace Arealight
  {
    __device__ inline Color3f L(const Color3f& color, const Normal3f& isectNormal, const Vector3f& w)
    {
      return color;
    }

    __device__ inline Color3f Sample_Li(const Color3f& color, const Point3f& isectPoint, const Normal3f& isectNormal, const float rngX, const float rngY, Geom* geometry,
      Vector3f *wi, Float *pdf) {
      
      const Intersection sIntr = Shapes::Sample(geometry, isectPoint, isectNormal, rngX, rngY, pdf);
      const float diff = glm::length2(isectPoint - sIntr.point);

      if (*pdf < 0.00001f || diff < 0.00001f) {
        return Color3f(0.0);
      }

      *wi = glm::normalize(sIntr.point -  isectPoint);

      return color;
    }

    __device__ inline float Pdf_Li(const Point3f& isectPoint, const Normal3f& isectNormal, const Vector3f &wi, Geom* shape) {

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
