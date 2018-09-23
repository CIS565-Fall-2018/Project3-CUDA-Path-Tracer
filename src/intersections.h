#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, glm::vec3 &bitangent, glm::vec3 &tangent, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

    tangent = glm::normalize(glm::mat3(sphere.transform) * glm::cross(Vector3f(0,1,0), (glm::normalize(objspaceIntersection))));
    bitangent = glm::normalize(glm::cross(normal, tangent));

    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


namespace Shapes
{

  namespace SquarePlane
  {
    __host__ __device__ inline float Intersect(const Geom& shape, const Ray& r, glm::vec3& intersectionPoint, glm::vec3 &normal, glm::vec3 &bitangent, glm::vec3 &tangent)
    {
      //Transform the ray
      glm::vec3 ro = multiplyMV(shape.inverseTransform, glm::vec4(r.origin, 1.0f));
      glm::vec3 rd = glm::normalize(multiplyMV(shape.inverseTransform, glm::vec4(r.direction, 0.0f)));

      //Ray-plane intersection
      float t = glm::dot(glm::vec3(0,0,1), (glm::vec3(0.5f, 0.5f, 0) - ro)) / glm::dot(glm::vec3(0,0,1), rd);
      Point3f P = Point3f(t * rd + ro);
      //Check that P is within the bounds of the square
      if(t > 0 && P.x >= -0.5f && P.x <= 0.5f && P.y >= -0.5f && P.y <= 0.5f)
      {
        intersectionPoint = multiplyMV(shape.transform, glm::vec4(P, 1.f));
        
        normal = glm::normalize(glm::mat3(shape.invTranspose) * Normal3f(0,0,1));
        tangent = glm::normalize(glm::mat3(shape.transform) * Normal3f(1,0,0));
        bitangent = glm::normalize(glm::mat3(shape.transform) * Normal3f(0,1,0));

        return t;
      }
      return -1.0f;
    }

    __host__ __device__ inline void Sample(Geom* geometry, const float rngX, const float rngY, Float *pdf, Intersection* intr) {
      *pdf = 1.0f / (geometry->scale[0] * geometry->scale[1]);

      const Point3f localPoint = Point3f(Point2f(rngX, rngY) - Point2f(0.5f, 0.5f), 0.0f);
      const Point3f worldPoint = Point3f(geometry->transform * glm::vec4(localPoint, 1.0f));

      intr->point = worldPoint;
      intr->normal = glm::normalize(glm::mat3(geometry->inverseTransform)  * Normal3f(0, 0, 1));
    }
  }

  __host__ __device__ inline void Sample(Geom* geometry, const float rngX, const float rngY, float *pdf, Intersection* intr)
  {
    if (geometry->type == SQUAREPLANE) {
      SquarePlane::Sample(geometry, rngX, rngY, pdf, intr);
    }
  }

  __host__ __device__ inline Intersection Sample(Geom* geometry, const Point3f& refPoint, const float rngX, const float rngY, float *pdf)
  {
    Intersection isectLight;
    Sample(geometry, rngX, rngY, pdf, &isectLight);

    Vector3f wi = glm::normalize(isectLight.point - refPoint);

    Vector3f toRef = refPoint - isectLight.point;
    float dist = glm::length2(toRef);

    float dot = glm::abs(glm::dot(isectLight.normal, -wi));

    if (dot <= 0.00001) {
      *pdf = 0;
      return isectLight;
    }

    *pdf = (dist) / (dot / (*pdf));

    return isectLight;
  }
}

namespace Intersections
{
  __host__ __device__ inline ShadeableIntersection Do(Ray targetRay, Geom* geometries, int geometrySize)
  {
    float t = -1.0f;
    glm::vec3 intersectPoint;
    glm::vec3 normal;
    glm::vec3 tangent;
    glm::vec3 bitangent;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    glm::vec3 tmp_bitangent;
    glm::vec3 tmp_tangent;

    // naive parse through global geoms

    for (int i = 0; i < geometrySize; i++)
    {
      Geom& geom = geometries[i];

      if (geom.type == CUBE)
      {
        t = boxIntersectionTest(geom, targetRay, tmp_intersect, tmp_normal, outside);
      }
      else if (geom.type == SPHERE)
      {
        t = sphereIntersectionTest(geom, targetRay, tmp_intersect, tmp_bitangent, tmp_tangent, tmp_normal, outside);
      }
      else if (geom.type == SQUAREPLANE)
      {
        t = Shapes::SquarePlane::Intersect(geom, targetRay, tmp_intersect, tmp_bitangent, tmp_tangent, tmp_normal);
      }
      // TODO: add more intersection tests here... triangle? metaball? CSG?

      // Compute the minimum t from the intersection tests to determine what
      // scene geometry object was hit first.
      if (t > EPSILON && t_min > t)
      {
        t_min = t;
        hit_geom_index = i;
        intersectPoint = tmp_intersect;
        normal = tmp_normal;
        bitangent = tmp_bitangent;
        tangent = tmp_tangent;
      }
    }

    ShadeableIntersection result;
    if (hit_geom_index == -1)
    {
      result.t = -1.0f;
      result.geom = nullptr;
      result.materialId = -1;
    }
    else
    {
      // The ray hits something
      result.t = t_min;
      result.geom = &geometries[hit_geom_index];
      result.materialId = geometries[hit_geom_index].materialid;
      result.intersectPoint = intersectPoint;
      result.surfaceNormal = normal;
      result.surfaceTangent = tangent;
      result.surfaceBitangent = bitangent;
      result.tangentToWorld = glm::mat3(
        tangent,
        bitangent,
        normal
      );

      result.worldToTangent = glm::transpose(result.tangentToWorld);
    }

    return result;
  }

  __host__ __device__ inline Ray SpawnRay(const Point3f& origin, const Normal3f& normal, const Vector3f &d)
  {
    Vector3f originOffset = normal * RayEpsilon;
    // Make sure to flip the direction of the offset so it's in
    // the same general direction as the ray direction
    originOffset = (glm::dot(d, normal) > 0) ? originOffset : -originOffset;
    const Point3f o(origin + originOffset);
    return Ray{o, d};
  }
}