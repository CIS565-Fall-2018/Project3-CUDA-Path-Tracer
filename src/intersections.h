#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define ENABLE_MESH_BOUNDING_CULL

#define KD_TREE_QUEUE_SIZE 128

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
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
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t)
{
  return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
  return glm::vec3(m * v);
}

// Returns +/- [0, 2]
__host__ __device__ int GetFaceIndex(const Point3f& P)
{
  int idx = 0;
  float val = -1;
  for (int i = 0; i < 3; i++)
  {
    if (glm::abs(P[i]) > val)
    {
      idx = i * glm::sign(P[i]);
      val = glm::abs(P[i]);
    }
  }
  return idx;
}

__host__ __device__ Normal3f GetCubeNormal(const Point3f& P)
{
  int idx = glm::abs(GetFaceIndex(Point3f(P)));
  Normal3f N(0, 0, 0);
  N[idx] = glm::sign(P[idx]);
  return N;
}

__host__ __device__ Normal3f GetCubeTangent(const Point3f& P)
{
  int idx = glm::abs(GetFaceIndex(Point3f(P)));

  Normal3f T;
  float direction = glm::sign(P[idx]);

  switch (idx)
  {
    // Z Faces
  case 2:
    T = Normal3f(1, 0, 0) * direction;
    break;

    // X Faces
  case 0:
    T = Normal3f(0, 0, 1) * -direction;
    break;

    // Y Faces
  default:
    T = Normal3f(1, 0, 0) * direction;
    break;
  }

  return T;
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
                                              glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec3& bitangent,
                                              glm::vec3& tangent, bool& outside)
{
  // Ray q;
  // q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
  // q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));
  //
  // float tmin = -1e38f;
  // float tmax = 1e38f;
  // glm::vec3 tmin_n;
  // glm::vec3 tmax_n;
  // for (int xyz = 0; xyz < 3; ++xyz) {
  //     float qdxyz = q.direction[xyz];
  //     /*if (glm::abs(qdxyz) > 0.00001f)*/ {
  //         float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
  //         float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
  //         float ta = glm::min(t1, t2);
  //         float tb = glm::max(t1, t2);
  //         glm::vec3 n;
  //         n[xyz] = t2 < t1 ? +1 : -1;
  //         if (ta > 0 && ta > tmin) {
  //             tmin = ta;
  //             tmin_n = n;
  //         }
  //         if (tb < tmax) {
  //             tmax = tb;
  //             tmax_n = n;
  //         }
  //     }
  // }
  //
  // if (tmax >= tmin && tmax > 0) {
  //     outside = true;
  //     if (tmin <= 0) {
  //         tmin = tmax;
  //         tmin_n = tmax_n;
  //         outside = false;
  //     }
  //     intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
  //     normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
  //     return glm::length(r.origin - intersectionPoint);
  // }
  // return -1;

  glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));
  
  Ray r_loc;
  r_loc.origin = ro;
  r_loc.direction = rd;
  
  float t_n = -1000000;
  float t_f = 1000000;
  for (int i = 0; i < 3; i++)
  {
    //Ray parallel to slab check
    if (r_loc.direction[i] == 0)
    {
      if (r_loc.origin[i] < -0.5f || r_loc.origin[i] > 0.5f)
      {
        return -1.0f;
      }
    }
    //If not parallel, do slab intersect check
    float t0 = (-0.5f - r_loc.origin[i]) / r_loc.direction[i];
    float t1 = (0.5f - r_loc.origin[i]) / r_loc.direction[i];
    if (t0 > t1)
    {
      float temp = t1;
      t1 = t0;
      t0 = temp;
    }
    if (t0 > t_n)
    {
      t_n = t0;
    }
    if (t1 < t_f)
    {
      t_f = t1;
    }
  }
  if (t_n < t_f)
  {
    float t = t_n > 0 ? t_n : t_f;
    if (t < 0)
    {
      return -1.0f;
    }
  
    //Lastly, transform the point found in object space by T
    glm::vec3 P = glm::vec3(r_loc.origin + t * r_loc.direction);
    intersectionPoint = multiplyMV(box.transform, glm::vec4(P, 1.f));
  
    Vector3f norm = glm::normalize(GetCubeNormal(P));
    Vector3f tang = glm::normalize(GetCubeTangent(P));
    Vector3f bitan = glm::normalize(glm::cross(norm, tang));
  
    normal = glm::normalize(glm::vec3(box.invTranspose * glm::vec4(norm, 0)));
    tangent = glm::normalize(glm::mat3(box.transform) * tang);
    bitangent = glm::normalize(glm::mat3(box.transform) * bitan);
  
    return glm::length(r.origin - intersectionPoint);
  }
  else
  {
    //If t_near was greater than t_far, we did not hit the cube
    return -1.0f;
  }
}


__host__ __device__ inline float boundingBoxIntersectionTest(glm::vec3 min, glm::vec3 max, Ray r_loc)
{
  float t_n = -1000000;
  float t_f = 1000000;
  for (int i = 0; i < 3; i++)
  {
    //Ray parallel to slab check
    if (r_loc.direction[i] == 0)
    {
      if (r_loc.origin[i] < min[i] || r_loc.origin[i] > max[i])
      {
        return -1.0f;
      }
    }
    //If not parallel, do slab intersect check
    float t0 = (min[i] - r_loc.origin[i]) / r_loc.direction[i];
    float t1 = (max[i] - r_loc.origin[i]) / r_loc.direction[i];
    if (t0 > t1)
    {
      const float temp = t1;
      t1 = t0;
      t0 = temp;
    }
    if (t0 > t_n)
    {
      t_n = t0;
    }
    if (t1 < t_f)
    {
      t_f = t1;
    }
  }
  if (t_n < t_f)
  {
    float t = t_n > 0 ? t_n : t_f;
    if (t < 0)
    {
      return -1.0f;
    }

    //Lastly, transform the point found in object space by T
    glm::vec3 P = glm::vec3(r_loc.origin + t * r_loc.direction);
    return glm::length(r_loc.origin - P);
  }
  else
  {
    //If t_near was greater than t_far, we did not hit the cube
    return -1.0f;
  }
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
                                                 glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec3& bitangent,
                                                 glm::vec3& tangent, bool& outside)
{
  //Transform the ray
  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));
  
  float A = pow(rd.x, 2.f) + pow(rd.y, 2.f) + pow(rd.z, 2.f);
  float B = 2 * (rd.x * ro.x + rd.y * ro.y + rd.z * ro.z);
  float C = pow(ro.x, 2.f) + pow(ro.y, 2.f) + pow(ro.z, 2.f) - 1.f; //Radius is 1.f
  float discriminant = B * B - 4 * A * C;
  //If the discriminant is negative, then there is no real root
  if (discriminant < 0)
  {
    return -1.0f;
  }
  
  float t = (-B - sqrt(discriminant)) / (2 * A);
  
  if (t < 0)
  {
    t = (-B + sqrt(discriminant)) / (2 * A);
  }
  
  if (t >= 0)
  {
    Ray rt;
    rt.origin = ro;
    rt.direction = rd;
    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);
    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
  
    tangent = glm::normalize(glm::mat3(sphere.transform) * glm::cross(Vector3f(0, 1, 0), (glm::normalize(objspaceIntersection))));
    bitangent = glm::normalize(glm::cross(normal, tangent));
  
    return glm::length(r.origin - intersectionPoint);
  }
  
  return -1.0f;

  // float radius = .5;
  //
  // glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
  // glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));
  //
  // Ray rt;
  // rt.origin = ro;
  // rt.direction = rd;
  //
  // float vDotDirection = glm::dot(rt.origin, rt.direction);
  // float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
  // if (radicand < 0) {
  //     return -1;
  // }
  //
  // float squareRoot = sqrt(radicand);
  // float firstTerm = -vDotDirection;
  // float t1 = firstTerm + squareRoot;
  // float t2 = firstTerm - squareRoot;
  //
  // float t = 0;
  // if (t1 < 0 && t2 < 0) {
  //     return -1;
  // } else if (t1 > 0 && t2 > 0) {
  //     t = min(t1, t2);
  //     outside = true;
  // } else {
  //     t = max(t1, t2);
  //     outside = false;
  // }
  //
  // glm::vec3 objspaceIntersection = getPointOnRay(rt, t);
  //
  // intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
  // normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
  //
  // tangent = glm::normalize(glm::mat3(sphere.transform) * glm::cross(Vector3f(0,1,0), (glm::normalize(objspaceIntersection))));
  // bitangent = glm::normalize(glm::cross(normal, tangent));
  //
  // // if (!outside) {
  // //     normal = -normal;
  // // }
  //
  // return glm::length(r.origin - intersectionPoint);
}


namespace Shapes
{
  namespace Triangles
  {
    __host__ __device__ inline float TriArea(const glm::vec3 &p1, const glm::vec3 &p2, const glm::vec3 &p3)
    {
      return glm::length(glm::cross(p1 - p2, p3 - p2)) * 0.5f;
    }

    __host__ __device__ inline float IntersectSingle(const Geom& shape, const Triangle& tri, const Ray& r, glm::vec3& localIntersection, glm::vec3& normal, glm::vec2& uv)
    {
      // const bool result = glm::intersectRayTriangle(r.origin, r.direction, tri.p1, tri.p2, tri.p3, localIntersection);
      //
      // if (result)
      // {
      //   return glm::length(r.origin - localIntersection);
      // }
      //
      // return -1.0f;

      float t =  glm::dot(tri.planeNormal, (tri.p1 - r.origin)) / glm::dot(tri.planeNormal, r.direction);
      if (t < 0) { return -1.0f; }

      if (glm::dot(tri.planeNormal, r.direction) > 0.0f)
        return -1.0f;
      
      glm::vec3 P = r.origin + t * r.direction;
      // Barycentric test
      float A = 0.5f * glm::length(glm::cross(tri.p1 - tri.p2, tri.p1 - tri.p3));
      float A0 = 0.5f * glm::length(glm::cross(P - tri.p2, P - tri.p3));
      float A1 = 0.5f * glm::length(glm::cross(P - tri.p3, P - tri.p1));
      float A2 = 0.5f * glm::length(glm::cross(P - tri.p1, P - tri.p2));
      float sum = A0 + A1 + A2;
      
      if(sum <= A * (1.0f + EPSILON)) {
        localIntersection = P;

        normal = glm::normalize(tri.n1 * A0 / A + tri.n2 * A1 / A + tri.n3 * A2 / A);
        uv = (tri.uv1 * A0 / A) + (tri.uv2 * A1 / A) + (tri.uv3 * A2 / A);

        return t;
      }
      
      return -1.0f;
    }

    __host__ __device__ inline Normal3f GetNormal(const Triangle& tri, const Point3f &P)
    {
      const float A = TriArea(tri.p1, tri.p2, tri.p3);
      const float A0 = TriArea(tri.p2, tri.p3, P);
      const float A1 = TriArea(tri.p1, tri.p3, P);
      const float A2 = TriArea(tri.p1, tri.p2, P);
      return glm::normalize(tri.n1 * A0/A + tri.n2 * A1/A + tri.n3 * A2/A);
    }

    __host__ __device__ inline Vector2f GetUV(const Triangle& tri, const Point3f &P)
    {
      const float A = TriArea(tri.p1, tri.p2, tri.p3);
      const float A0 = TriArea(tri.p2, tri.p3, P);
      const float A1 = TriArea(tri.p1, tri.p3, P);
      const float A2 = TriArea(tri.p1, tri.p2, P);
      return (tri.uv1 * A0/A) + (tri.uv2 * A1/A) + (tri.uv3 * A2/A);
    }

    __host__ __device__ inline float BulkIntersect(const Geom& shape, Triangle* triangles, const Ray& originalRay, glm::vec3& intersectionPoint,
      glm::vec3& normal, glm::vec3& bitangent, glm::vec3& tangent, glm::vec2& uv)
    {
      Ray r;

      //Transform the ray - Overall transform on GEOM
      r.origin = multiplyMV(shape.inverseTransform, glm::vec4(originalRay.origin, 1.0f));
      r.direction = glm::normalize(multiplyMV(shape.inverseTransform, glm::vec4(originalRay.direction, 0.0f)));

      bool found = false;
      float minDist = FLT_MAX;
      int closestTriIndex = -1;
      glm::vec3 P;

      glm::vec3 tmp_normal;
      glm::vec2 tmp_uv;

      glm::vec3 sel_point;
      glm::vec3 sel_normal;
      glm::vec2 sel_uv;

      for (int idx = 0; idx < shape.numTriangles; ++idx)
      {
        const Triangle& tri = triangles[shape.meshStartIndex + idx];
        const float dist = IntersectSingle(shape, tri, r, P, tmp_normal, tmp_uv);

        //Check that P is within the bounds of the square
        if (dist > EPSILON && dist < minDist)
        {
          found = true;
          minDist = dist;
          sel_point = P;
          sel_normal = tmp_normal;
          sel_uv = tmp_uv;
          closestTriIndex = shape.meshStartIndex + idx;
        }
      }

      if (found)
      {
        const Triangle& chosenOne = triangles[closestTriIndex];
        intersectionPoint = multiplyMV(shape.transform, glm::vec4(sel_point, 1.f));

        Point2f deltaUV1 = chosenOne.uv2 - chosenOne.uv1;
        Point2f deltaUV2 = chosenOne.uv3 - chosenOne.uv1;
        
        Vector3f deltaPos1 = chosenOne.p2 - chosenOne.p1;
        Vector3f deltaPos2 = chosenOne.p3 - chosenOne.p1;

        float result = deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x;

        glm::vec3 tan;
        glm::vec3 bit;

        if (result != 0.0f)
        {
          float r = 1.0f / (result);
          tan = glm::normalize(glm::vec3((deltaUV2.y * deltaPos1.x - deltaUV1.y * deltaPos2.x) * r, (deltaUV2.y * deltaPos1.y - deltaUV1.y * deltaPos2.y) * r, (deltaUV2.y * deltaPos1.z - deltaUV1.y * deltaPos2.z) * r));
          bit = glm::normalize(glm::cross(sel_normal, tan));
          tan = glm::normalize(glm::cross(bit, sel_normal));
        } else {
          tan = glm::vec3(1.0f, 0.0f, 0.0f);
          bit = glm::normalize(glm::cross(sel_normal, tan));
          tan = glm::normalize(glm::cross(bit, sel_normal));
        }

        normal = glm::normalize(glm::mat3(shape.invTranspose) * sel_normal);
        tangent = glm::normalize(glm::mat3(shape.transform) * tan);
        bitangent = glm::normalize(glm::mat3(shape.transform) * bit);

        uv = sel_uv;

        return glm::length(originalRay.origin - intersectionPoint);
      }

      return -1.0f;
      
    }

    __host__ __device__ inline float BufferIntersect(const Geom& shape, Triangle* triangles, int startIdx, int endIdx, const Ray& originalRay, glm::vec3& intersectionPoint,
      glm::vec3& normal, glm::vec3& bitangent, glm::vec3& tangent, glm::vec2& uv)
    {
      Ray r;

      //Transform the ray - Overall transform on GEOM
      r.origin = multiplyMV(shape.inverseTransform, glm::vec4(originalRay.origin, 1.0f));
      r.direction = glm::normalize(multiplyMV(shape.inverseTransform, glm::vec4(originalRay.direction, 0.0f)));


      bool found = false;
      float minDist = FLT_MAX;
      int closestTriIndex = -1;
      glm::vec3 P;

      glm::vec3 tmp_normal;
      glm::vec2 tmp_uv;

      glm::vec3 sel_point;
      glm::vec3 sel_normal;
      glm::vec2 sel_uv;

      for (int idx = startIdx; idx <= endIdx; ++idx)
      {
        const Triangle& tri = triangles[idx];
        const float dist = IntersectSingle(shape, tri, r, P, tmp_normal, tmp_uv);

        //Check that P is within the bounds of the square
        if (dist > EPSILON && dist < minDist)
        {
          found = true;
          minDist = dist;
          sel_point = P;
          sel_normal = tmp_normal;
          sel_uv = tmp_uv;
          closestTriIndex = idx;
        }
      }

      if (found)
      {
        const Triangle& chosenOne = triangles[closestTriIndex];
        intersectionPoint = multiplyMV(shape.transform, glm::vec4(sel_point, 1.f));

        Point2f deltaUV1 = chosenOne.uv2 - chosenOne.uv1;
        Point2f deltaUV2 = chosenOne.uv3 - chosenOne.uv1;

        Vector3f deltaPos1 = chosenOne.p2 - chosenOne.p1;
        Vector3f deltaPos2 = chosenOne.p3 - chosenOne.p1;

        float result = deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x;

        glm::vec3 tan;
        glm::vec3 bit;

        if (result != 0.0f)
        {
          float r = 1.0f / (result);
          tan = glm::normalize(glm::vec3((deltaUV2.y * deltaPos1.x - deltaUV1.y * deltaPos2.x) * r, (deltaUV2.y * deltaPos1.y - deltaUV1.y * deltaPos2.y) * r, (deltaUV2.y * deltaPos1.z - deltaUV1.y * deltaPos2.z) * r));
          bit = glm::normalize(glm::cross(sel_normal, tan));
          tan = glm::normalize(glm::cross(bit, sel_normal));
        } else {
          tan = glm::vec3(1.0f, 0.0f, 0.0f);
          bit = glm::normalize(glm::cross(sel_normal, tan));
          tan = glm::normalize(glm::cross(bit, sel_normal));
        }

        normal = glm::normalize(glm::mat3(shape.invTranspose) * sel_normal);
        tangent = glm::normalize(glm::mat3(shape.transform) * tan);
        bitangent = glm::normalize(glm::mat3(shape.transform) * bit);

        uv = sel_uv;

        return glm::length(originalRay.origin - intersectionPoint);
      }

      return -1.0f;

    }
  }

  namespace SquarePlane
  {
    __host__ __device__ inline float Intersect(const Geom& shape, const Ray& r, glm::vec3& intersectionPoint,
                                               glm::vec3& normal, glm::vec3& bitangent, glm::vec3& tangent, glm::vec2& uv)
    {
      //Transform the ray
      const glm::vec3 ro = multiplyMV(shape.inverseTransform, glm::vec4(r.origin, 1.0f));
      const glm::vec3 rd = glm::normalize(multiplyMV(shape.inverseTransform, glm::vec4(r.direction, 0.0f)));

      //Ray-plane intersection
      const float t = glm::dot(glm::vec3(0, 0, 1), (glm::vec3(0.5f, 0.5f, 0) - ro)) / glm::dot(glm::vec3(0, 0, 1), rd);
      const Point3f P = Point3f(t * rd + ro);

      //Check that P is within the bounds of the square
      if (t > EPSILON && P.x >= -0.5f && P.x <= 0.5f && P.y >= -0.5f && P.y <= 0.5f)
      {
        intersectionPoint = multiplyMV(shape.transform, glm::vec4(P, 1.f));

        normal = glm::normalize(glm::mat3(shape.invTranspose) * Normal3f(0, 0, 1));
        tangent = glm::normalize(glm::mat3(shape.transform) * Normal3f(1, 0, 0));
        bitangent = glm::normalize(glm::mat3(shape.transform) * Normal3f(0, 1, 0));

        uv = Point2f(P.x + 0.5f, P.y + 0.5f);

        return glm::length(r.origin - intersectionPoint);
      }
      return -1.0f;
    }

    __host__ __device__ inline float Area(Geom* geometry)
    {
      return geometry->scale[0] * geometry->scale[1];
    }

    __host__ __device__ inline void Sample(Geom* geometry, const float rngX, const float rngY, Float* pdf,
                                           Intersection* intr)
    {
      *pdf = 1.0f / (geometry->scale[0] * geometry->scale[1]);

      const Point3f localPoint = Point3f(Point2f(rngX, rngY) - Point2f(0.5f, 0.5f), 0.0f);
      const Point3f worldPoint = Point3f(geometry->transform * glm::vec4(localPoint, 1.0f));

      intr->point = worldPoint;
      intr->normal = glm::normalize(glm::mat3(geometry->inverseTransform) * Normal3f(0, 0, 1));
    }
  }

  __host__ __device__ inline void Sample(Geom* geometry, const float rngX, const float rngY, float* pdf,
                                         Intersection* intr)
  {
    if (geometry->type == SQUAREPLANE)
    {
      SquarePlane::Sample(geometry, rngX, rngY, pdf, intr);
    }
  }

  __host__ __device__ inline Intersection Sample(Geom* geometry, const Point3f& refPoint, const float rngX,
                                                 const float rngY, float* pdf)
  {
    Intersection isectLight;
    Sample(geometry, rngX, rngY, pdf, &isectLight);

    const Vector3f wi = glm::normalize(isectLight.point - refPoint);

    const Vector3f toRef = refPoint - isectLight.point;
    const float distSq = glm::length2(toRef);

    const float dot = glm::abs(glm::dot(isectLight.normal, -wi));

    if (dot <= 0.00001)
    {
      *pdf = 0;
      return isectLight;
    }

    *pdf = (distSq) / (dot / (*pdf));

    return isectLight;
  }
}

 __device__ void kdIntersectionTest(const int nodeIdx, const Ray& targetRay, const Ray& transformedRay, const Geom& geom,  KDNode* kdNodes, Triangle* kdTriangles, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec3& bitangent,
  glm::vec3& tangent, glm::vec2& uv, float& minT)
{
  const KDNode currentNode = kdNodes[nodeIdx];
  const float boundT = boundingBoxIntersectionTest(currentNode.min, currentNode.max, transformedRay);

  if (boundT < EPSILON)
  {
    // Didn't intersect node's AABB
    return;
  }

  // Child Node
  if(currentNode.triStartIdx >= 0)
  {
    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    glm::vec2 tmp_uv;
    glm::vec3 tmp_bitangent;
    glm::vec3 tmp_tangent;

    const float t = Shapes::Triangles::BufferIntersect(geom, kdTriangles, currentNode.triStartIdx, currentNode.triEndIdx, targetRay, tmp_intersect, tmp_normal, tmp_bitangent, tmp_tangent, tmp_uv);
    if (t > EPSILON && t < minT)
    {
      minT = t;
      intersectionPoint = tmp_intersect;
      normal = tmp_normal;
      bitangent = tmp_bitangent;
      tangent = tmp_tangent;
      uv = tmp_uv;
    }

    return;
  }

  if (currentNode.leftChildIdx >= 0)
  {
    kdIntersectionTest(currentNode.leftChildIdx, targetRay, transformedRay, geom, kdNodes, kdTriangles, intersectionPoint, normal, bitangent, tangent, uv, minT);
  }

  if (currentNode.rightChildIdx >= 0)
  {
    kdIntersectionTest(currentNode.rightChildIdx, targetRay, transformedRay, geom, kdNodes, kdTriangles, intersectionPoint, normal, bitangent, tangent, uv, minT);
  }
}

 __device__ inline void kdIntersectionTest_Queue(const int startIdx, const Ray& targetRay, const Ray& transformedRay, const Geom& geom,  KDNode* kdNodes, Triangle* kdTriangles, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec3& bitangent,
  glm::vec3& tangent, glm::vec2& uv, float& minT)
{
  KDNode* queueContainer[KD_TREE_QUEUE_SIZE];

  // Starting
  queueContainer[0] = &kdNodes[startIdx];
  int writeIndex = 0;
  int readIndex = 0;

  while(readIndex <= writeIndex)
  {
    const KDNode* currentNode = queueContainer[readIndex];
    // Read Current Write Index Node
    const float boundT = boundingBoxIntersectionTest(currentNode->min, currentNode->max, transformedRay);

    // Didn't intersect node's AABB
    if (boundT < EPSILON)
    {
      readIndex++;
      continue;
    }

    // Child Node
    if(currentNode->triStartIdx >= 0)
    {
      glm::vec3 tmp_intersect;
      glm::vec3 tmp_normal;
      glm::vec2 tmp_uv;
      glm::vec3 tmp_bitangent;
      glm::vec3 tmp_tangent;

      const float t = Shapes::Triangles::BufferIntersect(geom, kdTriangles, currentNode->triStartIdx, currentNode->triEndIdx, targetRay, tmp_intersect, tmp_normal, tmp_bitangent, tmp_tangent, tmp_uv);
      if (t > EPSILON && t < minT)
      {
        minT = t;
        intersectionPoint = tmp_intersect;
        normal = tmp_normal;
        bitangent = tmp_bitangent;
        tangent = tmp_tangent;
        uv = tmp_uv;
      }

      readIndex++;
      continue;
    }

    if (currentNode->leftChildIdx >= 0)
    {
      writeIndex++;
      queueContainer[writeIndex] = &kdNodes[currentNode->leftChildIdx];
    }

    if (currentNode->rightChildIdx >= 0)
    {
      writeIndex++;
      queueContainer[writeIndex] = &kdNodes[currentNode->rightChildIdx];
    }

    readIndex++;
  }
}

namespace Intersections
{
  __device__ inline ShadeableIntersection Do(Ray targetRay, Geom* geometries, int geometrySize, Triangle* triangles, KDNode* kdNodes, Triangle* kdTriangles)
  {
    float t = -1.0f;
    glm::vec3 intersectPoint;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec3 tangent;
    glm::vec3 bitangent;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec2 tmp_uv;
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
        t = boxIntersectionTest(geom, targetRay, tmp_intersect, tmp_normal, tmp_bitangent, tmp_tangent, outside);
      }
      else if (geom.type == SPHERE)
      {
        t = sphereIntersectionTest(geom, targetRay, tmp_intersect, tmp_normal, tmp_bitangent, tmp_tangent, outside);
      }
      else if (geom.type == SQUAREPLANE)
      {
        t = Shapes::SquarePlane::Intersect(geom, targetRay, tmp_intersect, tmp_normal, tmp_bitangent, tmp_tangent, tmp_uv);
      }
      else if (geom.type == MESH)
      {

#ifdef ENABLE_MESH_BOUNDING_CULL
        Ray r_loc;
        r_loc.origin = multiplyMV(geom.inverseTransform, glm::vec4(targetRay.origin, 1.0f));
        r_loc.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(targetRay.direction, 0.0f)));

        const float temp = boundingBoxIntersectionTest(geom.min, geom.max, r_loc);

        if (temp > EPSILON) {
#endif
          t = Shapes::Triangles::BulkIntersect(geom, triangles, targetRay, tmp_intersect, tmp_normal, tmp_bitangent, tmp_tangent, tmp_uv);

#ifdef ENABLE_MESH_BOUNDING_CULL
        }
#endif
      }
       else if (geom.type == ACCELERATED_MESH)
      {
        const int startIdx = geom.kdRootNodeIndex;
      
        Ray r_loc;
        r_loc.origin = multiplyMV(geom.inverseTransform, glm::vec4(targetRay.origin, 1.0f));
        r_loc.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(targetRay.direction, 0.0f)));
      
        float meshT = FLT_MAX;
      
        kdIntersectionTest_Queue(0, targetRay, r_loc, geom, kdNodes, kdTriangles, tmp_intersect, tmp_normal, tmp_bitangent, tmp_tangent, tmp_uv, meshT);
        t = meshT;
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
        uv = tmp_uv;
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
      result.uv = uv;
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

  __host__ __device__ inline Ray SpawnRay(const Point3f& origin, const Normal3f& normal, const Vector3f& d)
  {
    Vector3f originOffset = normal * 0.0005f;
    // Make sure to flip the direction of the offset so it's in
    // the same general direction as the ray direction
    originOffset = (glm::dot(d, normal) > 0) ? originOffset : -originOffset;
    const Point3f o(origin + originOffset);
    return Ray{o, d};
  }
}
