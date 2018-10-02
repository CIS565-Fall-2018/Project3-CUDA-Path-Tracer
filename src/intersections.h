#pragma once

#include "sceneStructs.h"
#include "utilities.h"

#include <thrust/random.h>
#include <thrust/swap.h>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

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
       {
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
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
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
        t = thrust::min(t1, t2);
        outside = true;
    } else {
        t = thrust::max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

#define BOUND_INTERSECTION 1

#ifdef BOUND_INTERSECTION

__host__ __device__
bool bounding_volume_intersection_culling(const Ray &ray, Geom& geom)
{
    glm::vec3& minimum_vertex = geom.mesh_object.bounding_min;
    glm::vec3& maximum_vertex = geom.mesh_object.bounding_max;


    float minimum_t_from_ray_x = (minimum_vertex.x - ray.origin.x) / ray.direction.x;
    float maximum_t_from_ray_x = (maximum_vertex.x - ray.origin.x) / ray.direction.x;

    float minimum_t_from_ray_y = (minimum_vertex.y - ray.origin.y) / ray.direction.y;
    float maximum_t_from_ray_y = (maximum_vertex.y - ray.origin.y) / ray.direction.y;

    float minimum_t_from_ray_z = (minimum_vertex.z - ray.origin.z) / ray.direction.z; 
    float maximum_t_from_ray_z = (maximum_vertex.z - ray.origin.z) / ray.direction.z;

    //make sure they're in order (min, max)

    if (minimum_t_from_ray_x > maximum_t_from_ray_x)
    {
        thrust::swap(minimum_t_from_ray_x, maximum_t_from_ray_x);
    }

    if (minimum_t_from_ray_y > maximum_t_from_ray_y)
    {
        thrust::swap(minimum_t_from_ray_y, maximum_t_from_ray_y);         
    }
 
    if (minimum_t_from_ray_z > maximum_t_from_ray_z)
    {
        thrust::swap(minimum_t_from_ray_z, maximum_t_from_ray_z);
    }

    //check bounds
 
    if ((minimum_t_from_ray_x > maximum_t_from_ray_y) || (minimum_t_from_ray_y > maximum_t_from_ray_x)) 
        return false; 
 
    if ((minimum_t_from_ray_x > maximum_t_from_ray_z) || (minimum_t_from_ray_z > maximum_t_from_ray_x) || (minimum_t_from_ray_y > maximum_t_from_ray_z) || (minimum_t_from_ray_z > maximum_t_from_ray_y)) 
        return false; 
 
    return true; 
} 

#endif

__host__ __device__ float meshObjectIntersectionTest(Geom geom, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(geom.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    if(!bounding_volume_intersection_culling(q, geom))
    {
        return -1;
    }

    for(size_t i = 0; i < geom.mesh_object.num_triangles; i++)
    {
        Triangle& triangle = geom.mesh_object.dev_triangles[i];
        Vertex& v1 = triangle.vertices[0];
        Vertex& v2 = triangle.vertices[1];
        Vertex& v3 = triangle.vertices[2];

        glm::vec3 hit;
        if (glm::intersectRayTriangle(q.origin, q.direction, v1.position, v2.position, v3.position, hit))
        {
            //figuring out that hit.z is the distance (link below mentions that hit.z is the distance, not mentioned in documentation)
            //https://github.com/g-truc/glm/issues/6
            glm::vec3 intersection_triangle = getPointOnRay(q, hit.z);
            //glm::vec3 intersection_triangle = hit;
            intersectionPoint = multiplyMV(geom.transform, glm::vec4(intersection_triangle, 1.f));
            //normal = triangle.vertices->normal;
            normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(intersection_triangle, 0.f)));
            return hit.z;
        }
    }

    return -1;
}
