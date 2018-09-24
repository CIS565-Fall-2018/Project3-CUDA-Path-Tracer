#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"
#include "common.h"
#include "shapeFunctions.h"

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
        glm::vec3 &intersectionPoint, glm::vec3 &normal, glm::vec3 &tangent, glm::vec3 &bitangent, bool &outside) {
   
    Ray q;
    q.origin    = multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f));

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
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));

    /*
    Ray r_loc = transformRay(r, box.inverseTransform);

    float t_n = -1000000;
    float t_f = 1000000;
    for (int i = 0; i < 3; i++) {
        //Ray parallel to slab check
        if (r_loc.direction[i] == 0) {
            if (r_loc.origin[i] < -0.5f || r_loc.origin[i] > 0.5f) {
                return false;
                }
            }
        //If not parallel, do slab intersect check
        float t0 = (-0.5f - r_loc.origin[i]) / r_loc.direction[i];
        float t1 = (0.5f - r_loc.origin[i]) / r_loc.direction[i];
        if (t0 > t1) {
            float temp = t1;
            t1 = t0;
            t0 = temp;
            }
        if (t0 > t_n) {
            t_n = t0;
            }
        if (t1 < t_f) {
            t_f = t1;
            }
        }
    if (t_n < t_f)
        {
        float t = t_n > 0 ? t_n : t_f;
        if (t < 0) {
            return false;
            }
        //Lastly, transform the point found in object space by T
        intersectionPoint = getPointOnRay(r_loc, t); // r_loc.origin + t*r_loc.direction;
        //InitializeIntersection(isect, t, Point3f(P));
        return true;
        }
    else {//If t_near was greater than t_far, we did not hit the cube
        return false;
        }

        // Compute tangent and bitangent
        glm::vec3 localBit = glm::vec3(0.0f);
        glm::vec3 localTan = glm::vec3(0.0f);

        glm::vec3 tmin_n = glm::vec3();

        if (tmin_n.x < 0) {
            localBit.y = 1;
            localTan.z = 1;
        }
        else if (tmin_n.x > 0) {
            localBit.y = 1;
            localTan.z = -1;
        }
        else if (tmin_n.y < 0) {
            localBit.z = 1;
            localTan.x = 1;
            }
        else if (tmin_n.y > 0) {
            localBit.z = -1;
            localTan.x = 1;
            }
        else if (tmin_n.z < 0) {
            localBit.y = 1;
            localTan.x = -1;
            }
        else if (tmin_n.z > 0) {
            localBit.y = 1;
            localTan.x = 1;
            }
        tangent = glm::normalize(multiplyMV(box.transform, glm::vec4(localTan, 0.0f)));
        bitangent = glm::normalize(multiplyMV(box.transform, glm::vec4(localBit, 0.0f)));
        */

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
        glm::vec3 &intersectionPoint, glm::vec3 &normal, glm::vec3 &tangent, glm::vec3 &bitangent, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f));

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
    if (!outside) {
        normal = -normal;
    }

    // Compute tangent and bitangent
    tangent = glm::normalize(multiplyMV(sphere.transform, 
                                        glm::vec4(glm::cross(glm::vec3(0, 1, 0), (glm::normalize(objspaceIntersection))), 0)));
    bitangent = glm::normalize(glm::cross(normal, tangent));

    //return glm::length(r.origin - intersectionPoint);
    return t;
}
