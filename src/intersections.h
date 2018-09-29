#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define FACETED
//#define SMOOTH

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

// adapted from scratchapixel
__host__ __device__ bool boxIntersectionTest(Ray r, const glm::vec3 &box_min, const glm::vec3 &box_max) {

	float tmin, tmax, tymin, tymax, tzmin, tzmax;

	glm::vec3 invdir = glm::vec3(1.f / r.direction.x, 1.f / r.direction.y, 1.f / r.direction.z);
	int sign[3];
	sign[0] = (invdir.x < 0);
	sign[1] = (invdir.y < 0);
	sign[2] = (invdir.z < 0);

	glm::vec3 bounds[2];
	bounds[0] = box_min;
	bounds[1] = box_max;

	tmin = (bounds[sign[0]].x - r.origin.x) * invdir.x;
	tmax = (bounds[1 - sign[0]].x - r.origin.x) * invdir.x;
	tymin = (bounds[sign[1]].y - r.origin.y) * invdir.y;
	tymax = (bounds[1 - sign[1]].y - r.origin.y) * invdir.y;

	if ((tmin > tymax) || (tymin > tmax))
		return false;
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	tzmin = (bounds[sign[2]].z - r.origin.z) * invdir.z;
	tzmax = (bounds[1 - sign[2]].z - r.origin.z) * invdir.z;

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	return true;
}

__host__ __device__ bool rayTriangleIntersect(const glm::vec3 &origin, const glm::vec3 &direction,
	const glm::vec3& pos_0, const glm::vec3& pos_1, const glm::vec3& pos_2,
	glm::vec3& intersect, glm::vec3& bary) {

	glm::vec3 vec1 = pos_1 - pos_0;
	glm::vec3 vec2 = pos_2 - pos_1;
	glm::vec3 vec3 = pos_0 - pos_2;
	glm::vec3 normal = glm::cross(vec1, vec2);
	normal = glm::normalize(normal);

	float t = dot((pos_0 - origin), normal) / dot(direction, normal);
	intersect = origin + t * direction;

	glm::vec3 b1 = intersect - pos_0;
	glm::vec3 b2 = intersect - pos_1;
	glm::vec3 b3 = intersect - pos_2;

	float s1 = 0.5 * glm::length(glm::cross(vec1, b1));
	float s2 = 0.5 * glm::length(glm::cross(vec2, b2));
	float s3 = 0.5 * glm::length(glm::cross(vec3, b3));
	float s = s1 + s2 + s3;
	float s_real = 0.5 * glm::length(glm::cross(vec1, vec2));
	float bary1 = s1 / s;
	float bary2 = s2 / s;
	float bary3 = s3 / s;

	if (s < s_real + 0.001 && s > s_real - 0.001) {
		bary = glm::vec3(bary1, bary2, bary3);
		return true;
	}
	else {
		return false;
	}

}

__host__ __device__ float triangleMeshIntersectionTest(Geom triMesh, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside,
	Triangle *dev_tris, int numtris, glm::vec3 &box_min, glm::vec3 &box_max) {
	Ray q;
	q.origin	=				 multiplyMV(triMesh.inverseTransform, glm::vec4(r.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(triMesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

	// initial bounding box test
	bool initial_check = boxIntersectionTest(q, box_min, box_max);
	if (!initial_check) {
		return -1;
	}

	glm::vec3 pos_0; glm::vec3 pos_1; glm::vec3 pos_2;
	glm::vec3 norm_0; glm::vec3 norm_1; glm::vec3 norm_2;
	glm::vec3 norm_interp;
	glm::vec3 obj_intersect;
	glm::mat3 mp; glm::mat3 mn;

	glm::vec3 bary = glm::vec3(0.0f);
	glm::vec3 pos_intersect = glm::vec3(1E5);

	bool intersection = false;

	// naive intersection test against all triangles in the scene
	for (int i = 0; i < numtris; ++i) {
		pos_0 = dev_tris[i].v0;
		pos_1 = dev_tris[i].v1;
		pos_2 = dev_tris[i].v2;

		bool intersect = rayTriangleIntersect(q.origin, q.direction, pos_0, pos_1, pos_2, obj_intersect, bary);

		if (intersect) {
			intersection = true;
			if (glm::length(q.origin - obj_intersect) < glm::length(q.origin - pos_intersect)) {
#ifdef FACETED
				pos_intersect = obj_intersect;
				glm::vec3 v1 = pos_1 - pos_0;
				glm::vec3 v2 = pos_2 - pos_0;
				glm::vec3 norm = glm::normalize(glm::cross(v1, v2));
				norm_interp = norm;
#endif
#ifdef SMOOTH
				norm_0 = dev_tris[i].n0;
				norm_1 = dev_tris[i].n1;
				norm_2 = dev_tris[i].n2;
				mn = glm::mat3(norm_0, norm_1, norm_2);
				norm_interp = mn * bary; // interpolated face norm
#endif
			}
		}
	}
	if (intersection) {
		outside = true;
		normal = glm::normalize(multiplyMV(triMesh.transform, glm::vec4(norm_interp, 0.0f)));
		intersectionPoint = multiplyMV(triMesh.transform, glm::vec4(pos_intersect, 1.0f));
		return glm::length(r.origin - intersectionPoint) - 0.01;
	}

	return -1;
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

    return glm::length(r.origin - intersectionPoint);
}
