#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <queue>

#include "sceneStructs.h"
#include "utilities.h"

#include <thrust/device_vector.h>

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

__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
	return glm::vec3(m * v);
}

__host__ __device__ bool aabbBoxIntersectKD(const Ray& r, glm::vec3 min, glm::vec3 max,float& near)
{
	float tnear = FLT_MIN;
	float tfar = FLT_MAX;

	for (int i = 0; i<3; i++)
	{
		float t0, t1;

		if (fabs(r.direction[i]) < EPSILON)
		{
			if (r.origin[i] < min[i] || r.origin[i] > max[i])
				return false;
			else
			{
				t0 = FLT_MIN;
				t1 = FLT_MAX;
			}
		}
		else
		{
			t0 = (min[i] - r.origin[i]) / r.direction[i];
			t1 = (max[i] - r.origin[i]) / r.direction[i];
		}

		tnear = glm::max(tnear, glm::min(t0, t1));
		tfar = glm::min(tfar, glm::max(t0, t1));
	}

	if (tfar < tnear) return false; // no intersection

	if (tfar < 0) return false; // behind origin of ray

	near = tnear;

	return true;

}

__host__ __device__ bool aabbBoxIntersectlocal(Geom meshgeom,const Ray& r, glm::vec3 min, glm::vec3 max)
{

	Ray q;
	q.origin = multiplyMV(meshgeom.inverseTransform, glm::vec4(r.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(meshgeom.inverseTransform, glm::vec4(r.direction, 0.0f)));

	float tnear = FLT_MIN;
	float tfar = FLT_MAX;

	for (int i = 0; i<3; i++)
	{
		float t0, t1;

		if (fabs(q.direction[i]) < EPSILON)
		{
			if (q.origin[i] < min[i] || q.origin[i] > max[i])
				return false;
			else
			{
				t0 = FLT_MIN;
				t1 = FLT_MAX;
			}
		}
		else
		{
			t0 = (min[i] - q.origin[i]) / q.direction[i];
			t1 = (max[i] - q.origin[i]) / q.direction[i];
		}

		tnear = glm::max(tnear, glm::min(t0, t1));
		tfar = glm::min(tfar, glm::max(t0, t1));
	}

	if (tfar < tnear) return false; // no intersection

	if (tfar < 0) return false; // behind origin of ray

	return true;

}

__host__ __device__ bool aabbBoxIntersect(const Ray& r, glm::vec3 min, glm::vec3 max)
{
	float tnear = FLT_MIN;
	float tfar = FLT_MAX;

	for (int i = 0; i<3; i++)
	{
		float t0, t1;

		if (fabs(r.direction[i]) < EPSILON)
		{
			if (r.origin[i] < min[i] || r.origin[i] > max[i])
				return false;
			else
			{
				t0 = FLT_MIN;
				t1 = FLT_MAX;
			}
		}
		else
		{
			t0 = (min[i] - r.origin[i]) / r.direction[i];
			t1 = (max[i] - r.origin[i]) / r.direction[i];
		}

		tnear = glm::max(tnear, glm::min(t0, t1));
		tfar = glm::min(tfar, glm::max(t0, t1));
	}

	if (tfar < tnear) return false; // no intersection

	if (tfar < 0) return false; // behind origin of ray

	return true;

}

__host__ __device__ bool BBIntersect(const Ray &r, glm::vec3 min, glm::vec3 max, float* t)
{


	float t_n = -FLT_MAX;
	float t_f = FLT_MAX;
	{
		if (r.direction[0] == 0)
		{
			if (r.origin[0] < min.x || r.origin[0] > max.x) {
				return false;
			}
		}
		float t0 = (min.x - r.origin[0]) / r.direction[0];
		float t1 = (max.x - r.origin[0]) / r.direction[0];
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
		if (r.direction[1] == 0) {
			if (r.origin[1] < min.y || r.origin[1] > max.y) {
				return false;
			}
		}
		t0 = (min.y - r.origin[1]) / r.direction[1];
		t1 = (max.y - r.origin[1]) / r.direction[1];
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
		if (r.direction[2] == 0) {
			if (r.origin[2] < min.z || r.origin[2] > max.z) {
				return false;
			}
		}
		t0 = (min.z - r.origin[2]) / r.direction[2];
		t1 = (max.z - r.origin[2]) / r.direction[2];
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
		if ((r.origin[0] >= min.x && r.origin[0] <= max.x) &&
			(r.origin[1] >= min.y && r.origin[1] <= max.y) &&
			(r.origin[2] >= min.z && r.origin[2] <= max.z))
		{
			*t = t_n;
		}
		else
		{
			float result_t = t_n > 0 ? t_n : t_f;
			if (result_t < 0)
				return false;


			*t = result_t;
		}


		return true;
	}
	else
	{
		return false;
	}
}
__host__ __device__ bool intersectAABBarrays(Ray r, glm::vec3 mins, glm::vec3 maxs, float& dist)
{

	bool result = false;
	glm::vec3 invdir(1.0f / r.direction.x,
		1.0f / r.direction.y,
		1.0f / r.direction.z);

	float v1 = (mins[0] - r.origin.x)*invdir.x;
	float v2 = (maxs[0] - r.origin.x)*invdir.x;
	float v3 = (mins[1] - r.origin.y)*invdir.y;
	float v4 = (maxs[1] - r.origin.y)*invdir.y;
	float v5 = (mins[2] - r.origin.z)*invdir.z;
	float v6 = (maxs[2] - r.origin.z)*invdir.z;

	float dmin = max(max(min(v1, v2), min(v3, v4)), min(v5, v6));
	float dmax = min(min(max(v1, v2), max(v3, v4)), max(v5, v6));

	if (dmax < 0)
	{
		dist = dmax;
		result = false;
		return result;
	}
	if (dmin > dmax)
	{
		dist = dmax;
		result = false;
		return result;
	}
	dist = dmin;
	result = true;
	return result;
}
__host__ __device__ bool KDhit(Geom meshgeom
	, GPUKDtreeNode* nodelst
	, Ray& r
	, int& startidx
	, int& endidx
	, int* gputrilst
	, int& size

)
{
	if (!nodelst) return false;
	Ray q;
	q.origin = multiplyMV(meshgeom.inverseTransform, glm::vec4(r.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(meshgeom.inverseTransform, glm::vec4(r.direction, 0.0f)));
	int curnodeidx = 0;
	int count = 0;
	GPUKDtreeNode* node = NULL;
	bool nodeIDs[100] = { false };
	while (count<6)
	{
		node = &nodelst[curnodeidx];
		float near1 = 0, near2 = 0;
		bool lefthit = false;
		bool righthit = false;
		if(node->leftidx!=-1)
		lefthit = aabbBoxIntersectKD(q, nodelst[node->leftidx].minB, nodelst[node->leftidx].maxB, near1);
		if (node->rightidx != -1)
		righthit = aabbBoxIntersectKD(q, nodelst[node->rightidx].minB, nodelst[node->rightidx].maxB, near2);
		if (lefthit&&righthit&&node->trsize>1 && (node->leftidx != -1 && node->rightidx != -1))
		{
			if (near1 < near2) curnodeidx = node->leftidx;
			else curnodeidx = node->rightidx;
		}
		else if (lefthit&& node->trsize>1 && (node->leftidx != -1))
		{
			curnodeidx = node->leftidx;
		}
		else if (righthit&&node->trsize>1 && (node->rightidx != -1))
		{
			curnodeidx = node->rightidx;
		}
		else
		{
			continue;
		}
		count++;
	}
	if (count == 0)
		return false;
	else
	{
		startidx = node->GPUtriangleidxinLst;
		endidx = startidx + node->trsize;
		return true;
	}
}





/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */


__host__ __device__ float triangleIntersectionTest(Geom meshgeom, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside ,Triangle tri)
{
	Ray q;
	q.origin = multiplyMV(meshgeom.inverseTransform, glm::vec4(r.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(meshgeom.inverseTransform, glm::vec4(r.direction, 0.0f)));

	glm::vec3 bary;

	float t = -1;
	if (glm::intersectRayTriangle(q.origin, q.direction, tri.Triverts[0].pos, tri.Triverts[1].pos, tri.Triverts[2].pos, bary)) {
		t = bary.z;
	}

	glm::vec3 objspaceIntersection = getPointOnRay(q, t);

	intersectionPoint = multiplyMV(meshgeom.transform, glm::vec4(objspaceIntersection, 1.f));

	normal = glm::normalize(glm::cross(tri.Triverts[0].pos - tri.Triverts[1].pos, tri.Triverts[0].pos - tri.Triverts[2].pos))/*tri.Trinormal*/;
	normal = glm::normalize(multiplyMV(meshgeom.transform, glm::vec4(normal, 0.0f)));
	outside = true;

	if (glm::dot(q.origin, normal) < 0) {
		outside = false;
	}

	return t;
	/*float tmin = -1e38f;
	float tmax = 1e38f;
	glm::vec3 tmin_n;
	glm::vec3 tmax_n;

	glm::vec3 res;
	bool is_intersect = false;
	
	if (glm::dot(tri.Trinormal, q.direction) < 0) outside = true;
	is_intersect= glm::intersectRayTriangle(q.origin, q.direction, tri.Triverts[0].pos, tri.Triverts[1].pos, tri.Triverts[2].pos, res);

	if (is_intersect)
	{
		
		intersectionPoint = q.origin + q.direction*res.z;
		normal = tri.Trinormal;
		intersectionPoint = multiplyMV(meshgeom.transform, glm::vec4(intersectionPoint, 1.0f));
		normal = glm::normalize(multiplyMV(meshgeom.transform, glm::vec4(normal, 0.0f)));
		return glm::length(r.origin - intersectionPoint);
	}

	return -1;*/
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
