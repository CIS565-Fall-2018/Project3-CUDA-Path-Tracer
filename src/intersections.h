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


// Returns +/- [0, 2]
__host__ __device__ int GetFaceIndex(const glm::vec3& P)
{
	int idx = 0;
	float val = -1;
	for(int i = 0; i < 3; i++){
		if(glm::abs(P[i]) > val){
			idx = i * glm::sign(P[i]);
			val = glm::abs(P[i]);
		}
	}
	return idx;
}

__host__ __device__ glm::vec3 GetCubeNormal(const glm::vec3& P, const int faceIndex)
{
	glm::vec3 N(0,0,0);
	N[faceIndex] = glm::sign(P[faceIndex]);
	return N;
}

__host__ __device__ glm::vec3 GetCubeTangent(const glm::vec3& P, const int faceIndex)
{
	glm::vec3 N(0,0,0);
	if(faceIndex == 0 && glm::sign(P[faceIndex]) > 0) {
		N = glm::vec3(0.f, 0.f, -1.f);
	} else if(faceIndex == 1 && glm::sign(P[faceIndex]) > 0) {
		N = glm::vec3(0.f, 0.f, 1.f);
	} else if(faceIndex == 2 && glm::sign(P[faceIndex]) > 0) {
		N = glm::vec3(1.f, 0.f, 0.f);
	} else if(faceIndex == 0 && glm::sign(P[faceIndex]) < 0) {
		N = glm::vec3(0.f, 0.f, 1.f);
	} else if(faceIndex == 1 && glm::sign(P[faceIndex]) < 0) {
		N = glm::vec3(1.f, 0.f, 0.f);
	} else if(faceIndex == 2 && glm::sign(P[faceIndex]) < 0) {
		N = glm::vec3(-1.f, 0.f, 0.f);
	}
	return N;
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
__host__ __device__ float boxIntersectionTest(Geom box, Ray r, ShadeableIntersection* intersection, bool &outside) 
{
    glm::vec3 ray_origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    glm::vec3 ray_direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

	float t_n = -1000000;
	float t_f = 1000000;

	for(int i = 0; i < 3; i++)
	{
		//Ray parallel to slab check
		if(ray_direction[i] == 0)
		{
			if(ray_origin[i] < -0.5f || ray_origin[i] > 0.5f)
			{
				return -1;
			}
		}
		//If not parallel, do slab intersect check
		float t0 = (-0.5f - ray_origin[i])/ray_direction[i];
		float t1 = (0.5f - ray_origin[i])/ray_direction[i];
		if(t0 > t1)
		{
			float temp = t1;
			t1 = t0;
			t0 = temp;
		}
		if(t0 > t_n)
		{
			t_n = t0;
		}

		if(t1 < t_f)
		{
			t_f = t1;
		}

	}
	if(t_n < t_f)
	{
		const float t = t_n > 0 ? t_n : t_f;
		
		if (t < 0)
		{
			return -1;
		}
		//Lastly, transform the point found in object space by T
		const glm::vec3 objSpaceIntersection = ray_origin + t * ray_direction;
		const glm::vec3 intersectionPoint = multiplyMV(box.transform, glm::vec4(objSpaceIntersection, 1.0f));
		intersection->m_intersectionPointWorld = intersectionPoint;

		// Computing normal, tangent and bitangent
		const int faceIndex = glm::abs(GetFaceIndex(objSpaceIntersection));
		const glm::vec3 localNormal = GetCubeNormal(objSpaceIntersection, faceIndex);
		const glm::vec3 localTangent = GetCubeTangent(objSpaceIntersection, faceIndex);

		const glm::vec3 normal = glm::normalize(multiplyMV(box.transform, glm::vec4(localNormal, 0.0f)));
		const glm::vec3 tangent = glm::normalize(glm::vec3(box.transform * glm::vec4(localTangent, 0.0f)));
		const glm::vec3 bitangent = glm::normalize(glm::cross(normal, tangent));

		intersection->m_surfaceNormal = normal;
		intersection->m_surfaceTangent = tangent;
		intersection->m_surfaceBiTangent = bitangent;

		// Compute transformation matrices
		intersection->m_tangentToWorld = glm::mat3(tangent, bitangent, normal);
		intersection->m_worldToTangent = glm::transpose(intersection->m_tangentToWorld);

		return glm::length(r.origin - intersectionPoint);
	}
	return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius of 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r, ShadeableIntersection* intersection, bool &outside)
{
    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

	float A = pow(rd.x, 2.f) + pow(rd.y, 2.f) + pow(rd.z, 2.f);
	float B = 2*(rd.x*ro.x + rd.y * ro.y + rd.z * ro.z);
	float C = pow(ro.x, 2.f) + pow(ro.y, 2.f) + pow(ro.z, 2.f) - 0.25f;//Radius is 1.f
	float discriminant = B*B - 4*A*C;
	
	//If the discriminant is negative, then there is no real root
	if(discriminant < 0)
	{
		return -1;
	}

	float sqrtDiscriminant = sqrt(discriminant);

	float t = (-B - sqrtDiscriminant)/(2*A);

	if(t < 0)
	{
		t = (-B + sqrtDiscriminant)/(2*A);
	}
	
	if(t >= 0)
	{
		const glm::vec3 objspaceIntersection = ro + t*rd;
		const glm::vec3 intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));

		intersection->m_intersectionPointWorld = intersectionPoint;

		const glm::vec3 normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f))) * (outside ? -1.f : 1.f);
		const glm::vec3 tangent = glm::normalize(glm::mat3(sphere.transform) * glm::cross(glm::vec3(0.f, 1.f, 0.f), glm::normalize(objspaceIntersection)));
		const glm::vec3 bitangent = glm::normalize(glm::cross(normal, tangent));

		intersection->m_surfaceNormal = normal;
		intersection->m_surfaceTangent = tangent;
		intersection->m_surfaceBiTangent = bitangent;

		// Compute transformation matrices
		intersection->m_tangentToWorld = glm::mat3(tangent, bitangent, normal);
		intersection->m_worldToTangent = glm::transpose(glm::mat3(tangent, bitangent, normal));

		return glm::length(r.origin - intersectionPoint);
	}

	return -1;

    
}
