#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define MAX_STACK_SIZE 100

__host__ __device__ bool myEqual(float a, float b)
{
	if (glm::abs(a - b) < EPSILON)
		return true;
	else
		return false;
}

template<class T>
__host__ __device__ T barycentricInterpolate(const glm::vec3 &P, const glm::vec3 p[3], const T v[3])
{
	float S = 0.5f * glm::length(glm::cross(p[0] - p[1], p[2] - p[1]));
	float s0 = 0.5f * glm::length(glm::cross(p[1] - P, p[2] - P)) / S;
	float s1 = 0.5f * glm::length(glm::cross(p[2] - P, p[0] - P)) / S;
	float s2 = 0.5f * glm::length(glm::cross(p[0] - P, p[1] - P)) / S;
	return v[0] * s0 + v[1] * s1 + v[2] * s2;
}

//overload
template<class T>
__host__ __device__ T barycentricInterpolate(const glm::vec3 &P, const glm::vec3 p[3])
{
	float S = 0.5f * glm::length(glm::cross(p[0] - p[1], p[2] - p[1]));
	float s0 = 0.5f * glm::length(glm::cross(p[1] - P, p[2] - P)) / S;
	float s1 = 0.5f * glm::length(glm::cross(p[2] - P, p[0] - P)) / S;
	float s2 = 0.5f * glm::length(glm::cross(p[0] - P, p[1] - P)) / S;
	return T() * s0 + T() * s1 + T() * s2;
}

//full specialization of overload
template<>
__host__ __device__ float barycentricInterpolate<float>(const glm::vec3 &P, const glm::vec3 p[3])
{
	float S = 0.5f * glm::length(glm::cross(p[0] - p[1], p[2] - p[1]));
	float s0 = 0.5f * glm::length(glm::cross(p[1] - P, p[2] - P)) / S;
	float s1 = 0.5f * glm::length(glm::cross(p[2] - P, p[0] - P)) / S;
	float s2 = 0.5f * glm::length(glm::cross(p[0] - P, p[1] - P)) / S;
	return s0 + s1 + s2;
}

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

//My code here
__host__ __device__ void ComputeTBN(const glm::mat4 &T, const glm::mat4 &invTransT, glm::vec3 &t, glm::vec3 &b, glm::vec3 &n)
{
	glm::vec3 norL = glm::normalize(n);
	glm::vec3 tanL(0,-1,0);

	if (glm::abs(glm::abs(glm::dot(norL, glm::vec3(1.0, 0, 0))) - 1.0) > 0.5)//local normal is not parallel to x axis// (norL != glm::vec3(1,0,0) && norL != glm::vec3(-1,0,0))//
		tanL = glm::vec3(1.0, 0, 0);
	else
		tanL = glm::vec3(0, 1.0, 0);

	tanL = glm::normalize(glm::cross(norL, tanL));

	n = glm::normalize(multiplyMV(invTransT, glm::vec4(norL, 0)));
	t = glm::normalize(multiplyMV(T, glm::vec4(tanL, 0)));
	b = glm::normalize(multiplyMV(T, glm::vec4(glm::cross(norL, tanL), 0)));
}

//My code here
__host__ __device__ void ComputeTriangleTBNUVOutside(const glm::mat4 &T, const glm::mat4 &invTransT, Ray& r, const Triangle& triangle, const glm::vec3& P, glm::vec3 &t, glm::vec3 &b, glm::vec3 &n, glm::vec2 &uv, bool &outside)
{
	glm::vec3 normal = glm::normalize(barycentricInterpolate<glm::vec3>(P, triangle.v, triangle.n));
	uv = barycentricInterpolate<glm::vec2>(P, triangle.v, triangle.t);
	outside = glm::dot(r.direction, triangle.planeNormal) > 0 ? false : true;

	////calculate tangent and bitangent according to the uv axes direction of the triangle
	//glm::vec2 deltaUV1 = triangle.t[1] - triangle.t[0];
	//glm::vec2 deltaUV2 = triangle.t[2] - triangle.t[0];
	//glm::vec3 deltaP1 = triangle.v[1] - triangle.v[0];
	//glm::vec3 deltaP2 = triangle.v[2] - triangle.v[0];

	//glm::vec3 tangent = glm::normalize((deltaUV2.y * deltaP1 - deltaUV1.y * deltaP2) / (deltaUV2.y * deltaUV1.x - deltaUV1.y * deltaUV2.x));
	////(deltaUV2.y * deltaUV1.x - deltaUV1.y * deltaUV2.x) can't be 0
	////otherwise it will not be a triangle
	////but deltaUV2.y can
	////when deltaUV2.y is 0, deltaUV1.y can't be 0
	//glm::vec3 bitangent = glm::normalize(deltaUV2.y == 0 ? (deltaP1 - deltaUV1.x * t) / deltaUV1.y : (deltaP2 - deltaUV2.x * t) / deltaUV2.y);

	//n = glm::normalize(multiplyMV(invTransT, glm::vec4(normal, 0)));
	//t = glm::normalize(multiplyMV(T, glm::vec4(tangent, 0)));
	//b = glm::normalize(multiplyMV(T, glm::vec4(bitangent, 0)));

	//traditional way
	glm::vec3 norL = normal;
	glm::vec3 tanL(0, -1, 0);

	if (glm::abs(glm::abs(glm::dot(norL, glm::vec3(1.0, 0, 0))) - 1.0) > 0.5)//local normal is not parallel to x axis// (norL != glm::vec3(1,0,0) && norL != glm::vec3(-1,0,0))//
		tanL = glm::vec3(1.0, 0, 0);
	else
		tanL = glm::vec3(0, 1.0, 0);

	tanL = glm::normalize(glm::cross(norL, tanL));

	n = glm::normalize(multiplyMV(invTransT, glm::vec4(norL, 0)));
	t = glm::normalize(multiplyMV(T, glm::vec4(tanL, 0)));
	b = glm::normalize(multiplyMV(T, glm::vec4(glm::cross(norL, tanL), 0)));
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
__host__ __device__ float boxIntersectionTest(const Geom& box, const Ray& r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, 
		glm::vec3 &tangent, glm::vec3 &bitangent,//My code here.
		bool &outside) {
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
		//normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f))); //what the hell? translating normal should always use invTransposeT
		//My code here.
		normal = tmin_n;
		ComputeTBN(box.transform, box.invTranspose, tangent, bitangent, normal);
		
		//if (box.translation.y > 14)
		//	printf("%f,%f,%f--%f,%f,%f--%f,%f,%f\n", tangent.x, tangent.y, tangent.z, bitangent.x, bitangent.y, bitangent.z, normal.x, normal.y, normal.z);
		
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
__host__ __device__ float sphereIntersectionTest(const Geom& sphere, const Ray& r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal,
	glm::vec3 &tangent, glm::vec3 &bitangent,//My code here.
	bool &outside) {
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
    //normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
	//My code here.
	normal = objspaceIntersection;
	if (!outside) {
        normal = -normal;
    }
	ComputeTBN(sphere.transform, sphere.invTranspose, tangent, bitangent, normal);

    return glm::length(r.origin - intersectionPoint);
}


__host__ __device__ float squareIntersectionTest(const Geom& square, const Ray& r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal,
	glm::vec3 &tangent, glm::vec3 &bitangent,//My code here.
	bool &outside)
{

	float half_width = 0.5f;

	glm::vec3 origin = multiplyMV(square.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 direction = glm::normalize(multiplyMV(square.inverseTransform, glm::vec4(r.direction, 0.0f)));

	float t = -1;
	glm::vec3 p;

	if (direction.z < 0)
	{
		normal = glm::vec3(0, 0, 1);
		t = glm::dot(-origin, normal) / glm::dot(direction, normal);
	}
	else if (direction.z > 0)
	{
		normal = glm::vec3(0, 0, -1);
		t = glm::dot(-origin, normal) / glm::dot(direction, normal);
	}
	else
	{
		return -1;
	}

	if (t > 0)
	{
		p = origin + direction * t;
		if (p.y >= -half_width && p.y <= half_width && p.x >= -half_width && p.x <= half_width)
		{
			outside = false;
			intersectionPoint = multiplyMV(square.transform, glm::vec4(p, 1.f));
			ComputeTBN(square.transform, square.invTranspose, tangent, bitangent, normal);
		}
		else
		{
			return -1;
		}
	}
	else
	{
		return -1;
	}

	return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool aabbIntersectionTest(const Ray& r, const glm::vec3& minAABB, const glm::vec3& maxAABB)
{

	const glm::vec3 &origin = r.origin;
	const glm::vec3 &direction = r.direction;

	if (origin.x<maxAABB.x && origin.y<maxAABB.y && origin.z<maxAABB.z &&
		origin.x>minAABB.x && origin.y>minAABB.y && origin.z>minAABB.z)
		return true;

	if (direction.x < 0)
	{
		glm::vec3 normal(1, 0, 0);
		float t = glm::dot(maxAABB - origin, normal) / glm::dot(direction, normal);
		if (t > 0)
		{
			glm::vec3 p = origin + direction * t;
			if (p.y >= minAABB.y&&p.y <= maxAABB.y&&p.z >= minAABB.z&&p.z <= maxAABB.z)
			{
				return true;
			}
		}
	}
	else if (direction.x > 0)
	{
		glm::vec3 normal(-1, 0, 0);
		float t = glm::dot(minAABB - origin, normal) / glm::dot(direction, normal);
		if (t > 0)
		{
			glm::vec3 p = origin + direction * t;
			if (p.y >= minAABB.y&&p.y <= maxAABB.y&&p.z >= minAABB.z&&p.z <= maxAABB.z)
			{
				return true;
			}
		}
	}

	if (direction.y < 0)
	{
		glm::vec3 normal(0, 1, 0);
		float t = glm::dot(maxAABB - origin, normal) / glm::dot(direction, normal);
		if (t > 0)
		{
			glm::vec3 p = origin + direction * t;
			if (p.x >= minAABB.x&&p.x <= maxAABB.x&&p.z >= minAABB.z&&p.z <= maxAABB.z)
			{
				return true;

			}
		}

	}
	else if (direction.y > 0)
	{
		glm::vec3 normal(0, -1, 0);
		float t = glm::dot(minAABB - origin, normal) / glm::dot(direction, normal);
		if (t > 0)
		{
			glm::vec3 p = origin + direction * t;
			if (p.x >= minAABB.x&&p.x <= maxAABB.x&&p.z >= minAABB.z&&p.z <= maxAABB.z)
			{
				return true;
			}
		}
	}

	if (direction.z < 0)
	{
		glm::vec3 normal(0, 0, 1);
		float t = glm::dot(maxAABB - origin, normal) / glm::dot(direction, normal);
		if (t > 0)
		{
			glm::vec3 p = origin + direction * t;
			if (p.y >= minAABB.y&&p.y <= maxAABB.y&&p.x >= minAABB.x&&p.x <= maxAABB.x)
			{
				return true;
			}
		}
	}
	else if (direction.z > 0)
	{
		glm::vec3 normal(0, 0, -1);
		float t = glm::dot(minAABB - origin, normal) / glm::dot(direction, normal);
		if (t > 0)
		{
			glm::vec3 p = origin + direction * t;
			if (p.y >= minAABB.y&&p.y <= maxAABB.y&&p.x >= minAABB.x&&p.x <= maxAABB.x)
			{
				return true;
			}
		}
	}

	return false;
}

__host__ __device__ float triangleIntersectionTest(const Triangle& triangle, const Ray& r) 
{
	//1. Ray-plane intersection
	float t = glm::dot(triangle.planeNormal, (triangle.v[0] - r.origin)) / glm::dot(triangle.planeNormal, r.direction);
	if (t < 0) return -1;

	//2. Barycentric test
	glm::vec3 P = r.origin + t * r.direction;
	float sum = barycentricInterpolate<float>(P, triangle.v);

	if (myEqual(sum, 1.0f))
	{
		return t;
	}
		
	return -1;
}

__host__ __device__ float nodeIntersectionTest(const Triangle * triangles, const int triangleIndex, const Ray& r, int* hitIndex)
{
	if (triangleIndex == -1)
		return -1;

	const Triangle triangle = triangles[triangleIndex];

	float t = -1, t_tmp = -1;
	int hit = -1, hit_tmp = -1;

	//root
	if (aabbIntersectionTest(r, triangle.min, triangle.max))
	{

		t = triangleIntersectionTest(triangle, r);
		if (t > 0)
		{
			hit = triangleIndex;
		}

		//left sub-tree
		if (triangle.leftIndex != -1)
		{
			t_tmp = nodeIntersectionTest(triangles, triangle.leftIndex, r, &hit_tmp);
			if ((t_tmp > 0 && t < 0) || (t_tmp > 0 && t > 0 && t_tmp < t))
			{
				t = t_tmp;
				hit = hit_tmp;
			}
		}

		//right sub-tree
		if (triangle.rightIndex != -1)
		{
			t_tmp = nodeIntersectionTest(triangles, triangle.rightIndex, r, &hit_tmp);
			if ((t_tmp > 0 && t < 0) || (t_tmp > 0 && t > 0 && t_tmp < t))
			{
				t = t_tmp;
				hit = hit_tmp;
			}
		}

		//printf("triangleIndex: %d, hit: %d, t: %f\n", triangleIndex, hit, t);
	}
	*hitIndex = hit;
	return t;
}

__host__ __device__ float nodeIntersectionTestLoop(const Triangle * triangles, const int triangleIndex, const Ray& r, int* hitIndex)
{
	if (triangleIndex == -1)
		return -1;

	int stack[MAX_STACK_SIZE] = { -1 };
	stack[0] = triangleIndex;
	int top = 0;

	float t = -1;
	int hit = -1;

	while (top > -1)
	{
		int currentIndex = stack[top];
		Triangle current = triangles[currentIndex];
		if(aabbIntersectionTest(r, current.min, current.max))
		{
			float t_tmp = triangleIntersectionTest(current, r);
			top--;
			if (t_tmp > 0 && ((t > 0 && t_tmp < t)||(t<0)))//t_tmp is valid and smaller than previous t_tmp, or t_tmp is valid and no previous t_tmp exists
			{
				hit = currentIndex;
				t = t_tmp;
			}

			if (current.leftIndex != -1 && top < MAX_STACK_SIZE - 1)
			{
				stack[++top] = current.leftIndex;
			}

			if (current.rightIndex != -1 && top < MAX_STACK_SIZE - 1)
			{
				stack[++top] = current.rightIndex;
			}
		}
		else
		{
			top--;
		}
	}

	*hitIndex = hit;
	return t;
}

__host__ __device__ float meshIntersectionTest(const Triangle * triangles, const Geom& mesh, const Ray& r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal,
	glm::vec3 &tangent, glm::vec3 &bitangent, glm::vec2 &uv,//My code here.
	bool &outside) {

	glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = rd;

	int hit = -1;
	float t = nodeIntersectionTestLoop(triangles, mesh.root, rt, &hit); //nodeIntersectionTest(triangles, mesh.root, rt, &hit);

	if (t > 0)//hit
	{
		glm::vec3 objspaceIntersection = rt.origin + t * rt.direction;
		//compute tangent, bitangent and normal for intersection, then transform them into world space
		ComputeTriangleTBNUVOutside(mesh.transform, mesh.invTranspose, rt, triangles[hit], objspaceIntersection, tangent, bitangent, normal, uv, outside);
		//printf("hit: %d, outside: %d\n", hit, outside);
		if (!outside)
		{
			normal = -normal;
		}
		//convert ot world space
		intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objspaceIntersection, 1.f));
		return glm::length(r.origin - intersectionPoint);
	}
	else//not hit
	{
		return -1;
	}
}
