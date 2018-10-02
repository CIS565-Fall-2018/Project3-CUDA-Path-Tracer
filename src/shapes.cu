#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

namespace Shapes
{
	/**
	* Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
	*/
	__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) 
	{
		return glm::vec3(m * v);
	}

	namespace SquarePlane
	{
		__host__ __device__ float Area(const Geom* plane)
		{
			return (plane->scale.x * plane->scale.y);
		}

		__host__ __device__ float TestIntersection(const Geom* plane, Ray r, ShadeableIntersection* intersection, bool outside) 
		{
			const glm::vec3 ray_origin = multiplyMV(plane->inverseTransform, glm::vec4(r.origin, 1.0f));
			const glm::vec3 ray_direction = glm::normalize(multiplyMV(plane->inverseTransform, glm::vec4(r.direction, 0.0f)));

			const float t = (glm::vec3(0.5f, 0.5f, 0) - ray_origin).z / ray_direction.z;
			const glm::vec3 P = glm::vec3(t * ray_direction + ray_origin);

			//Check that P is within the bounds of the square
			if(t > 0 && P.x >= -0.5f && P.x <= 0.5f && P.y >= -0.5f && P.y <= 0.5f)
			{
				const glm::vec3 objSpaceIntersection = P;
				const glm::vec3 intersectionPoint = multiplyMV(plane->transform, glm::vec4(objSpaceIntersection, 1.0f));
				intersection->m_intersectionPointWorld = intersectionPoint;

				// Computing normal, tangent and bitangent
				const glm::vec3 normal = glm::normalize(glm::mat3(plane->inverseTransform) * glm::vec3(0.f, 0.f, 1.f));
				const glm::vec3 tangent = glm::normalize(glm::vec3(plane->transform * glm::vec4(1.f, 0.f, 0.f, 0.f)));
				const glm::vec3 bitangent = glm::normalize(glm::vec3(plane->transform * glm::vec4(0.f, 1.f, 0.f, 0.f)));

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

		__host__ __device__ void Sample(const Geom* plane, const glm::vec2* xi, float *pdf, glm::vec3* intersectionPoint, glm::vec3* intersectionNormal)
		{
			*intersectionPoint = glm::vec3(plane->transform * glm::vec4((*xi)[0] - 0.5f, (*xi)[1] - 0.5f, 0.f, 1.f));
			*intersectionNormal = glm::normalize(glm::mat3(plane->inverseTransform) * glm::vec3(0.f, 0.f, 1.f));
			*pdf = 1.f / Area(plane);
		}
	}

	namespace Cube
	{
		namespace
		{
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
			
		} // Anonymous namespace end

		  /**
		  * Test intersection between a ray and a transformed cube. Untransformed,
		  * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
		  *
		  * @param intersectionPoint  Output parameter for point of intersection.
		  * @param normal             Output parameter for surface normal.
		  * @param outside            Output param for whether the ray came from outside.
		  * @return                   Ray parameter `t` value. -1 if no intersection.
		  */
		__host__ __device__ float TestIntersection(const Geom* box, Ray r, ShadeableIntersection* intersection, bool outside) 
		{
			glm::vec3 ray_origin = multiplyMV(box->inverseTransform, glm::vec4(r.origin   , 1.0f));
			glm::vec3 ray_direction = glm::normalize(multiplyMV(box->inverseTransform, glm::vec4(r.direction, 0.0f)));

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
				const glm::vec3 intersectionPoint = multiplyMV(box->transform, glm::vec4(objSpaceIntersection, 1.0f));
				intersection->m_intersectionPointWorld = intersectionPoint;

				// Computing normal, tangent and bitangent
				const int faceIndex = glm::abs(GetFaceIndex(objSpaceIntersection));
				const glm::vec3 localNormal = GetCubeNormal(objSpaceIntersection, faceIndex);
				const glm::vec3 localTangent = GetCubeTangent(objSpaceIntersection, faceIndex);

				const glm::vec3 normal = glm::normalize(multiplyMV(box->transform, glm::vec4(localNormal, 0.0f)));
				const glm::vec3 tangent = glm::normalize(glm::vec3(box->transform * glm::vec4(localTangent, 0.0f)));
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
	}

	namespace Sphere
	{
		/**
		* Test intersection between a ray and a transformed sphere. Untransformed,
		* the sphere always has radius of 0.5 and is centered at the origin.
		*
		* @param intersectionPoint  Output parameter for point of intersection.
		* @param normal             Output parameter for surface normal.
		* @param outside            Output param for whether the ray came from outside.
		* @return                   Ray parameter `t` value. -1 if no intersection.
		*/
		__host__ __device__ float TestIntersection(Geom* sphere, Ray r, ShadeableIntersection* intersection, bool outside)
		{
			glm::vec3 ro = multiplyMV(sphere->inverseTransform, glm::vec4(r.origin, 1.0f));
			glm::vec3 rd = glm::normalize(multiplyMV(sphere->inverseTransform, glm::vec4(r.direction, 0.0f)));

			float A = pow(rd.x, 2.f) + pow(rd.y, 2.f) + pow(rd.z, 2.f);
			float B = 2*(rd.x*ro.x + rd.y * ro.y + rd.z * ro.z);
			float C = pow(ro.x, 2.f) + pow(ro.y, 2.f) + pow(ro.z, 2.f) - 0.25f;//Radius is 0.5f
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
				const glm::vec3 intersectionPoint = multiplyMV(sphere->transform, glm::vec4(objspaceIntersection, 1.f));

				intersection->m_intersectionPointWorld = intersectionPoint;

				const glm::vec3 normal = glm::normalize(multiplyMV(sphere->invTranspose, glm::vec4(objspaceIntersection, 0.f))) * (outside ? -1.f : 1.f);
				const glm::vec3 tangent = glm::normalize(glm::mat3(sphere->transform) * glm::cross(glm::vec3(0.f, 1.f, 0.f), glm::normalize(objspaceIntersection)));
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
	}

	namespace Mesh
	{
		
	}

	namespace Implicit
	{

		#define SPHERE_RADIUS 1.f
		#define RAY_MARCH_EPSILON 0.000001f
		#define RAY_STEPS 100
		#define ROUNDED_BOX_RADIUS 0.1f
		#define CYLINDER_RADIUS 0.5f
		#define CYLINDER_HEIGHT 1.0f
		#define BOX_WIDTH 1.0f

		__host__ __device__ float SphereSDF(glm::vec3 point) 
		{
			return glm::length(point) - SPHERE_RADIUS;
		}

		__host__ __device__ float BoxSDF(glm::vec3 point)
		{
			glm::vec3 d = glm::abs(point) - glm::vec3(BOX_WIDTH);
			return glm::min(glm::max(d.x, glm::max(d.y,d.z)), 0.f) + glm::length(glm::max(d, glm::vec3(0.f)));
		}


		__host__ __device__ float UdRoundBox(glm::vec3 p)
		{
			return glm::length(glm::max(glm::abs(p)- glm::vec3(1.f), glm::vec3(0.0))) - ROUNDED_BOX_RADIUS;
		}

		__host__ __device__ float TorusSDF(glm::vec3 p) 
		{
			glm::vec2 t  = glm::vec2(2.f, 1);
			glm::vec2 q = glm::vec2(glm::length(glm::vec2(p.x, p.z)) - t.x, p.y);
			return glm::length(q) - t.y;
		}

		__host__ __device__ float SdCappedCylinder(glm::vec3 p)
		{
			glm::vec2 d = glm::abs(glm::vec2(glm::length(glm::vec2(p.x, p.z)), p.y)) - glm::vec2(CYLINDER_RADIUS, CYLINDER_HEIGHT);
			return glm::min(glm::max(d.x,d.y), 0.f) + glm::length(glm::max(d, 0.f));
		}

		__host__ __device__ float FuncUnion(float d1, float d2)
		{
			return glm::max(d1, d2);
		}

		__host__ __device__ float FuncIntersection(float d1, float d2)
		{
			return glm::min(d1, d2);
		}

		__host__ __device__ float FuncSubtract(float d1, float d2)
		{
			return glm::max(-d1, d2);
		}

		__host__ __device__ float FunkySDF(glm::vec3 p)
		{
			return 0.f;
		}

		__host__ __device__ float Raymarch(glm::vec3* eye, glm::vec3* dir, float max) 
		{
			float depth = 0.f;
			for(int i = 0; i < RAY_STEPS; ++i) 
			{
				const glm::vec3 point = *eye + depth * (*dir);
				float dist = TorusSDF(point);
				if(dist < RAY_MARCH_EPSILON) 
				{
					return depth;
				}
				if(depth >= max) 
				{
					return max;
				}
				depth += dist;
			}
			return max;
		}

		__host__ __device__ void ComputeTBN(const glm::vec3 P, const Geom* geometry, glm::vec3* nor, glm::vec3* tan, glm::vec3* bit)
		{

			const float epsilon = 0.0001f;

			float dx = TorusSDF(glm::vec3(P[0] + epsilon, P[1], P[2])) - TorusSDF(glm::vec3(P[0] - epsilon, P[1], P[2]));
			float dy = TorusSDF(glm::vec3(P[0], P[1] + epsilon, P[2])) - TorusSDF(glm::vec3(P[0], P[1] - epsilon, P[2]));
			float dz = TorusSDF(glm::vec3(P[0], P[1], P[2] + epsilon)) - TorusSDF(glm::vec3(P[0], P[1], P[2] - epsilon));

			glm::vec3 normal = glm::normalize(glm::vec3(dx, dy, dz));

			float tanX = P.x + epsilon;
			float tanY = P.y + epsilon;
			float tanZ = ((dx * (tanX - P.x) + dy * (tanY - P.y)) / (-dz)) + P.z;

			glm::vec3 tangent = glm::normalize(glm::vec3(tanX, tanY, tanZ));
			glm::vec3 bitangent = glm::normalize(glm::cross(normal, tangent));

			*nor = glm::normalize(glm::mat3(geometry->invTranspose) * normal);
			*tan = glm::normalize(glm::mat3(geometry->transform) * tangent);
			*bit = glm::normalize(glm::mat3(geometry->transform) * bitangent);
		}

		/**
		* Test intersection between a ray and a transformed sphere. Untransformed,
		* the sphere always has radius of 0.5 and is centered at the origin.
		*
		* @param intersectionPoint  Output parameter for point of intersection.
		* @param normal             Output parameter for surface normal.
		* @param outside            Output param for whether the ray came from outside.
		* @return                   Ray parameter `t` value. -1 if no intersection.
		*/
		__host__ __device__ float TestIntersection(Geom* geometry, const Ray* r, ShadeableIntersection* intersection, bool outside)
		{
			glm::vec3 ro = multiplyMV(geometry->inverseTransform, glm::vec4(r->origin, 1.0f));
			glm::vec3 rd = glm::normalize(multiplyMV(geometry->inverseTransform, glm::vec4(r->direction, 0.0f)));

			float max = 2000;

			float t = Raymarch(&ro, &rd, max);
			if(t < max) 
			{
				const glm::vec3 objspaceIntersection = ro + t*rd;
				const glm::vec3 intersectionPoint = multiplyMV(geometry->transform, glm::vec4(objspaceIntersection, 1.f));

				intersection->m_intersectionPointWorld = intersectionPoint;

				glm::vec3 normal(0.f);
				glm::vec3 tangent(0.f);
				glm::vec3 bitangent(0.f);

				ComputeTBN(objspaceIntersection, geometry, &normal, &tangent, &bitangent);

				intersection->m_surfaceNormal = normal;
				intersection->m_surfaceTangent = tangent;
				intersection->m_surfaceBiTangent = bitangent;

				// Compute transformation matrices
				intersection->m_tangentToWorld = glm::mat3(tangent, bitangent, normal);
				intersection->m_worldToTangent = glm::transpose(glm::mat3(tangent, bitangent, normal));

				return glm::length(r->origin - intersectionPoint);
			}
			return -1;
		}
	}


} // namespace Shapes end







