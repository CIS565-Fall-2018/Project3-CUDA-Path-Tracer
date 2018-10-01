#pragma once

#include "shapes.h"

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

		__host__ __device__ float Pdf(const ShadeableIntersection* refIntersection, const glm::vec3* wi, const Geom* plane, int num_geoms, Geom* geoms) 
		{
			Ray ray = SpawnRay(refIntersection, wi);

			ShadeableIntersection intersection;

			const bool didIntersect = SceneIntersect(&ray, num_geoms, geoms, &intersection);

			if(!didIntersect)
			{
				return 0.f;
			}

			const glm::vec3 distance = refIntersection->m_intersectionPointWorld - intersection.m_intersectionPointWorld;
			const float len = glm::length2(distance);
			const float cosAngle = glm::abs(glm::dot(intersection.m_surfaceNormal, -(*wi)));

			float pdf = 0.f;

			// Can't divide by zero
			if(cosAngle < 0.001) {
				pdf = 0.f;
			} else {
				pdf = (len) / (cosAngle * Area(plane));
			}
			return pdf;
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


#define RayEpsilon 0.00005f

	__host__ __device__ Ray SpawnRay(const ShadeableIntersection* intersection, const glm::vec3* direction)
	{
		Ray newRay;
		glm::vec3 originOffset = intersection->m_surfaceNormal * RayEpsilon;
		newRay.origin = intersection->m_intersectionPointWorld + (glm::dot(*direction, intersection->m_surfaceNormal) > 0 ? originOffset : -originOffset);
		newRay.direction = *direction;
		return newRay;
	}

	__host__ __device__ bool SceneIntersect(const Ray* ray, const int geoms_size, Geom* allGeoms, ShadeableIntersection* intersection) 
	{
		float t;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;

		ShadeableIntersection tempIntersection;
		ShadeableIntersection hitIntersection;

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = allGeoms[i];
			t = 0.0f;

			if (geom.type == CUBE)
			{
				t = Cube::TestIntersection(&geom, *ray, &tempIntersection, false);
			}
			else if (geom.type == SPHERE)
			{
				t = Sphere::TestIntersection(&geom, *ray, &tempIntersection, false);
			}
			else if (geom.type == PLANE)
			{
				t = SquarePlane::TestIntersection(&geom, *ray, &tempIntersection, false);
			}

			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				hitIntersection = tempIntersection;
				hitIntersection.m_geometryHitId = geom.geometryId;
			}
		}

		if (hit_geom_index != -1)
		{
			//The ray hits something
			intersection->m_surfaceNormal = hitIntersection.m_surfaceNormal;
			intersection->m_intersectionPointWorld = hitIntersection.m_intersectionPointWorld;
			intersection->m_geometryHitId = hitIntersection.m_geometryHitId;
			return true;
		}
		intersection->m_geometryHitId = -1;
		return false;
	}

} // namespace Shapes end







