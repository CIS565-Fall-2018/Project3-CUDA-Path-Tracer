#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

#include "sceneStructs.h"
#include "shapes.cu"

namespace Lights
{

#define EPSILON 0.001
#define ISZERO(p) abs(p) < EPSILON

	__host__ __device__ bool isSamplePoint(const glm::vec3* p1, const glm::vec3* p2) 
	{
		return glm::distance(*p1, *p2) < EPSILON;
	}

	// Could potentially keep other type of lights
	namespace DiffuseAreaLight
	{
		__host__ __device__ glm::vec3 L(const Material* material, const glm::vec3* point, const glm::vec3* normal)
		{
			// This seems to be giving the intersection with the light geometry
			const bool isFront = glm::dot(*normal, -(*point)) > 0;
			return isFront? (material->emittance * material->color) : glm::vec3(0.f);
		}

		__host__ __device__ glm::vec3 Sample_Li(Geom* lightGeometry, glm::vec2* xi, glm::vec3* wiW, float* pdf, const Material* material, ShadeableIntersection* refIntersection)
		{

			// 1. Get an Intersection on the surface of its Shape by invoking shape->Sample.

			// Get the intersection with the geometry of the object
			glm::vec3 intersection_point(0.f);
			glm::vec3 intersection_normal(0.f);

			// Currently this is the only implemented shape
			if(lightGeometry->type == PLANE)
			{
				Shapes::SquarePlane::Sample(lightGeometry, xi, pdf, &intersection_point, &intersection_normal);
			}

			if(isSamplePoint(&intersection_point, &refIntersection->m_intersectionPointWorld)) 
			{
				*pdf = 0;
				return glm::vec3(0.f);
			}

			// Vector from reference point to out light's intersection point
			const glm::vec3 wi = glm::normalize(intersection_point - intersection_point);

			// distance betweern light and intersection point
			const float r = glm::distance2(intersection_point, refIntersection->m_intersectionPointWorld);

			// grazing angle
			const float cosAngle = glm::abs(glm::dot(intersection_normal, -wi));

			// initial area of shape of light
			const float area = *pdf;

			// Can't divide by zero
			if(cosAngle == 0.f) 
			{
				*pdf = 0.f;
			} 
			else 
			{
				*pdf = (r) / (cosAngle / area);
			}

			// 2. Check if the resultant PDF is zero or that the reference Intersection
			//    and the resultant Intersection are the same point in space, and return black if this is the case.
			if(ISZERO(*pdf)) 
			{
				return glm::vec3(0.f);
			}

			// 3. Set wi to the normalized vector from the reference Intersection's point to the Shape's intersection point.
			*wiW = glm::normalize(intersection_point - refIntersection->m_intersectionPointWorld);

			// 4. Return the light emitted along ωi from our intersection point
			return L(material, wiW, &intersection_normal);

		}

		__host__ __device__ float Pdf_Li(ShadeableIntersection* intersection, const glm::vec3 &wi)
		{
			// TODO:
			return 0.f;
			//return shape->Pdf(ref, wi);
		}

	} // namespace DiffuseAreaLight end

} // namespace Lights end







