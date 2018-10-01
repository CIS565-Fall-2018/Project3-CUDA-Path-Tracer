#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "sceneStructs.h"
#include "warpfunctions.cu"

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
		__host__ __device__ glm::vec3 L(const Material* material, ShadeableIntersection* intersection, const glm::vec3 &w)
		{
			// This seems to be giving the intersection with the light geometry
			const glm::vec3 normal = intersection->m_surfaceNormal;
			const bool isFront = glm::dot(normal, w) > 0;
			return isFront? (material->emittance * material->color) : glm::vec3(0.f);
		}

		__host__ __device__ glm::vec3 Sample_Li(Geom* lightGeometry, glm::vec2* xi, float* pdf, const Material* material, ShadeableIntersection* intersection)
		{
			return glm::vec3(1.0f);
		}

		__host__ __device__ float Pdf_Li(ShadeableIntersection* intersection, const glm::vec3 &wi)
		{
			return 0.f;
			//return shape->Pdf(ref, wi);
		}

	} // namespace DiffuseAreaLight end

} // namespace Lights end







