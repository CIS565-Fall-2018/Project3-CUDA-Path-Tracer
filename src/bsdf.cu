#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "sceneStructs.h"
#include "warpfunctions.cu"

enum BxDFType
{
	BSDF_REFLECTION = 1 << 0,   // This BxDF handles rays that are reflected off surfaces
	BSDF_TRANSMISSION = 1 << 1, // This BxDF handles rays that are transmitted through surfaces
	BSDF_DIFFUSE = 1 << 2,      // This BxDF represents diffuse energy scattering, which is uniformly random
	BSDF_GLOSSY = 1 << 3,       // This BxDF represents glossy energy scattering, which is biased toward certain directions
	BSDF_SPECULAR = 1 << 4,     // This BxDF handles specular energy scattering, which has no element of randomness
	BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION | BSDF_TRANSMISSION
};

namespace BSDF
{
	namespace
	{
		// Lamberts Material-------------------------------------------------------
		__host__ __device__ glm::vec3 Lamberts_F(const glm::vec3* wo, const glm::vec3* wi, const Material* material)
		{
			return material->color * Common::InvPi;
		}

		__host__ __device__ float Lamberts_Pdf(const glm::vec3* wo, glm::vec3* wi)
		{
			return Common::SameHemisphere(wo, wi) ? Common::AbsCosTheta(wi) * Common::InvPi : 0;
		}

		__host__ __device__ glm::vec3 Lamberts_SampleF(const glm::vec3* wo, glm::vec3* wi, float* pdf, const glm::vec2* xi, const Material* material)
		{
			// 1. Cosine sample the hemisphere
			*wi = WarpFunctions::SquareToHemisphereCosine(xi);

			if (wo->z < 0) {
				wi->z = -wi->z;
			}

			// 2. Calculate the pdf
			*pdf = Lamberts_Pdf(wo, wi);

			// 3. return f
			return Lamberts_F(wo, wi, material);
		}

		// TODO: Add more bxdfs

	} // Anonymous namespace end

	__host__ __device__ glm::vec3 Sample_F(const glm::vec3* woW, glm::vec3* wiW, float* pdf, glm::vec2* xi, const Material* material, const ShadeableIntersection* intersection)
	{
		// TODO: 

		// 1. Select Random Bxdf
		const int numBxDFs = 1;
		// TODO : This can be done later as we are using only one material

		// 2. Rewriting the random number
		glm::vec2 temp = glm::vec2(xi[0], xi[1]);

		// 3. Converting wo, wi to tangent space
		const glm::vec3 woL = intersection->m_worldToTangent * (*woW);
		glm::vec3 wiL;// = worldToTangent * (*wiW);

		// 4. Getting the color of the random bxdf
		const glm::vec3 selBxdfCol = Lamberts_SampleF(&woL, &wiL, pdf, &temp, material);
		*wiW = intersection->m_tangentToWorld * wiL;

		// if it is glass material, then we dont need to check other bxdf as it will only reflect in one direction
		// TODO : This can be done later as we are using only one material

		// 5. Finding the average pdf of the remaining bxdfs
		// TODO : This can be done later as we are using only one material

		// 6. Iterate through bxdf and sum result of f()
		// TODO : This can also be done later as we are using only one material

		return selBxdfCol;
	}

	__host__ __device__ glm::vec3 F(const glm::vec3* woW, const glm::vec3* wiW, const Material* material, const ShadeableIntersection* intersection, const BxDFType flags)
	{
		glm::vec3 color(0.f);

		glm::vec3 woL = intersection->m_worldToTangent * (*woW);
		glm::vec3 wiL = intersection->m_worldToTangent * (*wiW);

		/*for (int i = 0; i < numBxDFs; ++i) {
			sum += bxdfs[i]->MatchesFlags(flags) ? bxdfs[i]->f(woL, wiL) : Color3f(0.f);
		}*/

		// TODO : This can be done later as we are using only one material

		color += Lamberts_F(&woL, &wiL, material);

		return color;
	}
	
	__host__ __device__ float Pdf(const glm::vec3* woW, const glm::vec3* wiW, const Material* material, const ShadeableIntersection* intersection, const BxDFType flags)
	{
		int numPdfs = 0;
		float sumPdf = 0;

		// Converting them to tangent space before sending to bxdf pdf
		glm::vec3 woL = intersection->m_worldToTangent * (*woW);
		glm::vec3 wiL = intersection->m_worldToTangent * (*wiW);

		/*for (int i = 0; i < numBxDFs; ++i) {
			if (bxdfs[i]->MatchesFlags(flags)) {
				sumPdf += bxdfs[i]->Pdf(woL, wiL);
				numPdfs++;
			}
		}*/
		// TODO : This can be done later as we are using only one material
		sumPdf = Lamberts_Pdf(&woL, &wiL);

		//sumPdf /= (1.f * numPdfs);
		return sumPdf;
	}
} // namespace BSDF end







