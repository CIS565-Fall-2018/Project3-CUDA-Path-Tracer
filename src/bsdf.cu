#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "sceneStructs.h"
#include "warpfunctions.cu"



namespace BSDF
{
	namespace
	{
		// Lamberts BRDF -------------------------------------------------------
		__host__ __device__ glm::vec3 Lamberts_F(const glm::vec3* wo, const glm::vec3* wi, const Material* material)
		{
			return material->color * InvPi;
		}

		__host__ __device__ float Lamberts_Pdf(const glm::vec3* wo, glm::vec3* wi)
		{
			return Common::SameHemisphere(wo, wi) ? Common::AbsCosTheta(wi) * InvPi : 0;
		}

		__host__ __device__ glm::vec3 Lamberts_SampleF(const glm::vec3* wo, glm::vec3* wi, float* pdf, const glm::vec2* xi, const Material* material)
		{
			// 1. Cosine sample the hemisphere
			*wi = WarpFunctions::SquareToHemisphereCosine(xi);

			if (wo->z < 0) {
				wi->z *= -1;
			}

			// 2. Calculate the pdf
			*pdf = Lamberts_Pdf(wo, wi);

			// 3. return f
			return Lamberts_F(wo, wi, material);
		}

		// Specular BRDF -------------------------------------------------------
		__host__ __device__ glm::vec3 SpecularR_F(const glm::vec3* wo, const glm::vec3* wi, const Material* material)
		{
			return material->color;
		}

		__host__ __device__ float SpecularR_Pdf(const glm::vec3* wo, glm::vec3* wi)
		{
			return 1.f;
		}

		__host__ __device__ glm::vec3 SpecularR_SampleF(const glm::vec3* wo, glm::vec3* wi, float* pdf, const glm::vec2* xi, const Material* material)
		{
			*wi = glm::vec3(-(*wo).x, -(*wo).y, (*wo).z);
			*pdf = SpecularR_Pdf(wo, wi);
			return SpecularR_F(wo, wi, material);
		}

		// Specular BTDF -------------------------------------------------------

		__host__ __device__ glm::vec3 SpecularT_F(const glm::vec3* wo, const glm::vec3* wi, const Material* material)
		{
			return material->color;
		}

		__host__ __device__ float SpecularT_Pdf(const glm::vec3* wo, glm::vec3* wi)
		{
			return 1.f;
		}

		__host__ __device__ glm::vec3 SpecularT_SampleF(const glm::vec3* wo, glm::vec3* wi, float* pdf, const glm::vec2* xi, const Material* material)
		{
			const bool entering = wo->z > 0;

			const float etaA = material->refractive.etaA;
			const float etaB = material->refractive.etaB;

			const float etaI = entering ? etaA : etaB;
			const float etaT = entering ? etaB : etaA;

			// Check if refraction has occured (to see for total internal reflection)
			const bool refracted = Common::Refract(*wo, Common::Faceforward(glm::vec3(0.f, 0.f, 1.f), *wo), etaI / etaT, wi);

			if (!refracted)
				return glm::vec3(0.f);

			*pdf = 1.f;

			// TODO :: Add Fresnel
			const glm::vec3 ft = material->color / Common::AbsCosTheta(wi);

			return ft;
		}

		// TODO: Add more bxdfs

	} // Anonymous namespace end

	__host__ __device__ glm::vec3 Sample_F(glm::vec3 woW, glm::vec3* wiW, float* pdf, glm::vec2* xi, const Material* material, ShadeableIntersection* intersection)
	{
		// TODO: 

		// 1. Select Random Bxdf
		const int numBxDFs = material->numBxdfs;
		// TODO : This can be done later as we are using only one material

		int randomBxdf = int((*xi)[0] * numBxDFs) % numBxDFs;
		BxDFType selBxdf = material->bxdfTypes[randomBxdf];

		// 2. Rewriting the random number
		glm::vec2 temp = glm::vec2((*xi)[0], (*xi)[1]);

		// 3. Converting wo, wi to tangent space
		const glm::vec3 woL = intersection->m_worldToTangent * (woW);
		glm::vec3 wiL;// = worldToTangent * (*wiW);

		// 4. Getting the color of the random bxdf

		glm::vec3 selBxdfCol(0.f);

		// TODO : Optimize this
		if(selBxdf == BxDFType::BSDF_REFLECTION)
		{
			selBxdfCol = SpecularR_SampleF(&woL, &wiL, pdf, &temp, material);
		}
		else if(selBxdf == BxDFType::BSDF_TRANSMISSION)
		{
			selBxdfCol = SpecularT_SampleF(&woL, &wiL, pdf, &temp, material);
		}
		else if(selBxdf == BxDFType::BSDF_DIFFUSE)
		{
			selBxdfCol = Lamberts_SampleF(&woL, &wiL, pdf, &temp, material);
		}

		const glm::vec3 wow = intersection->m_tangentToWorld * wiL;

		*wiW = wow;

		if(selBxdf == BxDFType::BSDF_REFLECTION || selBxdf == BxDFType::BSDF_TRANSMISSION)
		{
			return selBxdfCol;
		}

		return selBxdfCol;
	}

	__host__ __device__ glm::vec3 F(const glm::vec3* woW, const glm::vec3* wiW, const Material* material, const ShadeableIntersection* intersection)
	{
		glm::vec3 color(0.f);

		glm::vec3 woL = intersection->m_worldToTangent * (*woW);
		glm::vec3 wiL = intersection->m_worldToTangent * (*wiW);

		/*for (int i = 0; i < numBxDFs; ++i) {
			sum += bxdfs[i]->MatchesFlags(flags) ? bxdfs[i]->f(woL, wiL) : Color3f(0.f);
		}*/

		// TODO : This can be done later as we are using only one material
		if(material->hasReflective)
		{
			color += SpecularR_F(&woL, &wiL, material);
		}
		else if(material->hasRefractive)
		{
			color += SpecularT_F(&woL, &wiL, material);
		}
		else
		{
			color += Lamberts_F(&woL, &wiL, material);
		}

		return color;
	}
	
	__host__ __device__ float Pdf(const glm::vec3* woW, const glm::vec3* wiW, const Material* material, const ShadeableIntersection* intersection)
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
		if(material->hasReflective)
		{
			sumPdf = SpecularR_Pdf(&woL, &wiL);
		}
		else if(material->hasRefractive)
		{
			sumPdf = SpecularT_Pdf(&woL, &wiL);
		}
		else
		{
			sumPdf = Lamberts_Pdf(&woL, &wiL);
		}

		//sumPdf /= (1.f * numPdfs);
		return sumPdf;
	}
} // namespace BSDF end







