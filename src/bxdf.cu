#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "common.h"
#include "warpfunctions.h"

enum BxDFType
{
	BSDF_REFLECTION = 1 << 0,   // This BxDF handles rays that are reflected off surfaces
	BSDF_TRANSMISSION = 1 << 1, // This BxDF handles rays that are transmitted through surfaces
	BSDF_DIFFUSE = 1 << 2,      // This BxDF represents diffuse energy scattering, which is uniformly random
	BSDF_GLOSSY = 1 << 3,       // This BxDF represents glossy energy scattering, which is biased toward certain directions
	BSDF_SPECULAR = 1 << 4,     // This BxDF handles specular energy scattering, which has no element of randomness
	BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION | BSDF_TRANSMISSION
};

//---------------------------------------------------------------------------------
// BxDF
//---------------------------------------------------------------------------------
struct Bxdf
{
	int id = 0;

	Bxdf() {}

	__device__ virtual glm::vec3 SampleF(const glm::vec3* wo, glm::vec3* wi, float* pdf, const glm::vec2* xi, const glm::vec3 color)
	{
		return glm::vec3(1.f);
	}

	__device__ virtual float Pdf(const glm::vec3* wo, glm::vec3* wi)
	{
		return 0.f;
	}

	__device__ virtual glm::vec3 F(const glm::vec3* wo, const glm::vec3* wi, const glm::vec3 color)
	{
		return glm::vec3(0.f);
	}
};

//---------------------------------------------------------------------------------
// Lamberts or Diffuse BRDF
//---------------------------------------------------------------------------------
struct LambertsBrdf : public Bxdf
{

	LambertsBrdf()
	{
		id = 1;
	}

	// Lamberts BRDF -------------------------------------------------------
	__device__ glm::vec3 F(const glm::vec3* wo, const glm::vec3* wi, const glm::vec3 color)
	{
		return color * InvPi;
	}

	__device__ float Pdf(const glm::vec3* wo, glm::vec3* wi)
	{
		return Common::SameHemisphere(wo, wi) ? Common::AbsCosTheta(wi) * InvPi : 0;
	}

	__device__ glm::vec3 SampleF(const glm::vec3* wo, glm::vec3* wi, float* pdf, const glm::vec2* xi, const glm::vec3 color)
	{
		// 1. Cosine sample the hemisphere
		*wi = WarpFunctions::SquareToHemisphereCosine(xi);

		if (wo->z < 0) {
			wi->z *= -1;
		}

		// 2. Calculate the pdf
		*pdf = Pdf(wo, wi);

		// 3. return f
		return F(wo, wi, color);
	}
};

//---------------------------------------------------------------------------------
// Specular BRDF
//---------------------------------------------------------------------------------
struct SpecularBrdf : public Bxdf
{

	SpecularBrdf()
	{
		id = 2;
	}

	// Specular BRDF -------------------------------------------------------
	__device__ glm::vec3 F(const glm::vec3* wo, const glm::vec3* wi, const glm::vec3 color)
	{
		return color;
	}

	__device__ float Pdf(const glm::vec3* wo, glm::vec3* wi)
	{
		return 1.f;
	}

	__device__ glm::vec3 SampleF(const glm::vec3* wo, glm::vec3* wi, float* pdf, const glm::vec2* xi, const glm::vec3 color)
	{
		*wi = glm::vec3(-(*wo).x, -(*wo).y, (*wo).z);
		*pdf = Pdf(wo, wi);
		return F(wo, wi, color);
	}
};
