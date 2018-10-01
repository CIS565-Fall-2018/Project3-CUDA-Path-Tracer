#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include "common.cu"
#include "warpfunctions.cu"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

#define MAX_BXDFS 3

enum BxDFType
{
	BSDF_REFLECTION = 1 << 0,   // This BxDF handles rays that are reflected off surfaces
	BSDF_TRANSMISSION = 1 << 1, // This BxDF handles rays that are transmitted through surfaces
	BSDF_DIFFUSE = 1 << 2,      // This BxDF represents diffuse energy scattering, which is uniformly random
	BSDF_GLOSSY = 1 << 3,       // This BxDF represents glossy energy scattering, which is biased toward certain directions
	BSDF_SPECULAR = 1 << 4,     // This BxDF handles specular energy scattering, which has no element of randomness
	BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION | BSDF_TRANSMISSION
};


enum GeomType 
{
    SPHERE,
    CUBE,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom 
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

//---------------------------------------------------------------------------------
// BxDF
//---------------------------------------------------------------------------------
struct Bxdf
{
	__device__ virtual glm::vec3 SampleF(const glm::vec3* wo, glm::vec3* wi, float* pdf, const glm::vec2* xi, const glm::vec3 color);
	__device__ virtual float Pdf(const glm::vec3* wo, glm::vec3* wi);
	__device__ virtual glm::vec3 F(const glm::vec3* wo, const glm::vec3* wi, const glm::vec3 color);
};

//---------------------------------------------------------------------------------
// Lamberts or Diffuse BRDF
//---------------------------------------------------------------------------------
struct LambertsBrdf : public Bxdf
{
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

struct Material 
{
	glm::vec3 color;
	struct {
		float exponent;
		glm::vec3 color;
	} specular;
	float hasReflective;
	float hasRefractive;
	float indexOfRefraction;
	float emittance;

	Bxdf bxdfs[MAX_BXDFS];
	int numBxdfs = 0;

	void AddBxdf(BxDFType bxdfType)
	{
		if(bxdfType == BSDF_DIFFUSE && numBxdfs < (MAX_BXDFS - 1))
		{
			bxdfs[numBxdfs++] = LambertsBrdf();
		}
	}
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	int remainingBounces;
	bool isRayDead;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection 
{
	glm::vec3 m_intersectionPointWorld;
	bool m_didIntersect;
	float t;
	glm::vec3 m_surfaceNormal;
	glm::vec3 m_surfaceTangent;
	glm::vec3 m_surfaceBiTangent;

	glm::mat3 m_worldToTangent;
	glm::mat3 m_tangentToWorld;

	int materialId;
};
