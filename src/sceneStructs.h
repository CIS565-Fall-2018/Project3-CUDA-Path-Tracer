#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

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
	PLANE,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom 
{
	int geometryId;
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};


struct Material 
{
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;

	struct {
		float etaA;
		float etaB;
	} refractive;

    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

	BxDFType bxdfTypes[MAX_BXDFS];
	int numBxdfs;
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

struct PathSegment 
{
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	int remainingBounces;
	bool isRayDead;
	bool isRefractedRay;
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

	int m_geometryHitId;
};
