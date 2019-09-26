#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
	TRIANGLE
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

	// Triangle information
	glm::vec3 pos[3];
	glm::vec3 nor[3];
	glm::vec2 uv[3];
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
	float dispersion;
    float emittance;
	int textureOffset;
	int texWidth;
	int texHeight;
	int normalOffset;
	int norWidth;
	int norHeight;
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
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 uv;
};

struct Bounds {
	glm::vec3 min;
	glm::vec3 max;
};

struct KDHelperNode {
	Bounds bounds;
	KDHelperNode *left;
	KDHelperNode *right;
	uint8_t axis;
	std::vector<Geom> geoms;
};

struct KDNode {
	Bounds bounds; // 24 bytes
	union { // 4 bytes
		uint32_t primOffset; // if leaf
		uint32_t secondChildOffset; // if interior
	};
	uint16_t numPrims; // non zero if leaf
	uint8_t axis; // 1 byte
	uint8_t pad[1];
};
