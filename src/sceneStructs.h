#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
	DIAMOND,
	MANDELBULB,
	TRIANGLE
};

struct Bounds {
	glm::vec3 max = glm::vec3(-999999999.f);
	glm::vec3 min = glm::vec3(99999999.f);
};

struct Triangle {
	glm::vec3 pts[3];
	glm::vec2 uvs[3];
	glm::vec3 normals[3];
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Texture {
	int width, height;
	glm::vec3 *data;
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
	Triangle t;
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
    float emittance;
	float textureOffset = -1;
	int tex_height, tex_width;
	float normMapOffset = -1;
	int n_m_height, n_m_width;
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
  glm::vec2 uvs;
};

struct KDTreeNode {
	Bounds bounds;
	KDTreeNode *left;
	KDTreeNode *right;
	std::vector<Geom> geoms;
	int axis;
};

struct LinearKDNode {
	Bounds bounds;
	union {
		int primitivesOffset;
		int secondChildOffset;
	};
	uint16_t nPrimitives;
	uint8_t axis;
	uint8_t pad[1];
};