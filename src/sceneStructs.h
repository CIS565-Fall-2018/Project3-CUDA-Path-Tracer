#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))
#define MAX_RECORD_DEPTH 8

enum GeomType {
	SQUARE,//facing positive z
	SPHERE,
	CUBE,
	MESH
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
	int triangleStartIndex;//for debug, will be deleted
	int triangleCount;//for debug, will be deleted
	int root;
	int id;//for shadow feeler
};

struct Triangle {
	Triangle(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2)
	{
		v[0] = p0;
		v[1] = p1;
		v[2] = p2;
		leftIndex = -1;
		rightIndex = -1;
	}
	glm::vec3 v[3];
	glm::vec2 t[3];
	glm::vec3 n[3];
	glm::vec3 min;//for BVH
	glm::vec3 max;//for BVH
	glm::vec3 center;//for kdtree
	glm::vec3 planeNormal;
	int leftIndex;//left child
	int rightIndex;//right child
};

struct Material {
	glm::vec3 color;
	struct {
		glm::vec3 color;
		float indexOfRefraction;
	} specularReflective;
	struct {
		glm::vec3 color;
		float indexOfRefraction;
	} specularTransmissive;
	struct {
		glm::vec3 color;
		float emittance;
	} emissive;
	uint8_t type;//0-diffuse, 1-emmisive, 2-specular reflective, 3-specular transmissive
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
	//PathSegment()
	//{
	//	ray.origin = glm::vec3(0, 0, 0);
	//	ray.direction = glm::vec3(0, 0, 0);
	//	color = glm::vec3(0, 0, 0);
	//	pixelIndex = 0;
	//	remainingBounces = 0;
	//	for (int i = 0; i < MAX_RECORD_DEPTH; i++)
	//	{
	//		geomIds[i] = 0;
	//		outsides[i] = false;
	//	}
	//	hitLight = false;
	//}
	////////
	Ray ray;
	glm::vec3 color;//when using multi-importance sampling, initiate this to (0, 0, 0)
	int pixelIndex;
	int remainingBounces;
	//My code here
	int geomIds[MAX_RECORD_DEPTH];//for debug
	bool outsides[MAX_RECORD_DEPTH];//for debug
	bool hitLight;//for naive integrator
	glm::vec3 throughput;//for multi-importance sampling
	bool hitSpecular;//for full light integrator
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
	ShadeableIntersection()
	{
		t = -1;
		surfaceNormal = glm::vec3(0, 0, 0);
		materialId = -1;
		geomId = -1;
		point = glm::vec3(0, 0, 0);
		uv = glm::vec2(0, 0);
		surfaceTangent = glm::vec3(0, 0, 0);
		surfaceBitangent = glm::vec3(0, 0, 0);
		outside = false;
	}
	float t;
	glm::vec3 surfaceNormal;
	int materialId;
	//My code here.
	int geomId;
	glm::vec3 point;
	glm::vec2 uv;
	glm::vec3 surfaceTangent;
	glm::vec3 surfaceBitangent;
	bool outside;
};
