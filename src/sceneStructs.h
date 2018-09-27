#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
  SPHERE,
  CUBE,
  SQUAREPLANE,
  MESH,
  ACCELERATED_MESH
};

struct Ray
{
  glm::vec3 origin;
  glm::vec3 direction;
};

struct Bounds
{
  float xMin;
  float yMin;
  float zMin;

  float xMax;
  float yMax;
  float zMax;
};

struct KDNode
{
  int leftChildIdx;
  int rightChildIdx;

  int triStartIdx;
  int triEndIdx;

  glm::vec3 min;
  glm::vec3 max;
};

struct Geom
{
  enum GeomType type;
  int id;
  int materialid;
  int meshStartIndex;
  int numTriangles;
  int kdRootNodeIndex{-1};
  glm::vec3 translation;
  glm::vec3 rotation;
  glm::vec3 scale;
  glm::mat4 transform;
  glm::mat4 inverseTransform;
  glm::mat4 invTranspose;

  glm::vec3 min;
  glm::vec3 max;
  
};

enum MaterialType
{
  DIFFUSE,
  TRANSMISSIVE,
  SPECULAR,
  ROUGH_SPECULAR,
  ROUGH_DIFFUSE,
  ROUGH_TRANSMISSIVE,
  GLASS
};

struct Material
{
  glm::vec3 color;

  struct
  {
    float exponent;
    glm::vec3 color;
  } specular;

  float hasReflective;
  float hasRefractive;
  float indexOfRefraction;
  float emittance;
  float roughness;
  int diffuseMapId{-1};
  int bumpMapId{-1};
  int normalMapId{-1};
  int emissiveMapId{-1};
  MaterialType type;
};

struct Camera
{
  glm::ivec2 resolution;
  glm::vec3 position;
  glm::vec3 lookAt;
  glm::vec3 view;
  glm::vec3 up;
  glm::vec3 right;
  glm::vec2 fov;
  glm::vec2 pixelLength;
};

struct RenderState
{
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
  glm::vec3 throughput{1.0f};
  int pixelIndex;
  int remainingBounces;
  bool rayFromSpecular;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec2 uv;
  glm::vec3 intersectPoint;
  glm::vec3 surfaceNormal;
  glm::vec3 surfaceTangent;
  glm::vec3 surfaceBitangent;
  glm::mat3 tangentToWorld;
  glm::mat3 worldToTangent;
  int materialId;
  Geom* geom;
};
