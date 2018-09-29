#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum IntegratorType {
    NAIVE,
    DIRECT,
    FULL
};

enum GeomType {
    SPHERE,
    CUBE,
    SQUAREPLANE
};

enum BxDFType {
    DIFFUSE,
    REFLECTIVE,
    REFRACTIVE
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
    int id;
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
    BxDFType bxdfs[5];
    int numBxDFs;
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
    glm::vec3 throughput;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  glm::vec3 tangent;
  glm::vec3 bitangent;
  int materialId;
  glm::vec3 point;
  int geomId;
};

// https://stackoverflow.com/questions/37013191/is-it-possible-to-create-a-thrusts-function-predicate-for-structs-using-a-given
struct pathBounceZero {
    __host__ __device__
        bool operator()(const PathSegment &path) {
            return (path.remainingBounces > 0);
        }
};

struct intersectMaterialCompare {
    __host__ __device__
    bool operator()(const ShadeableIntersection &s1, const ShadeableIntersection &s2) {
        return (s1.materialId < s2.materialId);
    }
};