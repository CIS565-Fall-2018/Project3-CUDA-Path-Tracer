#pragma once

typedef float Float;
typedef glm::vec3 Color3f;
typedef glm::vec3 Point3f;
typedef glm::vec3 Normal3f;
typedef glm::vec2 Point2f;
typedef glm::ivec2 Point2i;
typedef glm::ivec3 Point3i;
typedef glm::vec3 Vector3f;
typedef glm::vec2 Vector2f;
typedef glm::ivec2 Vector2i;
typedef glm::mat4 Matrix4x4;
typedef glm::mat3 Matrix3x3;

__host__ __device__
bool SameHemisphere(const glm::vec3& wo, const glm::vec3& wi) {
    return wo.z * wi.z > 0;
    }

__host__ __device__
float AbsCosTheta(const glm::vec3 &w) {
    return std::abs(w.z);
    }

__host__ __device__
float AbsDot(const glm::vec3& a, const glm::vec3& b) {
    return glm::abs(glm::dot(a, b));
}

__host__ __device__
Ray transformRay(const Ray &r, const glm::mat4 &T) {
    glm::vec4 o = glm::vec4(r.origin, 1);
    glm::vec4 d = glm::vec4(r.direction, 0);

    o = T * o;
    d = T * d;

    Ray ray;
    ray.origin = glm::vec3(o);
    ray.direction = glm::vec3(d);
    return ray;
}