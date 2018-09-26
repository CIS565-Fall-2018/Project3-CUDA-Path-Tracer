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
float CosTheta(const Vector3f &w) { 
    return w.z; 
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

__host__ __device__
Normal3f Faceforward(const Normal3f &n, const Vector3f &v) {
    return (glm::dot(n, v) < 0.f) ? -n : n;
}

__host__ __device__
bool Refract(const Vector3f &wi, const Normal3f &n, float eta,
             Vector3f *wt) {
    // Compute cos theta using Snell's law
    float cosThetaI = glm::dot(n, wi);
    float sin2ThetaI = std::max(float(0), float(1 - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return false;
    float cosThetaT = std::sqrt(1 - sin2ThetaT);
    *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * Vector3f(n);
    return true;
    }
