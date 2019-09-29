#pragma once

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

__device__ const float Pi = 3.14159265358979323846;
__device__ const float TwoPi = 6.28318530717958647692;
__device__ const float InvPi = 0.31830988618379067154;
__device__ const float Inv2Pi = 0.15915494309189533577;
__device__ const float Inv4Pi = 0.07957747154594766788;
__device__ const float PiOver2 = 1.57079632679489661923;
__device__ const float PiOver4 = 0.78539816339744830961;
__device__ const float Sqrt2 = 1.41421356237309504880;
__device__ const float RayEpsilon = 0.001f;
__device__ const float FloatEpsilon = 0.000002f;

__host__ __device__ inline float AbsDot(glm::vec3 a, glm::vec3 b) {
	return glm::abs(glm::dot(a, b));
}

__host__ __device__ inline bool IsBlack(glm::vec3 a) {
	return a.x == 0.0f && a.y == 0.0f && a.z == 0.0f;
}

__host__ __device__ inline bool SameHemisphere(glm::vec3 w, glm::vec3 wp, glm::vec3 n) {
	return glm::dot(w, n) > 0 && glm::dot(wp, n) > 0;
}

__host__ __device__ inline float AbsCosTheta(glm::vec3 w, glm::vec3 n) {
	return glm::abs(glm::dot(w, n));
}

__host__ __device__ inline bool fequal(float a, float b) {
	return std::abs(a - b) < FloatEpsilon;
}

__host__ __device__ glm::vec2 ConcentricSampleDisk(glm::vec2 u) {
	glm::vec2 uOffset = 2.0f * u - glm::vec2(1.0f, 1.0f);
	if (uOffset.x == 0 && uOffset.y == 0) {
		return glm::vec2(0.0f, 0.0f);
	}
	float theta, r;
	if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
		r = uOffset.x;
		theta = PiOver4 * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
	}
	return r * glm::vec2(std::cos(theta), std::sin(theta));
}

__host__ __device__ glm::vec3 squareToSphereUniform(glm::vec2 sample) {
	float r = glm::sqrt(sample.x);
	float phi = sample.y * 2 * Pi;
	float u = r * glm::cos(phi);
	float v = r * glm::sin(phi);
	float x = u * glm::sqrt(1 - glm::pow(r, 2.0f)) * 2;
	float y = v * glm::sqrt(1 - glm::pow(r, 2.0f)) * 2;
	return glm::vec3(x, y, 1 - 2 * glm::pow(r, 2.0f));
}