#include <cuda_runtime.h>
#include "glm/glm.hpp"

namespace Common
{
	#define Pi 3.14159265358979323846f
	#define TwoPi 6.28318530717958647692f
	#define PiOver4 0.78539816339744830961f

	#define InvPi 0.31830988618379067154f
	#define Inv2Pi 0.15915494309189533577f
	#define Inv4Pi 0.07957747154594766788f
	#define Inv8Pi 0.03978873577f

	__host__ __device__ inline float AbsCosTheta(const glm::vec3* w)
	{
		return glm::abs(w->z);
	}

	__host__ __device__ inline bool SameHemisphere(const glm::vec3* w, const glm::vec3* wp)
	{
		return (w->z * wp->z > 0);
	}

	__host__ __device__ inline glm::vec3 Faceforward(const glm::vec3 &n, const glm::vec3 &v) {
		return (glm::dot(n, v) < 0.f) ? -n : n;
	}


	__host__ __device__ inline bool Refract(glm::vec3 wi, glm::vec3 n, float eta, glm::vec3 *wt) 
	{
		// Compute cos theta using Snell's law
		float cosThetaI = glm::dot(n, wi);
		float sin2ThetaI = glm::max(float(0), float(1 - cosThetaI * cosThetaI));
		float sin2ThetaT = eta * eta * sin2ThetaI;

		// Handle total internal reflection for transmission
		if (sin2ThetaT >= 1) return false;
		float cosThetaT = glm::sqrt(1 - sin2ThetaT);
		*wt = eta * -wi + (eta * cosThetaI - cosThetaT) * glm::vec3(n);
		return true;
	}

} // namespace Common end







