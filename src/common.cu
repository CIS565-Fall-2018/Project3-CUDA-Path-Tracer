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

} // namespace Common end







