#include <cuda_runtime.h>
#include "glm/glm.hpp"

namespace Common
{
	const float Pi = 3.14159265358979323846;
	const float TwoPi = 6.28318530717958647692;
	const float PiOver4 = 0.78539816339744830961;

	const float InvPi = 0.31830988618379067154;
	const float Inv2Pi = 0.15915494309189533577;
	const float Inv4Pi = 0.07957747154594766788;
	const float Inv8Pi = 0.03978873577;

	__host__ __device__ inline float AbsCosTheta(const glm::vec3* w)
	{
		return glm::abs(w->z);
	}

	__host__ __device__ inline bool SameHemisphere(const glm::vec3* w, const glm::vec3* wp)
	{
		return (w->z * wp->z > 0);
	}

} // namespace Common end







