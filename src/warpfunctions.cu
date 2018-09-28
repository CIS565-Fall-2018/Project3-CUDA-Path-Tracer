#include <cuda_runtime.h>
#include "sceneStructs.h"
#include "common.cu"

namespace WarpFunctions
{
	
	__host__ __device__ glm::vec3 SquareToDiskUniform(const glm::vec2* sample)
	{
		const float radius = (*sample)[0];
		glm::vec3 result;
		const float angle = (*sample)[1] * 2.f * Common::Pi;
		result[0] = radius * cos(angle);
		result[1] = radius * sin(angle);
		result[2] = 0;
		return result;
	}

	__host__ __device__ glm::vec3 SquareToDiskConcentric(const glm::vec2* sample)
	{
		glm::vec3 result;
		float phi, r;
		float a = 2.f * (*sample)[0] - 1;
		float b = 1 - 2.f * (*sample)[1];

		if (a > -b) {
			if (a > b) {
				r = a;
				phi = (Common::PiOver4) * (b / a);
			}
			else {
				r = b;
				phi = (Common::PiOver4) * (2.f - (a / b));
			}
		}
		else {
			if (a < b) {
				r = -a;
				phi = (Common::PiOver4) * (4.f + (b / a));
			}
			else {
				r = -b;
				if (b != 0) {
					phi = (Common::PiOver4) * (6.f - (a / b));
				}
				else {
					phi = 0;
				}
			}
		}
		result[0] = r * cos(phi);
		result[1] = r * sin(phi);
		result[2] = 0;
		return result;
	}

	__host__ __device__ float SquareToDiskPDF(const glm::vec3* sample)
	{
		return Common::InvPi;
	}

	__host__ __device__ glm::vec3 SquareToSphereUniform(const glm::vec2* sample)
	{
		const float phi = (*sample)[1] * Common::Pi;
		const float theta = (*sample)[0] * Common::TwoPi;

		glm::vec3 result;
		const float sinPhi = sin(phi);
		result[0] = cos(theta) * sinPhi;
		result[1] = cos(phi);
		result[2] = sin(theta) * sinPhi;
		return result;
	}

	__host__ __device__ float SquareToSphereUniformPDF(const glm::vec3* sample)
	{
		return (Common::Inv4Pi);
	}

	__host__ __device__ glm::vec3 SquareToSphereCapUniform(const glm::vec2* sample, float thetaMin)
	{
		const float phi = (*sample)[1] * thetaMin * Common::Pi / 180.f;
		const float theta = (*sample)[0] * 2.f * Common::Pi;

		glm::vec3 result;
		result[0] = cos(theta) * sin(phi);
		result[1] = cos(phi);
		result[2] = sin(theta) * sin(phi);
		return result;
	}

	__host__ __device__ float SquareToSphereCapUniformPDF(const glm::vec2* sample, float thetaMin)
	{
		return (Common::Inv2Pi) / (1.f - cos(glm::radians(180.f - thetaMin)));
	}

	__host__ __device__ glm::vec3 SquareToHemisphereUniform(const glm::vec2* sample)
	{
		const float z = (*sample)[0];
		const float r = glm::sqrt(glm::max(0.f, 1.f - z * z));
		const float phi = 2 * 3.14159265 * (*sample)[1];
		return glm::vec3(r * std::cos(phi), r * std::sin(phi), z);
	}

	__host__ __device__ float SquareToHemisphereUniformPDF(const glm::vec2* sample)
	{
		return (Common::Inv2Pi);
	}

	__host__ __device__ glm::vec3 SquareToHemisphereCosine(const glm::vec2* sample)
	{
		const glm::vec3 d = SquareToDiskConcentric(sample);
		const float z = glm::sqrt(glm::max(0.f, 1.f - d.x * d.x - d.y * d.y));
		return glm::vec3(d.x, d.y, z);
	}

	__host__ __device__ float SquareToHemisphereCosinePDF(const glm::vec2* sample)
	{
		return (Common::Inv8Pi);
	}

} // namespace WarpFunctions end







