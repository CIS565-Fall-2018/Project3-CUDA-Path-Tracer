#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

namespace Shapes
{

#define RayEpsilon 0.00005f

	/**
	* Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
	*/
	glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v);



	Ray SpawnRay(const ShadeableIntersection* intersection, const glm::vec3* direction);

	bool SceneIntersect(const Ray* ray, const int geoms_size, Geom* allGeoms, ShadeableIntersection* intersection);

} // namespace Shapes end







