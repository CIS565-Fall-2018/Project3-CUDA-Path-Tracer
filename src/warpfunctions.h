#pragma once

namespace WarpFunctions
{

	glm::vec3 SquareToDiskUniform(const glm::vec2* sample);

	glm::vec3 SquareToDiskConcentric(const glm::vec2* sample);

	float SquareToDiskPDF(const glm::vec3* sample);

	glm::vec3 SquareToSphereUniform(const glm::vec2* sample);

	float SquareToSphereUniformPDF(const glm::vec3* sample);

	glm::vec3 SquareToSphereCapUniform(const glm::vec2* sample, float thetaMin);

	float SquareToSphereCapUniformPDF(const glm::vec2* sample, float thetaMin);

	glm::vec3 SquareToHemisphereUniform(const glm::vec2* sample);

	float SquareToHemisphereUniformPDF(const glm::vec2* sample);

	glm::vec3 SquareToHemisphereCosine(const glm::vec2* sample);

	float SquareToHemisphereCosinePDF(const glm::vec2* sample);

} // namespace WarpFunctions end







