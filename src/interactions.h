#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0,1);
	float rand = u01(rng);
	if (m.emittance > 0)
	{
		pathSegment.color *= m.color*m.emittance;
		pathSegment.remainingBounces = 0;
		return;
	}
	else if (m.hasReflective>0&&m.diffuse>0)
	{
		float diffmag = 0.8;
		if (rand<=diffmag)
		{
			pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
			pathSegment.color *= m.specular.color;
		}
		else
		{
			pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
			pathSegment.color *= m.specular.color;
		}
		pathSegment.ray.origin = intersect + 0.001f*normal;
		pathSegment.remainingBounces--;
	}
	else if (m.hasReflective > 0)
	{
		float lambertVal = fabs(glm::dot(normal, (pathSegment.ray.direction)));
		pathSegment.remainingBounces--;
		pathSegment.ray.origin = intersect+0.001f*normal;
		pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
		pathSegment.color *= m.specular.color;
		//pathSegment.color *= lambertVal*m.color;
	}
	//Schlick's approximation
	else if (rand > 1 - m.hasRefractive && rand < 1) 
	{ 
		float epsilon = 1e-7;
		float actualidx;
		float rf_c = 0.2f;
		glm::vec3 actualcol = m.color;
		actualidx = m.indexOfRefraction;
		actualidx += rf_c*(1-(pathSegment.ray.wavelength));
		float rlerp = abs(pathSegment.ray.wavelength - 0.86f);
		float glerp = abs(pathSegment.ray.wavelength - 0.5f);
		float blerp = abs(pathSegment.ray.wavelength - 0.18f);

		//rlerp = 1.f - rlerp;
		//glerp = 1.f - glerp;
		//blerp = 1.f - blerp;

		rlerp = 1 / rlerp;
		glerp = 1 / glerp;
		blerp = 1 / blerp;

		glm::vec3 idxlst(rlerp, glerp, blerp);

		int largeidx = 0;
		if (rlerp > glerp)
			largeidx = 0;
		else
			largeidx = 1;
		if (idxlst[largeidx] > blerp)
			largeidx = largeidx;
		else
			largeidx = 2;

		//rlerp *= rlerp;
		//glerp *= glerp;
		//blerp *= blerp;



		float N = 200.f;
		glm::vec3 Albedored(1, 0, 0); glm::vec3 Albedogreen(0, 1, 0); glm::vec3 Albedoblue(0, 0, 1);
		glm::vec3 dispersionAlbedo = (1 / N)*(rlerp*Albedored + glerp*Albedogreen + blerp*Albedoblue);


		if (rand < 0.5)
			actualcol = 1.2f*glm::vec3(largeidx == 0 ? 1 : 0, largeidx == 1 ? 1 : 0, largeidx == 2 ? 1 : 0);
		else
			actualcol *=  1.1f;

		//glm::clamp(actualcol, glm::vec3(0), glm::vec3(1));

		float indexRatio;
		float theta = (180.0f / PI) * acos(glm::dot(pathSegment.ray.direction, normal) /*/ (glm::length(pathSegment.ray.direction) * glm::length(normal))*/);
		bool in = true;
		if (theta >= 90.0f) {
			
			indexRatio = 1.f / actualidx /*m.indexOfRefraction*/;
		}
		else
		{
			//in = true;
			indexRatio = actualidx/*m.indexOfRefraction*/;
		}
		float R0in = (1 - actualidx) / (1 + actualidx) * (1 - actualidx) / (1 + actualidx);
		float RSchlickin = R0in + (1.0f - R0in) * glm::pow(1.0f - glm::abs(glm::dot(normal, pathSegment.ray.direction)), 5);

		float R0 = (1 - indexRatio) / (1 + indexRatio) * (1 - indexRatio) / (1 + indexRatio);
		float RSchlick = R0 + (1.0f - R0) * glm::pow(1.0f - glm::abs(glm::dot(normal, pathSegment.ray.direction)), 5);
		if (in)
		{
			if (RSchlick < rand)
			{
				pathSegment.ray.direction = glm::normalize(glm::refract(pathSegment.ray.direction, normal, indexRatio));
				pathSegment.color *= actualcol*m.color;/*m.color * m.specular.color*/
			}
			else
			{
				pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
				pathSegment.color *= 1.f*m.color;
			}
		}
		else
		{
			if (RSchlickin < rand)
			{
				pathSegment.ray.direction = glm::normalize(glm::refract(pathSegment.ray.direction, normal, indexRatio));
				pathSegment.color *= actualcol*m.color;/*m.color * m.specular.color*/
			}
			else
			{
				pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, -normal));
				pathSegment.color *= 1.f*m.color;
			}
		}
		pathSegment.ray.origin = intersect + 1e-3f * (glm::normalize(pathSegment.ray.direction));
		
		pathSegment.remainingBounces--;
	}

	else 
	{
		pathSegment.remainingBounces--;
		pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.ray.origin = intersect+0.001f*normal;
		pathSegment.color *= m.color;
	}
}
