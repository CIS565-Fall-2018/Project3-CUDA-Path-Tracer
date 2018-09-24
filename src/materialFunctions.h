#pragma once

#include "samplingFunctions.h"
#include "common.h"

/*
Functions for Lambert, reflective and refractive
*/

namespace Lambert {
    __host__ __device__
    glm::vec3 f(const glm::vec3 color) {
        return color * INVPI;
    }

    __host__ __device__
    float Pdf(const glm::vec3 &wo, const glm::vec3 &wi) {
        if (SameHemisphere(wo, wi)) {
            return INVPI * AbsCosTheta(wi);
        }
        return 0;
    }

    __host__ __device__
    glm::vec3 Sample_f(const glm::vec3 &wo, glm::vec3* wi, const glm::vec2 &sample, float* pdf, const glm::vec3 materialColor) {
        *wi = squareToHemisphereCosine(sample);
        if (wo.z < 0) wi->z *= -1;
        *wi = glm::normalize(*wi);
        *pdf = Pdf(wo, *wi);
        return f(materialColor);
    }
}
