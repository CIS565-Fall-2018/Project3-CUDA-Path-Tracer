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
    glm::vec3 Sample_f(const glm::vec3 &wo, glm::vec3* wi, const glm::vec2 &sample, float* pdf, const Color3f materialColor) {
        *wi = squareToHemisphereCosine(sample);
        if (wo.z < 0) wi->z *= -1;
        *wi = glm::normalize(*wi);
        *pdf = Pdf(wo, *wi);
        return f(materialColor);
    }
}

namespace SpecularBRDF {

    Color3f f(const Vector3f &wo, const Vector3f &wi)
    {
        return Color3f(0.f);
    }


    float Pdf(const Vector3f &wo, const Vector3f &wi)
    {
        return 0.f;
    }

    Color3f Sample_f(const Vector3f &wo, Vector3f *wi, const Point2f &sample, Float *pdf, const Color3f materialColor)
    {
        // Since our local surface normal is (0, 0, 1)
        // the reflection can be calculated by simply
        // negating the x and y
        *wi = Vector3f(-wo.x, -wo.y, wo.z);

        // Pdf is 1 for this wi
        *pdf = 1.f;

        // Calculate fresnel reflectance
        // Multiply by scaling factor, R
        // Divide by cos(wi) to cancel out Lambert term in LTE
        //return fresnel->Evaluate(CosTheta(*wi)) * materialColor / AbsCosTheta(*wi);

        return Color3f();
    }
}
