#pragma once

#include "samplingFunctions.h"
#include "common.h"

/*
Functions for Lambert, reflective and refractive
*/

__host__ __device__
Color3f FresnelNoOp() {
    return Color3f(1.f);
}

__host__ __device__
Color3f FresnelNoReflect() {
    return Color3f(0.f);
}

__host__ __device__
Color3f Fresnel(float cosThetaI, float etaI) {
    // Make sure cosThetaI is within legal values
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    // Check if we are entering or exiting the material
    // We need to swap eta is we are exiting
    float temp_etaT = 1.0f; // etaT;
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        float t = temp_etaT;
        temp_etaT = etaI;
        etaI = t;
        //std::swap(etaI, temp_etaT);
        cosThetaI = glm::abs(cosThetaI);
    }

    // Calculate sinTheta
    // sinThetaI uses sin^2 = 1 - cos^2
    // since we have cosThetaI
    float max = glm::max(0.f, 1.f - (cosThetaI * cosThetaI));
    float sinThetaI = glm::sqrt(max);
    // sinThetaT uses Snell's law because we have etaI, etaT, and now sinThetaI
    float sinThetaT = etaI / temp_etaT * sinThetaI;

    // If sinThetaT is greater than one,
    // total internal reflection
    // In terms of math, it is physically impossible
    if (sinThetaT >= 1) {
        return Color3f(1.f);
    }

    // Find cosThetaT using cos^2 = 1 - sin^2
    float cosThetaT = glm::sqrt(glm::max((float) 0, 1 - sinThetaT * sinThetaT));

    // Use fresnel dielectric equations to get final value
    float Rparl = ((temp_etaT * cosThetaI) - (etaI * cosThetaT)) / ((temp_etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (temp_etaT * cosThetaT)) / ((etaI * cosThetaI) + (temp_etaT * cosThetaT));
    return Color3f((Rparl * Rparl + Rperp * Rperp) / 2.f);
}

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
    glm::vec3 Sample_f(const glm::vec3 &wo, glm::vec3* wi, const glm::vec2 &sample, float* pdf, const Material &mat) {
        *wi = squareToHemisphereCosine(sample);
        if (wo.z < 0) wi->z *= -1;
        *wi = glm::normalize(*wi);
        *pdf = Pdf(wo, *wi);
        return f(mat.color);
    }
}

namespace SpecularBRDF {
    __host__ __device__
    Color3f f(const Vector3f &wo, const Vector3f &wi)
    {
        return Color3f(0.f);
    }

    __host__ __device__
    float Pdf(const Vector3f &wo, const Vector3f &wi)
    {
        return 0.f;
    }

    __host__ __device__
    Color3f Sample_f(const Vector3f &wo, Vector3f *wi, const Point2f &sample, Float *pdf, const Material &mat)
    {
        // Since our local surface normal is (0, 0, 1)
        // the reflection can be calculated by simply
        // negating the x and y
        //*wi = Vector3f(-wo.x, -wo.y, wo.z);
        (*wi).x = -wo.x;
        (*wi).y = -wo.y;
        (*wi).z = wo.z;

        // Pdf is 1 for this wi
        *pdf = 1.f;

        Color3f fresnel;
        if (mat.numBxDFs == 1) {
            fresnel = FresnelNoOp();
        }
        else {
            fresnel = Fresnel(CosTheta(*wi), mat.indexOfRefraction);
        }

        // Calculate fresnel reflectance
        // Multiply by scaling factor, R
        // Divide by cos(wi) to cancel out Lambert term in LTE
        return fresnel * mat.color / AbsCosTheta(*wi);
    }
}

namespace SpecularBTDF {
    __host__ __device__
    Color3f f(const Vector3f &wo, const Vector3f &wi)
    {
        return Color3f(0.f);
    }

    __host__ __device__
    float Pdf(const Vector3f &wo, const Vector3f &wi)
    {
        return 0.f;
    }

    __host__ __device__
    Color3f Sample_f(const Vector3f &wo, Vector3f *wi, const Point2f &sample, Float *pdf, const Material &mat)
    {
        // Calculate eta
        float eta = 1.0f / mat.indexOfRefraction; // etaA is 1.0f for air

        // If we are exiting material,
        // eta is reciprocal
        if (CosTheta(wo) <= 0.f) {
            eta = 1.f / eta;
        }

        // Calculate refraction ray in wi
        // If there is no refraction (aka total internal reflection),
        // return 0
        if (!Refract(wo, Faceforward(Normal3f(0, 0, 1), wo), eta, wi)) {
            return Color3f(0.f);
        }

        // Pdf of this wi is 1
        *pdf = 1.f;

        // (1 - fresnel reflectance) because this is refraction
        // The light transmitted is the portion not reflected
        // Multiply that by scaling factor, T
        // Divide by cos(wi) to cancel out Lambert in LTE
        Color3f fresnel;
        if (mat.numBxDFs == 1) {
            fresnel = FresnelNoReflect();
        }
        else {
            fresnel = Fresnel(CosTheta(*wi), mat.indexOfRefraction);
        }
        Color3f ft = mat.specular.color * (Color3f(1.f) - fresnel);
        return ft / AbsCosTheta(*wi);
    }
}
