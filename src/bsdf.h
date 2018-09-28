#pragma once
#include "glm/glm.hpp"
#include "common.h"
#include "sceneStructs.h"
#include "materialFunctions.h"
#include <thrust/random.h>

namespace BSDF {
    __host__ __device__
    Color3f Sample_f(const Vector3f &woW, Vector3f *wiW, float *pdf, const Material &mat, const ShadeableIntersection &isect, thrust::default_random_engine &rng)
    {
        /*
        if (BxDFsMatchingFlags(type) == 0) {
            return Color3f(0);
        }
        */
        thrust::uniform_real_distribution<float> u01(0, 1);

        glm::vec2 xi;
        xi.x = u01(rng);
        xi.y = u01(rng);

        unsigned int random = xi[0] * mat.numBxDFs;
        /*
        bool looped = false;
        while (!bxdfs[random]->MatchesFlags(type)) {
            random += 1;
            if (random >= numBxDFs && !looped) {
                random = 0;
                looped = true;
            }
            else {
                return Color3f(0);
            }
        }
        */

        xi[0] = u01(rng);

        glm::mat3 tangentToWorld = glm::mat3(isect.tangent, isect.bitangent, isect.surfaceNormal);
        glm::mat3 worldToTangent = glm::transpose(tangentToWorld);

        glm::vec3 wo = glm::normalize(worldToTangent * woW);
        glm::vec3 wi;

        Color3f sampledC = Color3f();

        switch (mat.bxdfs[random]) {
            case BxDFType::DIFFUSE:
                sampledC = Lambert::Sample_f(wo, &wi, xi, pdf, mat);
                break;
            case BxDFType::REFLECTIVE:
                sampledC = SpecularBRDF::Sample_f(wo, &wi, xi, pdf, mat);
                break;
            case BxDFType::REFRACTIVE:
                sampledC = SpecularBTDF::Sample_f(wo, &wi, xi, pdf, mat);
                break;
        }

        *wiW = glm::normalize(tangentToWorld * wi);

        /*
        if (!bxdfs[random]->MatchesFlags(BxDFType::BSDF_SPECULAR)) {
            for (unsigned int i = 0; i < numBxDFs; ++i) {
                if (bxdfs[i]->MatchesFlags(type)) {
                    if (bxdfs[i] != bxdfs[random]) {
                        *pdf += bxdfs[i]->Pdf(wo, wi);
                        sampledC += bxdfs[i]->f(wo, wi);
                    }
                }
            }
            *pdf = *pdf / (float) BxDFsMatchingFlags(type);
        }
        */

        *pdf = *pdf / (float) mat.numBxDFs;

        return sampledC;
    }

    __host__ __device__
    Color3f f(const Vector3f &woW, const Vector3f &wiW, Material & mat)
    {
        Color3f tempColor = Color3f(0.f);
        for (unsigned int i = 0; i < mat.numBxDFs; ++i) {

            switch (mat.bxdfs[i]) {
                case BxDFType::DIFFUSE:
                    tempColor += Lambert::f(mat.color);
                    break;
                case BxDFType::REFLECTIVE:
                    //tempColor += SpecularBRDF::f();  // Returns black
                    break;
                case BxDFType::REFRACTIVE:
                    //tempColor += SpecularBTDF::f();  // Returns black
                    break;
            }
        }
        return tempColor;
    }

}