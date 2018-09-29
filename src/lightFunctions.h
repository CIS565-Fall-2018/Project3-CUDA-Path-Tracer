#pragma once

#include "common.h"
#include "utilities.h"
#include "sceneStructs.h"
#include "shapeFunctions.h"

namespace DiffuseAreaLight {
    __host__ __device__
    Color3f L(const ShadeableIntersection &isect, const glm::vec3 &w, const Material & mat)
    {
        return mat.color * mat.emittance;
        //bool twoSided = true;
        //return (twoSided || glm::dot(isect.surfaceNormal, w) > 0) ? mat.color * mat.emittance : Color3f(0.f);
    }

    __host__ __device__
    Color3f Sample_Li(
        const ShadeableIntersection &ref, const Point2f &xi, Vector3f *wi, Float *pdf, const Geom & shape, const Material & mat) {

        ShadeableIntersection pShape;

        Shape::Sample(ref, pShape, xi, pdf, shape, mat, wi);

        if (*pdf == 0 || glm::distance2(ref.point, pShape.point) == 0) {
            return Color3f(0);
        }

        return DiffuseAreaLight::L(pShape, -*wi, mat);

    }

    __host__ __device__
    float Pdf_Li(const ShadeableIntersection &ref, const Vector3f &wi, const Geom & shape) {
        Ray ray;
        ray.origin = ref.point + wi * RAY_EPSILON;
        ray.direction = wi;

        ShadeableIntersection isectLight;
        // Ignore any alpha textures used for trimming the shape when performing
        // this intersection. Hack for the "San Miguel" scene, where this is used
        // to make an invisible area light.
        //if (!shape->Intersect(ray, &isectLight)) return 0;
        bool hit;
        float area;
        switch (shape.type) {
            case GeomType::SQUAREPLANE:
                hit = SquarePlane::Intersect(ray, shape, &isectLight);
                area = SquarePlane::Area(shape);
                break;

                // TODO other shapes

        }

        if (!hit) {
            return 0;
        }
        // Convert light sample weight to solid angle measure
        float dist2 = glm::distance2(ref.point, isectLight.point);
        Float pdf = (dist2) / (AbsDot(isectLight.surfaceNormal, -wi) * area);
        if (glm::isinf(pdf)) pdf = 0.f;
        return pdf;
    }

}

__device__
Color3f DirectLightingSample(
    const ShadeableIntersection & intersection
    , const PathSegment & pathSegment
    , const Material & isectMaterial
    , Material * materials
    , Geom * lights
    , int numLights
    , Geom * geoms
    , int geoms_size
    , thrust::default_random_engine &rng)
{

    Color3f Ld = Color3f(0.f);

    Vector3f woW = -pathSegment.ray.direction;

    thrust::uniform_real_distribution<float> u01(0, 1);

    /***********************************************************
    * Sample a random light
    ***********************************************************/

    int lightIndex = glm::min((int) (u01(rng) * numLights), numLights - 1);

    Vector3f wiW_light;
    float pdf_light;
    float pdf_bsdf;

    glm::vec2 sample;
    sample.x = u01(rng);
    sample.y = u01(rng);

    Color3f li_light = DiffuseAreaLight::Sample_Li(intersection, sample, &wiW_light, &pdf_light, lights[lightIndex], materials[lights[lightIndex].materialid]);

    pdf_light /= (float) numLights;
    if (pdf_light > PDF_EPSILON) {

        ShadeableIntersection visTest;
        Ray shadowFeeler;
        shadowFeeler.origin = GetNewRayOrigin(wiW_light, intersection.surfaceNormal, intersection.point);
        shadowFeeler.direction = wiW_light;
        if (sceneIntersect(shadowFeeler, geoms, geoms_size, visTest)) {
            if (visTest.geomId == lights[lightIndex].id) {

                // Calculate MIS term for light sample
                Color3f f_light = BSDF::f(woW, wiW_light, isectMaterial);
                float dot_light = AbsDot(wiW_light, glm::normalize(intersection.surfaceNormal));
                pdf_bsdf = BSDF::Pdf(woW, wiW_light, isectMaterial, intersection);

                float weight = PowerHeuristic(1, pdf_light, 1, pdf_bsdf);
                //float weight = BalanceHeuristic(1, pdf_light, 1, pdf_bsdf);
                Ld = Ld + f_light * dot_light * li_light * (weight / pdf_light);
            }
        }
    }


    /*******************************************************
    * Sample using BSDF
    *******************************************************/
    sample.x = u01(rng);
    sample.y = u01(rng);

    Vector3f wiW_bsdf;
    Color3f f_bsdf = BSDF::Sample_f(woW, &wiW_bsdf, &pdf_bsdf, isectMaterial, intersection, rng);
    float dot_bsdf = glm::abs(glm::dot(wiW_bsdf, intersection.surfaceNormal));

    if (pdf_bsdf > PDF_EPSILON) {
        // Check if light is visible from point
        ShadeableIntersection visTest;
        Ray shadowFeeler;
        shadowFeeler.origin = GetNewRayOrigin(wiW_bsdf, intersection.surfaceNormal, intersection.point);
        shadowFeeler.direction = wiW_bsdf;

        if (sceneIntersect(shadowFeeler, geoms, geoms_size, visTest)) {
            if (geoms[visTest.geomId].id == lights[lightIndex].id) {
                // Calculate MIS term for BSDF sampling
                Color3f li_bsdf = DiffuseAreaLight::L(visTest, wiW_bsdf, materials[lights[lightIndex].materialid]);

                pdf_light = DiffuseAreaLight::Pdf_Li(intersection, wiW_bsdf, lights[lightIndex]);
                float weight = PowerHeuristic(1, pdf_bsdf, 1, pdf_light);
                //float weight = BalanceHeuristic(1, pdf_bsdf, 1, pdf_light);

                Ld = Ld + f_bsdf * dot_bsdf * li_bsdf * (weight / pdf_bsdf);
                //Ld = Ld + f_bsdf * dot_bsdf * li_bsdf * (1.f / pdf_bsdf);
            }
        }
    }

    return Ld;

}