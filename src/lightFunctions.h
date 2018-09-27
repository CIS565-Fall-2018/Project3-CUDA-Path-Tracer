#pragma once

#include "common.h"
#include "utilities.h"
#include "sceneStructs.h"
#include "shapeFunctions.h"

namespace DiffuseAreaLight {
    __host__ __device__
    Color3f L(const ShadeableIntersection &isect, const glm::vec3 &wi, Material & mat)
    {
        return mat.color * mat.emittance;
        //bool twoSided = true;
        //return (twoSided || glm::dot(isect.surfaceNormal, w) > 0) ? mat.color * mat.emittance : Color3f(0.f);
    }

    __host__ __device__
    Color3f Sample_Li(
        const ShadeableIntersection &ref, const Point2f &xi, Vector3f *wi, Float *pdf, const Geom & shape, Material & mat) {

        ShadeableIntersection pShape;

        switch (shape.type) {
            case GeomType::SQUAREPLANE:
                //pShape = SquarePlane::Sample(xi, pdf, shape);

                *pdf = 1.f / (shape.scale.x * shape.scale.y);

                pShape.surfaceNormal = glm::normalize(glm::vec3(shape.invTranspose * (glm::vec4(0, 0, 1, 0))));
                pShape.point = glm::vec3(shape.transform * glm::vec4(xi.x - 0.5f, xi.y - 0.5f, 0, 1));

                Vector3f wi_temp = pShape.point - ref.point;
                wi_temp = glm::normalize(wi_temp);
                float angle = AbsDot(pShape.surfaceNormal, -wi_temp);
                if (glm::length(wi_temp) == 0 || angle == 0)
                    *pdf = 0;
                else {
                    // Convert from area measure, as returned by the Sample() call
                    // above, to solid angle measure.
                    *pdf *= glm::distance2(ref.point, pShape.point) / angle;
                    if (glm::isinf(*pdf)) *pdf = 0.f; 
                }

                if (*pdf == 0 || glm::distance2(ref.point, pShape.point) == 0) {
                    return Color3f(0);
                }

                *wi = glm::normalize(pShape.point - ref.point);
                return DiffuseAreaLight::L(pShape, -*wi, mat);


                break;

            // TODO other shapes
            
        }
        /*
        *wi = pShape.point - ref.point;
        float dist2 = glm::dot(*wi, *wi);
        //Vector3f wi = pShape.point - ref.point;
        wi = glm::normalize(wi);

        float angle = AbsDot(pShape.surfaceNormal, -*wi);
        if (glm::length(*wi) == 0 || angle == 0) {
            *pdf = 0;
        }
        else {
            // Convert from area measure, as returned by the Sample() call
            // above, to solid angle measure.
            *pdf *= dist2 / angle;
            if (glm::isinf(*pdf)) {
                *pdf = 0.f;
            }
        }
        */
        
        
        //pShape = Shape::Sample(ref, pShape, xi, pdf, shape, mat);
        
        /*
        if (*pdf < PDF_EPSILON || dist2 < EPSILON) {
            return Color3f(0);
        }
        *wi = glm::normalize(pShape.point - ref.point);
        //*wi = glm::normalize(*wi);
        return L(pShape, -*wi, mat);
        */
        return Color3f(0.f);
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