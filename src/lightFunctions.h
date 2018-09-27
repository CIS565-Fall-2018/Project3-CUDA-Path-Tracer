#pragma once

#include "common.h"
#include "utilities.h"
#include "sceneStructs.h"
#include "shapeFunctions.h"

namespace DiffuseAreaLight {
    __host__ __device__
    Color3f L(const ShadeableIntersection &isect, const glm::vec3 &w, Material & mat)
    {
        bool twoSided = true;
        return (twoSided || glm::dot(isect.surfaceNormal, w) > 0) ? mat.color * mat.emittance : Color3f(0.f);
    }

    __host__ __device__
    Color3f Sample_Li(
        const ShadeableIntersection &ref, const Point2f &xi, Vector3f *wi, Float *pdf, const Geom & shape, Material & mat) {

        ShadeableIntersection pShape;
        switch (shape.type) {
            case GeomType::SQUAREPLANE:
                pShape = SquarePlane::Sample(xi, pdf, shape);
                break;

            // TODO other shapes
            
        }
        
        if (*pdf < PDF_EPSILON || glm::distance(ref.point, pShape.point) < EPSILON) {
            return Color3f(0);
        }
        *wi = glm::normalize(pShape.point - ref.point);
        return L(pShape, -*wi, mat);
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
        float dist = glm::distance(ref.point, isectLight.point);
        Float pdf = (dist * dist) / (AbsDot(isectLight.surfaceNormal, -wi) * area);
        if (glm::isinf(pdf)) pdf = 0.f;
        return pdf;
    }

}