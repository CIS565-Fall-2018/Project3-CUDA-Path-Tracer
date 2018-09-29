#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"
#include "common.h"
#include "shapeFunctions.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__device__
bool sceneIntersect(const Ray & ray
                    , Geom * geoms
                    , int geoms_size
                    , ShadeableIntersection & intersection)
{
    intersection.t = -1;
    bool hit = false;
    bool overallHit = false;

    // naive parse through global geoms

    for (int i = 0; i < geoms_size; i++)
    {
        hit = false;
        ShadeableIntersection tmp_intersection;
        Geom & geom = geoms[i];

        if (geom.type == CUBE)
        {
            if (Cube::Intersect(ray, geom, &tmp_intersection)) {
                hit = true;
            }
        }
        else if (geom.type == SPHERE)
        {
            if (Sphere::Intersect(ray, geom, &tmp_intersection)) {
                hit = true;
            }
        }
        else if (geom.type == SQUAREPLANE)
        {
            if (SquarePlane::Intersect(ray, geom, &tmp_intersection)) {
                hit = true;
            }
        }
        // TODO: add more intersection tests here... triangle? metaball? CSG?

        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (hit) {
            if (tmp_intersection.t < intersection.t || intersection.t < 0)
            {
                intersection.t = tmp_intersection.t;
                intersection.materialId = geoms[i].materialid;
                intersection.geomId = i;
                intersection.surfaceNormal = tmp_intersection.surfaceNormal;
                intersection.tangent = tmp_intersection.tangent;
                intersection.bitangent = tmp_intersection.bitangent;
                intersection.point = tmp_intersection.point;
                overallHit = true;
            }
        }
    }
    return overallHit;
}
