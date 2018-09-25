#pragma once
#include "glm/glm.hpp"
#include "common.h"
#include "sceneStructs.h"

namespace Cube {

    // glm::sign gives compile error
    // so rewrite the function
    __host__ __device__ float sign(const float &i) {
        if (i < 0) {
            return -1.0f;
        }
        else if (i > 0) {
            return 1.0f;
        }
        return 0.0f;
    }

    __host__ __device__
    int GetFaceIndex(const Point3f& P)
    {
        int idx = 0;
        float val = -1;
        for (int i = 0; i < 3; i++) {
            if (glm::abs(P[i]) > val) {
                idx = i * sign(P[i]); // glm::sign gives compile error
                val = glm::abs(P[i]);
            }
        }
        return idx;
    }

    __host__ __device__
    Normal3f GetCubeNormal(const Point3f& P)
    {
        int idx = glm::abs(GetFaceIndex(Point3f(P)));
        Normal3f N(0, 0, 0);
        N[idx] = sign(P[idx]); // glm::sign gives compile error
        return N;
    }

    __host__ __device__
        void ComputeTBN(const Point3f& P, Normal3f* nor, Vector3f* tan, Vector3f* bit, const Geom &box)
        {
        Normal3f localNormal = GetCubeNormal(P);
        *nor = glm::normalize(glm::mat3(box.inverseTransform) * localNormal);

        Vector3f localTan, localBit;

        if (localNormal.x < 0) {
            localBit.y = 1;
            localTan.z = 1;
            }
        else if (localNormal.x > 0) {
            localBit.y = 1;
            localTan.z = -1;
            }
        else if (localNormal.y < 0) {
            localBit.z = 1;
            localTan.x = 1;
            }
        else if (localNormal.y > 0) {
            localBit.z = -1;
            localTan.x = 1;
            }
        else if (localNormal.z < 0) {
            localBit.y = 1;
            localTan.x = -1;
            }
        else if (localNormal.z > 0) {
            localBit.y = 1;
            localTan.x = 1;
            }

        *tan = glm::normalize(glm::mat3(box.transform) * localTan);
        *bit = glm::normalize(glm::mat3(box.transform) * localBit);
        }

    __host__ __device__
    bool Intersect(const Ray& r, const Geom &box , ShadeableIntersection* isect)
    {
        //Transform the ray
        Ray r_loc = transformRay(r, box.inverseTransform);

        float t_n = -FLT_MAX;
        float t_f = FLT_MAX;
        for (int i = 0; i < 3; i++) {
            //Ray parallel to slab check
            if (r_loc.direction[i] == 0) {
                if (r_loc.origin[i] < -0.5f || r_loc.origin[i] > 0.5f) {
                    return false;
                }
            }
            //If not parallel, do slab intersect check
            float t0 = (-0.5f - r_loc.origin[i]) / r_loc.direction[i];
            float t1 = (0.5f - r_loc.origin[i]) / r_loc.direction[i];
            if (t0 > t1) {
                float temp = t1;
                t1 = t0;
                t0 = temp;
            }
            if (t0 > t_n) {
                t_n = t0;
            }
            if (t1 < t_f) {
                t_f = t1;
            }
        }
        if (t_n < t_f)
            {
            float t = t_n > 0 ? t_n : t_f;
            if (t < 0) {
                return false;
            }
            //Lastly, transform the point found in object space by T
            glm::vec4 P = glm::vec4(r_loc.origin + t*r_loc.direction, 1);
            //InitializeIntersection(isect, t, Point3f(P));
            ComputeTBN(Point3f(P), &(isect->surfaceNormal), &(isect->tangent), &(isect->bitangent), box);
            isect->t = t;
            isect->point = Point3f(box.transform * P);
            return true;
        }
        else {//If t_near was greater than t_far, we did not hit the cube
            return false;
        }
    }

    

    glm::vec2 GetUVCoordinates(const glm::vec3 &point) {
        glm::vec3 abs = glm::min(glm::abs(point), 0.5f);
        glm::vec2 UV;//Always offset lower-left corner
        if (abs.x > abs.y && abs.x > abs.z)
        {
            UV = glm::vec2(point.z + 0.5f, point.y + 0.5f) / 3.0f;
            //Left face
            if (point.x < 0)
            {
                UV += glm::vec2(0, 0.333f);
            }
            else
            {
                UV += glm::vec2(0, 0.667f);
            }
        }
        else if (abs.y > abs.x && abs.y > abs.z)
        {
            UV = glm::vec2(point.x + 0.5f, point.z + 0.5f) / 3.0f;
            //Left face
            if (point.y < 0)
            {
                UV += glm::vec2(0.333f, 0.333f);
            }
            else
            {
                UV += glm::vec2(0.333f, 0.667f);
            }
        }
        else
        {
            UV = glm::vec2(point.x + 0.5f, point.y + 0.5f) / 3.0f;
            //Left face
            if (point.z < 0)
            {
                UV += glm::vec2(0.667f, 0.333f);
            }
            else
            {
                UV += glm::vec2(0.667f, 0.667f);
            }
        }
        return UV;
    }
}

namespace Sphere {
    __host__ __device__
    void ComputeTBN(const Point3f& P, Normal3f* nor, Vector3f* tan, Vector3f* bit, const Geom &sphere)
    {
        *nor = glm::normalize(glm::mat3(sphere.invTranspose) * glm::normalize(P));
        *tan = glm::normalize(glm::mat3(sphere.transform) * glm::cross(Vector3f(0, 1, 0), (glm::normalize(P))));
        *bit = glm::normalize(glm::cross(*nor, *tan));
    }

    __host__ __device__
    bool Intersect(const Ray &ray, const Geom &sphere, ShadeableIntersection* isect)
    {
        //Transform the ray
        Ray r_loc = transformRay(ray, sphere.inverseTransform);

        float A = pow(r_loc.direction.x, 2.f) + pow(r_loc.direction.y, 2.f) + pow(r_loc.direction.z, 2.f);
        float B = 2 * (r_loc.direction.x*r_loc.origin.x + r_loc.direction.y * r_loc.origin.y + r_loc.direction.z * r_loc.origin.z);
        float C = pow(r_loc.origin.x, 2.f) + pow(r_loc.origin.y, 2.f) + pow(r_loc.origin.z, 2.f) - 1.f;//Radius is 1.f
        float discriminant = B*B - 4 * A*C;
        //If the discriminant is negative, then there is no real root
        if (discriminant < 0) {
            return false;
        }
        float t = (-B - sqrt(discriminant)) / (2 * A);
        if (t < 0) {
            t = (-B + sqrt(discriminant)) / (2 * A);
        }
        if (t >= 0) {
            Point3f P = glm::vec3(r_loc.origin + t*r_loc.direction);
            //InitializeIntersection(isect, t, P);
            ComputeTBN(Point3f(P), &(isect->surfaceNormal), &(isect->tangent), &(isect->bitangent), sphere);
            isect->t = t;
            isect->point = Point3f(sphere.transform * glm::vec4(P, 1));
            return true;
        }
        return false;
    }

    Point2f GetUVCoordinates(const Point3f &point)
    {
        Point3f p = glm::normalize(point);
        float phi = atan2f(p.z, p.x);
        if (phi < 0) {
            phi += TWO_PI;
        }
        float theta = glm::acos(p.y);
        return Point2f(1 - phi / TWO_PI, 1 - theta / PI);
    }
}