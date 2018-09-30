#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

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

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

__forceinline__
__host__ __device__
void CoordinateSystem(const glm::vec3& v1, glm::vec3* v2, glm::vec3* v3) {
	if (glm::abs(v1.x) > glm::abs(v1.y))
		*v2 = glm::vec3(-v1.z, 0, v1.x) / glm::sqrt(v1.x * v1.x + v1.z * v1.z);
	else
		*v2 = glm::vec3(0, v1.z, -v1.y) / glm::sqrt(v1.y * v1.y + v1.z * v1.z);
	*v3 = glm::cross(v1, *v2);
}

__host__ __device__
glm::vec2 getCubeUV(const glm::vec3 &point) {
    glm::vec3 abs = glm::min(glm::abs(point), 0.5f);
    glm::vec2 UV;//Always offset lower-left corner
    if(abs.x > abs.y && abs.x > abs.z)
    {
        UV = glm::vec2(point.z + 0.5f, point.y + 0.5f)/3.0f;
        //Left face
        if(point.x < 0)
        {
            UV += glm::vec2(0, 0.333f);
        }
        else
        {
            UV += glm::vec2(0, 0.667f);
        }
    }
    else if(abs.y > abs.x && abs.y > abs.z)
    {
        UV = glm::vec2(point.x + 0.5f, point.z + 0.5f)/3.0f;
        //Left face
        if(point.y < 0)
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
        UV = glm::vec2(point.x + 0.5f, point.y + 0.5f)/3.0f;
        //Left face
        if(point.z < 0)
        {
            UV += glm::vec2(0.667f, 0.333f);
        }
        else
        {
            UV += glm::vec2(0.667f, 0.667f);
        }
    }
    return glm::clamp(UV, glm::vec2(0), glm::vec2(1));
}

__forceinline__
__host__ __device__
int sign(float i) {
	return (i < 0) ? -1 : (i == 0.f) ? 0 : 1;
}

__host__ __device__
int GetFaceIndex(const glm::vec3 &P) {
	int idx = 0;
	float val = -1;
	for (int i = 0; i < 3; i++) {
		if (glm::abs(P[i]) > val) {
			idx = i * sign(P[i]);
			val = glm::abs(P[i]);
		}
	}
	return idx;
}

__host__ __device__
void cubeComputeTBN(glm::mat4 transform, glm::mat4 invT, 
					const glm::vec3 P, glm::vec3 *nor, glm::vec3 *tan, glm::vec3 *bit) {
	int idx = glm::abs(GetFaceIndex(glm::vec3(P)));
	glm::vec3 N(0, 0, 0);
	N[idx] = sign(P[idx]);
	*nor = glm::vec3(glm::normalize(invT * glm::vec4(N, 0)));
	CoordinateSystem(*nor, tan, bit);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, glm::vec2 &uv) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
		uv = getCubeUV(getPointOnRay(q, tmin));
        normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

__forceinline__
__host__ __device__
glm::vec2 getSphereUV(const glm::vec3 &point) {
	glm::vec3 p = glm::normalize(point);
	float phi = atan2f(p.z, p.x);
	if (phi < 0) {
		phi += TWO_PI;
	}
	float theta = glm::acos(p.y);
	return glm::clamp(glm::vec2(1 - phi / TWO_PI, 1 - theta / PI), glm::vec2(0), glm::vec2(1));
}

__host__ __device__
void sphereComputeTBN(glm::mat4 transform, glm::mat4 invT, 
					  const glm::vec3 P, glm::vec3 *nor, glm::vec3 *tan, glm::vec3 *bit) {
	*nor = glm::vec3(glm::normalize(invT * glm::vec4(glm::normalize(P), 0)));
	*tan = glm::vec3(glm::normalize(transform * glm::vec4(glm::cross(glm::vec3(0, 1, 0), (glm::normalize(P))), 1)));
	*bit = glm::normalize(glm::cross(*nor, *tan));
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, glm::vec2 &uv) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
	uv = getSphereUV(objspaceIntersection);
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__
float getArea(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3) {
	glm::vec3 AB = p2 - p1;
	glm::vec3 AC = p3 - p1;
	return 0.5 * glm::length(glm::cross(AB, AC));
}

__host__ __device__
glm::vec3 getTriangleNor(glm::vec3 P, glm::vec3 *vertices, glm::vec3 *nors) {
	glm::vec3 v0 = vertices[1] - vertices[0];
	glm::vec3 v1 = vertices[2] - vertices[0];
	glm::vec3 v2 = P - vertices[0];
	float d00 = glm::dot(v0, v0);
	float d01 = glm::dot(v0, v1);
	float d11 = glm::dot(v1, v1);
	float d20 = glm::dot(v2, v0);
	float d21 = glm::dot(v2, v1);
	float d = d00 * d11 - d01 * d01;
	float l0 = (d11 * d20 - d01 * d21) / d;
	float l1 = (d00 * d21 - d01 * d20) / d;
	float l2 = 1 - l0 - l1;

	glm::vec3 result;
	result.x = nors[0][0] * l2 + nors[1][0] * l0 + nors[2][0] * l1;
	result.y = nors[0][1] * l2 + nors[1][1] * l0 + nors[2][1] * l1;
	result.z = nors[0][2] * l2 + nors[1][2] * l0 + nors[2][2] * l1;
	return result;
}

__host__ __device__
glm::vec2 getTriangleUV(glm::vec3 P, glm::vec3 *vertices, glm::vec2 *uvs) {
	glm::vec3 v0 = vertices[1] - vertices[0];
	glm::vec3 v1 = vertices[2] - vertices[0];
	glm::vec3 v2 = P - vertices[0];
	float d00 = glm::dot(v0, v0);
	float d01 = glm::dot(v0, v1);
	float d11 = glm::dot(v1, v1);
	float d20 = glm::dot(v2, v0);
	float d21 = glm::dot(v2, v1);
	float d = d00 * d11 - d01 * d01;
	float l0 = (d11 * d20 - d01 * d21) / d;
	float l1 = (d00 * d21 - d01 * d20) / d;
	float l2 = 1 - l0 - l1;

	glm::vec2 result;
	result.x = uvs[0][0] * l2 + uvs[1][0] * l0 + uvs[2][0] * l1;
	result.y = uvs[0][1] * l2 + uvs[1][1] * l0 + uvs[2][1] * l1;
	return result;
}

__host__ __device__
void triangleComputeTBN(Geom &geom,
						glm::vec3 P, glm::vec3 *nor, glm::vec3 *tan, glm::vec3 *bit) {
	*nor = getTriangleNor(P, geom.pos, geom.nor);
	CoordinateSystem(*nor, tan, bit);
}

__host__ __device__
float triangleIntersectionTest(Geom tri, Ray r, glm::vec3 &intersectionPoint,
							   glm::vec3 &normal, bool &outside, glm::vec2 &uv) {
	normal = -glm::normalize(glm::cross(
									  tri.pos[0] - tri.pos[1],
									  tri.pos[2] - tri.pos[1]));

	float D = glm::dot(normal, tri.pos[0]);
	if (D == 0) { return -1; }
	float t = (glm::dot(normal, tri.pos[0] - r.origin)) / glm::dot(normal, r.direction);
	if (t < 0) { return -1; }
	glm::vec3 P = r.origin + t * r.direction;
	outside = true;
	float S = getArea(tri.pos[0], tri.pos[1], tri.pos[2]);
	float S1 = getArea(P, tri.pos[1], tri.pos[2]) / S;
	float S2 = getArea(P, tri.pos[2], tri.pos[0]) / S;
	float S3 = getArea(P, tri.pos[0], tri.pos[1]) / S;

	normal = getTriangleNor(P, tri.pos, tri.nor);
	uv = getTriangleUV(P, tri.pos, tri.uv);
	uv = glm::clamp(uv, glm::vec2(0), glm::vec2(1));

	if (0 <= S1 && S1 <= 1 &&
		0 <= S2 && S2 <= 1 &&
		0 <= S3 && S3 <= 1 && (S1 + S2 + S3 - 1 < 0.001)) {
		intersectionPoint = P;
		return t;
	}
	return -1;
}

__host__ __device__
void computeTBN(Geom &geom, const glm::vec3 P, glm::vec3 *nor, glm::vec3 *tan, glm::vec3 *bit) {
	if (geom.type == CUBE) {
		cubeComputeTBN(geom.transform, geom.invTranspose, P, nor, tan, bit);
	} else if (geom.type == SPHERE) {
		sphereComputeTBN(geom.transform, geom.invTranspose, P, nor, tan, bit);
	} else if (geom.type == TRIANGLE) {
		triangleComputeTBN(geom, P, nor, tan, bit);
	}
}

// BOUNDING BOX STUFF
__host__ __device__
float boundsIntersectionTest(Bounds b, Ray r) {
	float tmin = -1e38f;
	float tmax = 1e38f;
	for (int xyz = 0; xyz < 3; ++xyz) {
		float qdxyz = r.direction[xyz];
		/*if (glm::abs(qdxyz) > 0.00001f)*/
		{
			float t1 = (b.min[xyz] - r.origin[xyz]) / qdxyz;
			float t2 = (b.max[xyz] - r.origin[xyz]) / qdxyz;
			float ta = glm::min(t1, t2);
			float tb = glm::max(t1, t2);
			if (ta > 0 && ta > tmin)
				tmin = ta;
			if (tb < tmax)
				tmax = tb;
		}
	}
	if (tmax >= tmin && tmax > 0) {
		return tmin;
	}
	return -1;

}

__device__ __host__
Bounds Union(const Bounds &b1, const Bounds &b2) {
	Bounds ret;
	ret.min.x = glm::min(b1.min.x, b2.min.x);
	ret.min.y = glm::min(b1.min.y, b2.min.y);
	ret.min.z = glm::min(b1.min.z, b2.min.z);
	ret.max.x = glm::max(b1.max.x, b2.max.x);
	ret.max.y = glm::max(b1.max.y, b2.max.y);
	ret.max.z = glm::max(b1.max.z, b2.max.z);
	return ret;
}

Bounds GetBounds(Geom geo) {
	if (geo.type == CUBE) {
		std::vector<glm::vec4> world;
		world.push_back(geo.transform * glm::vec4(-0.5, -0.5, -0.5, 1));
		world.push_back(geo.transform * glm::vec4(-0.5, -0.5, 0.5, 1));
		world.push_back(geo.transform * glm::vec4(-0.5, 0.5, -0.5, 1));
		world.push_back(geo.transform * glm::vec4(-0.5, 0.5, 0.5, 1));
		world.push_back(geo.transform * glm::vec4(0.5, -0.5, -0.5, 1));
		world.push_back(geo.transform * glm::vec4(0.5, -0.5, 0.5, 1));
		world.push_back(geo.transform * glm::vec4(0.5, 0.5, -0.5, 1));
		world.push_back(geo.transform * glm::vec4(0.5, 0.5, 0.5, 1));
		Bounds ret;
		ret.min = glm::vec3(FLT_MAX);
		ret.max = glm::vec3(FLT_MIN);
		for (const auto &v : world) {
			ret.min.x = v.x < ret.min.x ? v.x : ret.min.x;
			ret.max.x = v.x > ret.max.x ? v.x : ret.max.x;
			ret.min.y = v.y < ret.min.y ? v.y : ret.min.y;
			ret.max.y = v.y > ret.max.y ? v.y : ret.max.y;
			ret.min.z = v.z < ret.min.z ? v.z : ret.min.z;
			ret.max.z = v.z > ret.max.z ? v.z : ret.max.z;
		}
		ret.max += glm::vec3(0.0001);
		ret.min -= glm::vec3(0.0001);
		return ret;
	} else if (geo.type == SPHERE) {
		glm::vec3 center = glm::vec3(0);
		Bounds ret;
		ret.min = glm::vec3(center - glm::vec3(0.5));
		ret.max = glm::vec3(center + glm::vec3(0.5));
		ret.min = multiplyMV(geo.transform, glm::vec4(ret.min, 1));
		ret.max = multiplyMV(geo.transform, glm::vec4(ret.max, 1));
		ret.max += glm::vec3(0.0001);
		ret.min -= glm::vec3(0.0001);
		return ret;
	} else if (geo.type == TRIANGLE) {
		Bounds b0 { geo.pos[0], geo.pos[0] };
		Bounds b1 { geo.pos[1], geo.pos[1] };
		Bounds b2 { geo.pos[2], geo.pos[2] };
		Bounds ret = Union(b0, Union(b1, b2));
		ret.max += glm::vec3(0.0001);
		ret.min -= glm::vec3(0.0001);
		return ret;
	}
	return { glm::vec3(0), glm::vec3(0) };
}

Bounds GetBounds(std::vector<Geom> &geoms) {
	if (geoms.size() == 0) {
		return { glm::vec3(0), glm::vec3(0) };
	}
	Bounds ret = GetBounds(geoms[0]);
	for (const auto &geo : geoms) {
		ret = Union(ret, GetBounds(geo));
	}
	return ret;
}

int longestAxis(Bounds &b) {
	float xLength = b.max.x - b.min.x;
	float yLength = b.max.y - b.min.y;
	float zLength = b.max.z - b.min.z;
	if (xLength > yLength && xLength > zLength) {
		return 0;
	} else if (yLength > xLength && yLength > zLength) {
		return 1;
	} else {
		return 2;
	}
}

glm::vec3 getMidpoint(Bounds &b) {
	return glm::vec3((b.max.x - b.min.x) / 2,
		(b.max.y - b.min.y) / 2,
					 (b.max.z - b.min.z) / 2);
}