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
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
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
        normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
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
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
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
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__forceinline__ __host__ __device__ float TriArea(glm::vec3 &p1, glm::vec3 &p2, glm::vec3 &p3)
{
	return glm::length(glm::cross(p1 - p2, p3 - p2)) * 0.5f;
}

__forceinline__ __host__ __device__ glm::vec3 GetTriangleNormal(Geom &g, glm::vec3 &P) 
{
	float A = TriArea(g.t.pts[0], g.t.pts[1], g.t.pts[2]);
	float A0 = TriArea(g.t.pts[1], g.t.pts[2], P);
	float A1 = TriArea(g.t.pts[0], g.t.pts[2], P);
	float A2 = TriArea(g.t.pts[0], g.t.pts[1], P);
	return glm::normalize(g.t.normals[0] * A0 / A + g.t.normals[1] * A1 / A + g.t.normals[2] * A2 / A);
}

__forceinline__ __host__ __device__ glm::vec2 GetTriangleUVs(Geom &g, glm::vec3 &P)
{
	float A = TriArea(g.t.pts[0], g.t.pts[1], g.t.pts[2]);
	float A0 = TriArea(g.t.pts[1], g.t.pts[2], P);
	float A1 = TriArea(g.t.pts[0], g.t.pts[2], P);
	float A2 = TriArea(g.t.pts[0], g.t.pts[1], P);
	return glm::clamp(g.t.uvs[0] * A0 / A + g.t.uvs[1] * A1 / A + g.t.uvs[2] * A2 / A, 0.f, 1.f);
}
__host__ __device__ float triangleIntersectionTest(Geom triangle, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
	glm::vec3 *points = triangle.t.pts;
	float S = TriArea(points[0], points[1], points[2]);
	glm::vec3 e1 = points[0] - points[1];
	glm::vec3 e2 = points[2] - points[1];

	// find normal, (a, b, c)
	glm::vec3 n = glm::normalize((glm::cross(e1, e2)));

	// D = a(x0) + b(y0) + c(z0)
	float D = glm::dot(n, points[0]);

	// substitute ray for plane equation and solve for t
	float t = (glm::dot(n, points[0] - r.origin)) / (glm::dot(n, r.direction));

	// find the point on the plane
	glm::vec3 P = r.origin + t * r.direction;

	float S1 = TriArea(P, points[1], points[2]) / S;
	float S2 = TriArea(P, points[2], points[0]) / S;
	float S3 = TriArea(P, points[0], points[1]) / S;

	bool a = 0 <= S1 && S1 <= 1;
	bool b = 0 <= S2 && S2 <= 1;
	bool c = 0 <= S3 && S3 <= 1;
	bool d = (S1 + S2 + S3) - 1.0 < 0.001;

	if (a && b && c && d) { // was hit
		normal = GetTriangleNormal(triangle, P);
		intersectionPoint = P;
		return t;
	}
	else {
		return -1;
	}
}

__host__ __device__ glm::vec3 vRotateY(glm::vec3 p, float angle) {
	float c = cos(angle);
	float s = sin(angle);
	return glm::vec3(c * p.x - s * p.z, p.y, c * p.z + s * p.x);
}


__host__ __device__ glm::mat3 mRotate(glm::vec3 angle) {
	float c = cos(angle.x);
	float s = sin(angle.x);
	glm::mat3 rx(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c);

	c = cos(angle.y);
	s = sin(angle.y);
	glm::mat3 ry(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c);

	c = cos(angle.z);
	s = sin(angle.z);
	glm::mat3 rz(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0);

	return rz * ry * rx;
}

__forceinline__ __host__ __device__ glm::vec3 palette(float t, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d)
{
	return a + b * glm::cos(6.28318f *(c*t + d));
}

__host__ __device__ float mandelbulbSDF(glm::vec3 p, glm::vec3 *resColor) {
		glm::vec3 w = p;
		float m = glm::dot(w, w);

		glm::vec4 trap = glm::vec4(glm::abs(w), m);
		float dz = 1.0;


		for (int i = 0; i < 4; i++)
		{
#if 0
			float m2 = m * m;
			float m4 = m2 * m2;
			dz = 8.0*sqrt(m4*m2*m)*dz + 1.0;

			float x = w.x; float x2 = x * x; float x4 = x2 * x2;
			float y = w.y; float y2 = y * y; float y4 = y2 * y2;
			float z = w.z; float z2 = z * z; float z4 = z2 * z2;

			float k3 = x2 + z2;
			float k2 = inversesqrt(k3*k3*k3*k3*k3*k3*k3);
			float k1 = x4 + y4 + z4 - 6.0*y2*z2 - 6.0*x2*y2 + 2.0*z2*x2;
			float k4 = x2 - y2 + z2;

			w.x = p.x + 64.0*x*y*z*(x2 - z2)*k4*(x4 - 6.0*x2*z2 + z4)*k1*k2;
			w.y = p.y + -16.0*y2*k3*k4*k4 + k1 * k1;
			w.z = p.z + -8.0*y*k4*(x4*x4 - 28.0*x4*x2*z2 + 70.0*x4*z4 - 28.0*x2*z2*z4 + z4 * z4)*k1*k2;
#else
			//dz = 8.0*glm::pow(glm::sqrt(m), 7.0f)*dz + 1.0f;
			float x = glm::sqrt(m);
			dz = 8.0f * x * x * x * x * x * x * x * dz + 1.0f;
			float r = length(w);
			float b = 8.0f*acos(w.y / r);
			float a = 8.0f*glm::atan(w.x, w.z);
			w = p + pow(r, 8.0f) * glm::vec3(sin(b)*sin(a), cos(b), sin(b)*cos(a));
#endif        

			trap = glm::min(trap, glm::vec4(glm::abs(w), m));

			m = dot(w, w);
			if (m > 256.0)
				break;
		}

		glm::vec3 a(0.5, 0.5, 0.5);
		glm::vec3 b(.5, 0.5, 0.5);
		glm::vec3 c(2.0, 1.0, 0.0);
		glm::vec3 d(.50, 0.20, 0.25);
		*resColor = palette(0.25*log(m)*sqrt(m) / dz * 100.f, a, b, c, d);

		return 0.25*log(m)*sqrt(m) / dz;
	}

__host__ __device__ float diamondSDF(glm::vec3 p) {
	return glm::length(p) - 0.5f;
	float angle = 0.0;
	float angle2 = 0.0;

	glm::vec3 posr = p;
	posr = glm::vec3(posr.x, posr.y*cos(angle2) + posr.z*sin(angle2), posr.y*sin(angle2) - posr.z*cos(angle2));
	posr = glm::vec3(posr.x*cos(angle) + posr.z*sin(angle), posr.y, posr.x*sin(angle) - posr.z*cos(angle));

	float d = 0.94;
	float b = 0.5;

	float af2 = 4. / PI;
	float s = glm::atan(posr.y, posr.x);
	float sf = floor(s*af2 + b) / af2;
	float sf2 = floor(s*af2) / af2;

	glm::vec3 flatvec(cos(sf), sin(sf), 1.444);
	glm::vec3 flatvec2(cos(sf), sin(sf), -1.072);
	glm::vec3 flatvec3(cos(s), sin(s), 0);
	float csf1 = cos(sf + 0.21);
	float csf2 = cos(sf - 0.21);
	float ssf1 = sin(sf + 0.21);
	float ssf2 = sin(sf - 0.21);
	glm::vec3 flatvec4(csf1, ssf1, -1.02);
	glm::vec3 flatvec5(csf2, ssf2, -1.02);
	glm::vec3 flatvec6(csf2, ssf2, 1.03);
	glm::vec3 flatvec7(csf1, ssf1, 1.03);
	glm::vec3 flatvec8(cos(sf2 + 0.393), sin(sf2 + 0.393), 2.21);

	float d1 = dot(flatvec, posr) - d;                           // Crown, bezel facets
	d1 = glm::max(glm::dot(flatvec2, posr) - d, d1);                       // Pavillon, pavillon facets
	d1 = glm::max(glm::dot(glm::vec3(0., 0., 1.), posr) - 0.3f, d1);             // Table
	d1 = glm::max(glm::dot(glm::vec3(0., 0., -1.), posr) - 0.865f, d1);          // Cutlet
	d1 = glm::max(glm::dot(flatvec3, posr) - 0.911f, d1);                   // Girdle
	d1 = glm::max(glm::dot(flatvec4, posr) - 0.9193f, d1);                  // Pavillon, lower-girdle facets
	d1 = glm::max(glm::dot(flatvec5, posr) - 0.9193f, d1);                  // Pavillon, lower-girdle facets
	d1 = glm::max(glm::dot(flatvec6, posr) - 0.912f, d1);                   // Crown, upper-girdle facets
	d1 = glm::max(glm::dot(flatvec7, posr) - 0.912f, d1);                   // Crown, upper-girdle facets
	d1 = glm::max(glm::dot(flatvec8, posr) - 1.131f, d1);                   // Crown, star facets
	return d1;
}
__host__ __device__ glm::vec3 getDiamondNormal(glm::vec3 p)
{
	return glm::normalize(glm::vec3(
		diamondSDF(glm::vec3(p.x + EPSILON, p.y, p.z)) - diamondSDF(glm::vec3(p.x - EPSILON, p.y, p.z)),
		diamondSDF(glm::vec3(p.x, p.y + EPSILON, p.z)) - diamondSDF(glm::vec3(p.x, p.y - EPSILON, p.z)),
		diamondSDF(glm::vec3(p.x, p.y, p.z + EPSILON)) - diamondSDF(glm::vec3(p.x, p.y, p.z - EPSILON))
	));
}

__host__ __device__ glm::vec3 getMandelbulbNormal(glm::vec3 p)
{
	glm::vec3 color;
	return glm::normalize(glm::vec3(
		mandelbulbSDF(glm::vec3(p.x + EPSILON, p.y, p.z), &color) - mandelbulbSDF(glm::vec3(p.x - EPSILON, p.y, p.z), &color),
		mandelbulbSDF(glm::vec3(p.x, p.y + EPSILON, p.z), &color) - mandelbulbSDF(glm::vec3(p.x, p.y - EPSILON, p.z), &color),
		mandelbulbSDF(glm::vec3(p.x, p.y, p.z + EPSILON), &color) - mandelbulbSDF(glm::vec3(p.x, p.y, p.z - EPSILON), &color)
	));
}

__host__ __device__ float mandelbulbTrace(glm::vec3 cam, glm::vec3 ray, float maxdist, bool &outside, glm::vec3 *resColor) {
	float t = 0;
	float dist;
	outside = diamondSDF(cam) > 0.f;
	// "Actual" tracing
	for (int i = 0; i < 100; ++i)
	{
		glm::vec3 pos = ray * t + cam;
		float dist = outside ? mandelbulbSDF(pos, resColor) : -mandelbulbSDF(pos, resColor);
		//float dist = diamondSDF(pos);
		if (glm::abs(dist) < EPSILON) {
			break;
		}

		t += dist;

		if (t >= maxdist) {
			t = -1;
			break;
		}



	}

	return t;
}

__host__ __device__ float diamondTrace(glm::vec3 cam, glm::vec3 ray, float maxdist, bool &outside) {
	float t = 0;
	float dist;
	outside = diamondSDF(cam) > 0.f;
	// "Actual" tracing
	for (int i = 0; i < 100; ++i)
	{
		glm::vec3 pos = ray * t + cam;
		float dist = outside ? diamondSDF(pos) : -diamondSDF(pos);
		//float dist = diamondSDF(pos);
		if (glm::abs(dist) < EPSILON) {
			break;
		}

		t += dist;

		if (t >= maxdist) {
			t = -1;
			break;
		}

		
		
	}
	
	return t;
}

__host__ __device__ float mandelbulbIntersectionTest(Geom g, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, glm::vec3 *resColor) {
	glm::vec3 pt = multiplyMV(g.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 dir = multiplyMV(g.inverseTransform, glm::vec4(r.direction, 0.0f));
	// do trace to get t value
	float t = mandelbulbTrace(pt, dir, 100.f, outside, resColor);

	// calculate point on surface using ray and t value
	glm::vec3 objPt = pt + t * dir;
	intersectionPoint = multiplyMV(g.transform, glm::vec4(objPt, 1.0f));

	// calculate normal using gradient
	normal = glm::normalize(multiplyMV(g.invTranspose, glm::vec4(getMandelbulbNormal(objPt), 0.0f)));
	if (!outside) normal = -normal;
	return t;
}

__host__ __device__ float diamondIntersectionTest(Geom box, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
	glm::vec3 pt = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 dir = multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f));
	// do trace to get t value
	float t = diamondTrace(pt, dir, 100.f, outside);

	// calculate point on surface using ray and t value
	glm::vec3 objPt = pt + t * dir;
	intersectionPoint = multiplyMV(box.transform, glm::vec4(objPt, 1.0f));
	
	// calculate normal using gradient
	
	normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(getDiamondNormal(objPt), 0.0f)));
	if (!outside) normal = -normal;
	return t;

}

__host__ __device__ glm::vec2 computeSphereUVs(glm::vec3 &pt) {
	glm::vec3 p = glm::normalize(pt);
	float phi = atan2f(p.z, p.x);
	if (phi < 0)
	{
		phi += TWO_PI;
	}
	float theta = glm::acos(p.y);
	return glm::vec2(1 - phi / TWO_PI, 1 - theta / PI);
}

__host__ __device__ glm::vec2 computeCubeUVs(glm::vec3 &point) {
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

__host__ __device__ void CoordinateSystem(glm::vec3& v1, glm::vec3* v2, glm::vec3* v3)
{
	if (std::abs(v1.x) > std::abs(v1.y))
		*v2 = glm::vec3(-v1.z, 0, v1.x) / std::sqrt(v1.x * v1.x + v1.z * v1.z);
	else
		*v2 = glm::vec3(0, v1.z, -v1.y) / std::sqrt(v1.y * v1.y + v1.z * v1.z);
	*v3 = glm::cross(v1, *v2);
}
__host__ __device__ void computeSphereTBN(Geom &geom, glm::vec3& P, glm::vec3 &nor, glm::vec3* tan, glm::vec3* bit) {
	glm::vec4 v = glm::vec4(glm::cross(glm::vec3(0, 1, 0), glm::normalize(P)), 0.f);
	*tan = glm::normalize(multiplyMV(geom.transform, v));
	*bit = glm::normalize(glm::cross(nor, *tan));
}


__host__ __device__ void computeCubeTBN(Geom &geom, glm::vec3& P, glm::vec3 &nor, glm::vec3* tan, glm::vec3* bit) {
	int idx = 0;
	float val = -1;
	for (int i = 0; i < 3; i++) {
		if (glm::abs(P[i]) > val) {
			int sign = P[i] < 0 ? -1 : 1;
			idx = i * sign;
			val = glm::abs(P[i]);
		}
	}
	idx = glm::abs(idx);
	glm::vec3 n(0, 0, 0);
	n[idx] = P[idx] < 0 ? -1 : 1;
	glm::vec3 t(0.f);
	glm::vec3 b(0.f);
	if (glm::abs(n.z - 1.0f) < EPSILON) {
		t = glm::vec3(1, 0, 0);
		b = glm::vec3(0, 1, 0);
	}
	else if (glm::abs(n.z + 1.0f) < EPSILON) {
		t = glm::vec3(-1, 0, 0);
		b = glm::vec3(0, 1, 0);
	}
	else if (glm::abs(n.y - 1.0f) < EPSILON) {
		t = glm::vec3(-1, 0, 0);
		b = glm::vec3(0, 0, -1);
	}
	else if (glm::abs(n.y + 1.0f) < EPSILON) {
		t = glm::vec3(-1, 0, 0);
		b = glm::vec3(0, 0, 1);
	}
	else if (glm::abs(n.x - 1.0f) < EPSILON) {
		t = glm::vec3(0, -1, 0);
		b = glm::vec3(0, 0, -1);
	}
	else if (glm::abs(n.x + 1.0f) < EPSILON) {
		t = glm::vec3(0, 1, 0);
		b = glm::vec3(0, 0, -1);
	}
	*tan = glm::normalize(multiplyMV(geom.transform, glm::vec4(t, 0.f)));
	*bit = glm::normalize(multiplyMV(geom.transform, glm::vec4(b, 0.f)));
}

Bounds boundsUnion(Bounds &b1, Bounds &b2) {
	glm::vec3 min(std::min(b1.min.x, b2.min.x),
		std::min(b1.min.y, b2.min.y),
		std::min(b1.min.z, b2.min.z));
	glm::vec3 max(std::max(b1.max.x, b2.max.x),
		std::max(b1.max.y, b2.max.y),
		std::max(b1.max.z, b2.max.z));
	Bounds b;
	b.min = min;
	b.max = max;
	return b;
}

Bounds boundsUnion(Bounds &b1, glm::vec3 &p) {
	glm::vec3 min(std::min(b1.min.x, p.x),
		std::min(b1.min.y, p.y),
		std::min(b1.min.z, p.z));
	glm::vec3 max(std::max(b1.max.x, p.x),
		std::max(b1.max.y, p.y),
		std::max(b1.max.z, p.z));
	Bounds b;
	b.min = min;
	b.max = max;
	return b;
}

int getLongestAxis(Bounds &b) {
	glm::vec3 d = b.max - b.min;
	if (d.x > d.y && d.x > d.z)
		return 0;
	else if (d.y > d.z)
		return 1;
	else
		return 2;
}

Bounds applyTransformation(glm::mat4 &tr, Bounds &b) {
	glm::vec3 min = b.min;
	glm::vec3 max = b.max;
	Bounds ret;
	glm::vec3 p = multiplyMV(tr, glm::vec4(min, 1.0f));
	ret.min = p;
	ret.max = p;

	ret = boundsUnion(ret, multiplyMV(tr, glm::vec4(max.x, min.y, min.z, 1.0f)));
	ret = boundsUnion(ret, multiplyMV(tr, glm::vec4(min.x, max.y, min.z, 1.0f)));
	ret = boundsUnion(ret, multiplyMV(tr, glm::vec4(min.x, min.y, max.z, 1.0f)));
	ret = boundsUnion(ret, multiplyMV(tr, glm::vec4(min.x, max.y, max.z, 1.0f)));
	ret = boundsUnion(ret, multiplyMV(tr, glm::vec4(max.x, max.y, min.z, 1.0f)));
	ret = boundsUnion(ret, multiplyMV(tr, glm::vec4(max.x, min.y, max.z, 1.0f)));
	ret = boundsUnion(ret, multiplyMV(tr, glm::vec4(max.x, max.y, max.z, 1.0f)));
	return ret;
}

Bounds getGeoBounds(Geom &geom) {
	if (geom.type == CUBE)
	{
		glm::vec3 min = glm::vec3(-0.5f, -0.5f, -0.5f);
		glm::vec3 max = glm::vec3(0.5f, 0.5f, 0.5f);
		Bounds objectBound;
		objectBound.min = min;
		objectBound.max = max;
		return applyTransformation(geom.transform, objectBound);
	}
	else if (geom.type == SPHERE)
	{
		glm::vec3 min(-1.0f, -1.0f, -1.0f);
		glm::vec3 max(1.0f, 1.0f, 1.0f);
		Bounds objectBound;
		objectBound.min = min;
		objectBound.max = max;
		return applyTransformation(geom.transform, objectBound);
	}
	else if (geom.type == DIAMOND)
	{
		
	}
	else if (geom.type == MANDELBULB)
	{
		
	}
	else if (geom.type == TRIANGLE)
	{
		glm::vec3 min(FLT_MAX);
		glm::vec3 max(-FLT_MAX);
		glm::vec3 *points = geom.t.pts;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (points[i][j] < min[j]) {
					min[j] = points[i][j];
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (points[i][j] > max[j]) {
					max[j] = points[i][j];
				}
			}
		}
		Bounds ret;
		glm::vec3 epsilon(.001);
		ret.max = max + epsilon;
		ret.min = min - epsilon;
		return ret;
	}
}

__host__ __device__ float boundsIntersectionTest(Bounds b, Ray r) {
	float tmin = -1e38f;
	float tmax = 1e38f;
	for (int xyz = 0; xyz < 3; ++xyz) {
		float q = r.direction[xyz];
		{
			float t1 = (b.min[xyz] - r.origin[xyz]) / q;
			float t2 = (b.max[xyz] - r.origin[xyz]) / q;
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