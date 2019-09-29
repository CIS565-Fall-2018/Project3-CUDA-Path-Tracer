#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

//My code here
unsigned int myHash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

//My code here
void debugNodeIntersectionTest(int times, const Triangle * triangles, const int rootIndex)
{
	printf("debug node intersection test start!\n");

	for (int i = 0; i < times; i++)
	{
		Ray r;
		int hitIndex;
		int h = myHash((1 << 31) | (i << 22) | i) ^ myHash(i);
		thrust::default_random_engine rng = thrust::default_random_engine(h);
		thrust::uniform_real_distribution<float> u01(0.f, 1.f);

		r.origin = glm::vec3(u01(rng), u01(rng), u01(rng));
		r.direction = glm::normalize(glm::vec3(u01(rng), u01(rng), u01(rng)));

		nodeIntersectionTest(triangles, rootIndex, r, &hitIndex);
		printf("ray %d: %d\n", i, hitIndex);
	}

	printf("debug node intersection test done!\n");
}


//My code here
__host__ __device__ float PowerHeuristic(int nf, float fPdf, int ng, float gPdf)
{
	//TODO
	float f = nf * fPdf, g = ng * gPdf;
	return (f*f) / (f*f + g * g);
}

//My code here
__host__ __device__ bool operator<(const ShadeableIntersection& lhs, const ShadeableIntersection& rhs)
{
	return lhs.materialId < rhs.materialId;
}

//My code here
__host__ __device__ bool isNan(const glm::vec3& v)
{
	if (glm::isnan<float>(v.x) || glm::isnan<float>(v.y) || glm::isnan<float>(v.z))
		return true;
	else
		return false;
}

//My code here
__host__ __device__ bool isNan(float v)
{
	if (glm::isnan<float>(v))
		return true;
	else
		return false;
}

//My code here
__host__ __device__ bool isZero(const glm::vec3& v)
{
	if (glm::abs(v.x) < EPSILON && glm::abs(v.y) < EPSILON && glm::abs(v.z) < EPSILON)
		return true;
	else
		return false;
}

//My code here
__host__ __device__ bool isZero(float v)
{
	if (glm::abs(v) < EPSILON)
		return true;
	else
		return false;
}

//My code here
__host__ __device__ glm::vec2 squareToDiskUniform(const glm::vec2 &sample)
{
	return glm::sqrt(sample.y) * glm::vec2(glm::cos(2.f*glm::pi<float>()*sample.x), glm::sin(2.f*glm::pi<float>()*sample.x));
}

//My code here
__host__ __device__ glm::vec2 squareToDiskConcentric(const glm::vec2 &sample)
{
	//TODO
	float phi = 0;
	float a = 2.f*sample.x - 1.f;
	float b = 2.f*sample.y - 1.f;
	float r = 0;

	if (a > -b)
	{
		if (a > b)
		{
			r = a;
			phi = (glm::pi<float>() / 4.f) * (b / a);
		}
		else
		{
			r = b;
			phi = (glm::pi<float>() / 4.f) * (2.f - a / b);
		}
	}
	else
	{
		if (a < b)
		{
			r = -a;
			phi = (glm::pi<float>() / 4.f) * (4.f + (b / a));
		}
		else
		{
			r = -b;
			if (b != 0)
				phi = (glm::pi<float>() / 4.f) * (6.f - (a / b));
			else
				phi = 0;
		}
	}
	return glm::vec2(r*glm::cos(phi), r*glm::sin(phi));
}

//My code here
__host__ __device__ glm::vec3 squareToHemisphereCosine(const glm::vec2 &sample)
{
	glm::vec2 temp = squareToDiskConcentric(sample);// squareToDiskUniform(sample);//
	float z = glm::sqrt(1 - temp.x*temp.x - temp.y*temp.y);
	return glm::vec3(temp.x, temp.y, z);
}

//My code here
__host__ __device__ float squareToHemisphereCosinePDF(const glm::vec3 &sample)
{
	return glm::abs(glm::dot(sample, glm::vec3(0, 0, 1))) / glm::pi<float>();
}

//My code here
__global__ void kernelMapToBooleanAndGather(int n, glm::vec3* image, int* dev_paths_exist, const PathSegment* dev_paths)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n) return;

	if (dev_paths[index].remainingBounces > 0)
	{
		dev_paths_exist[index] = 1;
	}
	else
	{
		dev_paths_exist[index] = 0;
		image[dev_paths[index].pixelIndex] += dev_paths[index].color;
	}
}

//My code here
__global__ void kernelScatter(int n, PathSegment* dev_paths_out, const PathSegment* dev_paths_in, const int* dev_paths_exist, const int* dev_paths_indices)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n) return;

	if (dev_paths_exist[index])
	{
		dev_paths_out[dev_paths_indices[index] - 1] = dev_paths_in[index];//-1 because we are using inclusive scan
	}

}

//My code here
__host__ __device__ void SampleMaterialOld(PathSegment &pathSegment, const ShadeableIntersection &intersection, const Material &material, const glm::vec2 &xi)
{
	glm::vec3 Li = pathSegment.color;
	float pdf = 0;
	glm::vec3 wi(0, 0, 0);
	glm::vec3 p(0, 0, 0);

	glm::mat3 tangentToWorld(intersection.surfaceTangent, intersection.surfaceBitangent, intersection.surfaceNormal);


	if (material.type == 0) //diffuse
	{
		glm::vec3 color = material.color / glm::pi<float>();
		wi = squareToHemisphereCosine(xi);
		pdf = squareToHemisphereCosinePDF(wi);//take the pdf before convert to world coord
		wi = tangentToWorld * wi;
		wi = glm::normalize(wi);
		Li = color * Li / pdf * glm::abs(glm::dot(wi, intersection.surfaceNormal));
		p = intersection.point + MY_OFFSET * intersection.surfaceNormal;
	}
	else if (material.type == 1)//emissive
	{
		glm::vec3 color = material.emissive.color * material.emissive.emittance;
		Li = color * Li;
		pathSegment.hitLight = true;//early termination, mark it so that result won't be set to black
		pathSegment.remainingBounces = 0;
		p = intersection.point + MY_OFFSET * intersection.surfaceNormal;
	}
	else if (material.type == 2)//specular reflection
	{
		glm::vec3 color = material.specularReflective.color / glm::abs(glm::dot(-pathSegment.ray.direction, intersection.surfaceNormal));
		wi = glm::reflect(pathSegment.ray.direction, intersection.surfaceNormal);
		wi = glm::normalize(wi);
		pdf = 1;
		Li = color * Li / pdf * glm::abs(glm::dot(wi, intersection.surfaceNormal));
		p = intersection.point + MY_OFFSET * intersection.surfaceNormal;
	}
	else if (material.type == 3)//specular refraction
	{
		glm::vec3 color = material.specularTransmissive.color / glm::abs(glm::dot(-pathSegment.ray.direction, intersection.surfaceNormal));
		wi = intersection.outside ?
			glm::refract(pathSegment.ray.direction, intersection.surfaceNormal, material.specularTransmissive.indexOfRefraction) :
			glm::refract(pathSegment.ray.direction, intersection.surfaceNormal, 1.f / material.specularTransmissive.indexOfRefraction);
		if (isZero(wi))
		{
			pathSegment.remainingBounces = 0;
		}
		wi = glm::normalize(wi);
		pdf = 1;
		Li = color * Li / pdf * glm::abs(glm::dot(wi, -intersection.surfaceNormal));//when refract, normal is on the opposite side of wi
		p = intersection.point - MY_OFFSET * intersection.surfaceNormal;//when refract, normal is on the opposite side of wi
	}

	pathSegment.color = Li;
	pathSegment.ray.origin = p;
	pathSegment.ray.direction = wi;
}

//My code here
__host__ __device__ glm::vec3 SampleMaterial(const Ray& ray, const ShadeableIntersection &intersection, const Material &material, const glm::vec2 &xi, glm::vec3 &wi, float *pdf)
{
	glm::vec3 color_tmp(0, 0, 0);
	float pdf_tmp = 0;
	glm::vec3 wi_tmp(0, 0, 0);

	glm::mat3 tangentToWorld(intersection.surfaceTangent, intersection.surfaceBitangent, intersection.surfaceNormal);

	if (material.type == 0) //diffuse
	{
		color_tmp = material.color / glm::pi<float>();
		wi_tmp = squareToHemisphereCosine(xi);
		pdf_tmp = squareToHemisphereCosinePDF(wi_tmp);//take the pdf before convert to world coord
		wi_tmp = tangentToWorld * wi_tmp;
	}
	else if (material.type == 1)//emissive
	{
		color_tmp = material.emissive.color * material.emissive.emittance;
	}
	else if (material.type == 2)//specular reflection
	{
		color_tmp = material.specularReflective.color / glm::abs(glm::dot(-ray.direction, intersection.surfaceNormal));
		wi_tmp = glm::reflect(ray.direction, intersection.surfaceNormal);
		pdf_tmp = 1;
	}
	else if (material.type == 3)//specular refraction
	{
		color_tmp = material.specularTransmissive.color / glm::abs(glm::dot(-ray.direction, intersection.surfaceNormal));
		wi_tmp = intersection.outside ?
			glm::refract(ray.direction, intersection.surfaceNormal, material.specularTransmissive.indexOfRefraction) :
			glm::refract(ray.direction, intersection.surfaceNormal, 1.f / material.specularTransmissive.indexOfRefraction);
		if (isZero(wi_tmp))//total internal refraction
		{
			color_tmp = glm::vec3(0, 0, 0);
		}
		pdf_tmp = 1;
	}

	wi = glm::normalize(wi_tmp);
	if(pdf!=nullptr) *pdf = pdf_tmp;
	return color_tmp;
}

//My code here
__host__ __device__ glm::vec3 NotSampleMaterial(const ShadeableIntersection &intersection, const Material &material, const glm::vec3 &wi, float *pdf)
{
	glm::vec3 color_tmp(0, 0, 0);
	float pdf_tmp = 0;

	glm::mat3 tangentToWorld(intersection.surfaceTangent, intersection.surfaceBitangent, intersection.surfaceNormal);
	glm::mat3 worldToTangent = glm::inverse(tangentToWorld);
	glm::vec3 wiL = glm::normalize(worldToTangent * wi);

	if (material.type == 0) //diffuse
	{
		color_tmp = material.color / glm::pi<float>();
		pdf_tmp = squareToHemisphereCosinePDF(wiL);
	}
	else if (material.type == 1)//emissive
	{
		color_tmp = material.emissive.color * material.emissive.emittance;
	}
	else if (material.type == 2)//specular reflection
	{
		//we are samping light not material, so this will not happen
		color_tmp = glm::vec3(0, 0, 0);
		pdf_tmp = 0;
	}
	else if (material.type == 3)//specular refraction
	{
		//we are samping light not material, so this will not happen
		color_tmp = glm::vec3(0, 0, 0);
		pdf_tmp = 0;
	}

	if (pdf != nullptr) *pdf = pdf_tmp;
	return color_tmp;
}

//My code here
__host__ __device__ bool ShadowFeeler(const Ray& r, const Geom * geoms, int geomsSize, const Triangle * triangles, int * hitGeomId)
{
	float t = -1;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;
	glm::vec3 tmp_tangent;
	glm::vec3 tmp_bitangent;
	bool tmp_outside;
	glm::vec2 tmp_uv;

	for (int i = 0; i < geomsSize; i++)
	{
		Geom geom = geoms[i];

		if (geom.type == CUBE)
		{
			t = boxIntersectionTest(geom, r, tmp_intersect, tmp_normal, tmp_tangent, tmp_bitangent, tmp_outside);
		}
		else if (geom.type == SPHERE)
		{
			t = sphereIntersectionTest(geom, r, tmp_intersect, tmp_normal, tmp_tangent, tmp_bitangent, tmp_outside);
		}
		else if (geom.type == MESH)
		{
			t = meshIntersectionTest(triangles, geom, r, tmp_intersect, tmp_normal, tmp_tangent, tmp_bitangent, tmp_uv, tmp_outside);
		}
		else if (geom.type == SQUARE)
		{
			t = squareIntersectionTest(geom, r, tmp_intersect, tmp_normal, tmp_tangent, tmp_bitangent, tmp_outside);
		}

		if (t > 0.0f && t_min > t)
		{
			t_min = t;
			hit_geom_index = i;
		}
	}

	if (hit_geom_index != -1)
	{
		if (hitGeomId != nullptr)
		{
			*hitGeomId = hit_geom_index;
		}
		return true;
	}
	
	return false;
}

//My code here
__host__ __device__ glm::vec3 SampleLight(const Geom &light, const ShadeableIntersection &intersection, const Material &lightMaterial, const glm::vec2 xi, glm::vec3 &wi, float *pdf)
{
	glm::vec3 p_tmp(0, 0, 0);
	glm::vec3 n_tmp(0, 0, 0);
	float pdf_tmp = 1;
	glm::vec3 result(0, 0, 0);

	if (light.type == SQUARE)
	{
		glm::vec4 plocal((xi.x - 0.5), (xi.y - 0.5), 0, 1);
		glm::vec4 nlocal(0, 0, 1, 0);//this direction could also be 0,0,-1 but it does not matter in this case

		p_tmp = multiplyMV(light.transform, plocal);
		n_tmp = glm::normalize(multiplyMV(light.invTranspose, nlocal));

		float area = glm::abs(light.scale.x * light.scale.y);
		pdf_tmp = 1.0 / area;
	}
	else if (light.type == CUBE)
	{
		// do nothing
	}
	else if (light.type == SPHERE)
	{
		// do nothing
	}
	else if (light.type == MESH)
	{
		// do nothing
	}

	float absDot = glm::abs(glm::dot(n_tmp, glm::normalize(intersection.point - p_tmp)));

	wi = glm::normalize(p_tmp - intersection.point);//from material to light

	if (isZero(absDot) || isNan(absDot))
	{
		result = glm::vec3(0, 0, 0);
		if (pdf != nullptr) *pdf = 0;
	}
	else
	{
		result = lightMaterial.emissive.emittance * lightMaterial.emissive.color;
		//solid angle
		float distance = glm::length(p_tmp - intersection.point);
		if (pdf !=nullptr) *pdf = distance * distance * pdf_tmp / absDot;
	}

	return result;
}

//My code here
__host__ __device__ glm::vec3 NotSampleLight(const Geom &light, const ShadeableIntersection &intersection, const Material &lightMaterial, const glm::vec3 &wi, float *pdf)
{
	//glm::mat3 tangentToWorld(intersection.surfaceTangent, intersection.surfaceBitangent, intersection.surfaceNormal);
	glm::vec3 p_tmp(0, 0, 0);
	glm::vec3 n_tmp(0, 0, 0);
	float pdf_tmp = 1;
	glm::vec3 result(0, 0, 0);
	float t_tmp = -1;
	Ray tmp_r;
	tmp_r.origin = intersection.point;
	tmp_r.direction = wi;
	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;
	glm::vec3 tmp_tangent;
	glm::vec3 tmp_bitangent;
	bool tmp_outside;

	if (light.type == SQUARE)
	{
		t_tmp = squareIntersectionTest(light, tmp_r, tmp_intersect, tmp_normal, tmp_tangent, tmp_bitangent, tmp_outside);
		
		if (t_tmp > 0.0f)
		{
			p_tmp = tmp_intersect;
		}

		glm::vec4 nlocal(0, 0, 1, 0);

		n_tmp = glm::normalize(multiplyMV(light.invTranspose, nlocal));//this direction could also be 0,0,-1 but it does not matter in this case

		float area = glm::abs(light.scale.x * light.scale.y);
		pdf_tmp = 1.0 / area;
	}
	else if (light.type == CUBE)
	{
		// do nothing
	}
	else if (light.type == SPHERE)
	{
		// do nothing
	}
	else if (light.type == MESH)
	{
		// do nothing
	}

	float absDot = glm::abs(glm::dot(n_tmp, glm::normalize(intersection.point - p_tmp)));

	if (isZero(absDot) || isNan(absDot))
	{
		result = glm::vec3(0, 0, 0);
		if (pdf != nullptr) *pdf = 0;
	}
	else
	{
		result = lightMaterial.emissive.emittance * lightMaterial.emissive.color;
		//solid angle
		float distance = glm::length(p_tmp - intersection.point);
		if (pdf != nullptr) *pdf = distance * distance * pdf_tmp / absDot;
	}

	return result;
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static int* dev_paths_exist = NULL;
static PathSegment * dev_paths_temp = NULL;
static int* dev_paths_indices = NULL;
static int* dev_intersections_material_id = NULL;
static PathSegment * dev_paths_cache = NULL;
static ShadeableIntersection * dev_intersections_cache = NULL;
static bool paths_cached = false;
static Triangle * dev_triangles = NULL;
static Geom * dev_lights = NULL;

void pathtraceInit(Scene *scene) {
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need

	cudaMalloc(&dev_paths_exist, pixelcount * sizeof(int));
	cudaMalloc(&dev_paths_temp, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_paths_indices, pixelcount * sizeof(int));
	cudaMalloc(&dev_intersections_material_id, pixelcount * sizeof(int));
	cudaMalloc(&dev_paths_cache, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection));
	paths_cached = false;
	cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
	cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_paths_exist);
	cudaFree(dev_paths_temp);
	cudaFree(dev_paths_indices);
	cudaFree(dev_intersections_material_id);
	cudaFree(dev_paths_cache);
	cudaFree(dev_intersections_cache);
	cudaFree(dev_triangles);
	cudaFree(dev_lights);

	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, glm::vec3 initColor)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		//PathSegment & segment = pathSegments[index];//why ?
		PathSegment segment;

		segment.ray.origin = cam.position;
		segment.color = initColor;// glm::vec3(1.0f, 1.0f, 1.0f); // for multi-importance sampling
		segment.throughput = glm::vec3(1, 1, 1);
		segment.hitSpecular = false;

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
		segment.hitLight = false;
		for (int i = 0; i < MAX_RECORD_DEPTH; i++)
		{
			segment.geomIds[i] = -1;
			segment.outsides[i] = true;
		}
		pathSegments[index] = segment;
	}
}

__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, const Geom * geoms
	, const Triangle * triangles//My code here
	, const Material * materials
	, int geoms_size
	, ShadeableIntersection * intersections
	, int * intersections_material_id
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		//My code here. Pre-fetch
		PathSegment pathSegment = pathSegments[path_index];
		ShadeableIntersection intersection = intersections[path_index];
		int intersection_material_id = intersections_material_id[path_index];

		if (pathSegment.remainingBounces > 0)
		{
			float t;
			glm::vec3 intersect_point;
			glm::vec3 normal;
			float t_min = FLT_MAX;
			int hit_geom_index = -1;
			//My code here.
			glm::vec3 tangent;
			glm::vec3 bitangent;
			Geom hit_geom;
			bool outside;
			glm::vec2 uv;

			glm::vec3 tmp_intersect;
			//My code here.
			glm::vec3 tmp_normal;
			glm::vec3 tmp_tangent;
			glm::vec3 tmp_bitangent;
			bool tmp_outside;
			glm::vec2 tmp_uv;

			// naive parse through global geoms

			for (int i = 0; i < geoms_size; i++)
			{
				// is using reference faster in GPU?
				Geom geom = geoms[i];

				if (geom.type == CUBE)
				{
					t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_tangent, tmp_bitangent, tmp_outside);
				}
				else if (geom.type == SPHERE)
				{
					t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_tangent, tmp_bitangent, tmp_outside);
				}
				// TODO: add more intersection tests here... triangle? metaball? CSG?
				else if (geom.type == MESH)
				{
					t = meshIntersectionTest(triangles, geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_tangent, tmp_bitangent, tmp_uv, tmp_outside);
				}
				else if (geom.type == SQUARE)
				{
					t = squareIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_tangent, tmp_bitangent, tmp_outside);
				}

				// Compute the minimum t from the intersection tests to determine what
				// scene geometry object was hit first.
				if (t > 0.0f && t_min > t)
				{
					t_min = t;
					hit_geom_index = i;
					hit_geom = geom;
					intersect_point = tmp_intersect;
					normal = tmp_normal;
					tangent = tmp_tangent;
					bitangent = tmp_bitangent;
					outside = tmp_outside;
					uv = tmp_uv;
				}
			}

			if (hit_geom_index == -1)
			{
				intersection.t = -1.0f;
				intersection_material_id = -1;//here
			}
			else
			{
				//The ray hits something
				intersection.t = t_min;
				intersection.materialId = hit_geom.materialid;
				intersection.surfaceNormal = normal;
				//My code here. Also I don't think access global memory multiple times is a good idea even though its value may be cached.
				//I think it's better to have a local copy stored in a register and write it back to global buffer at the end.
				intersection.point = intersect_point;
				intersection.surfaceTangent = tangent;
				intersection.surfaceBitangent = bitangent;
				intersection.geomId = hit_geom_index;
				intersection.outside = outside;
				intersection_material_id = materials[hit_geom.materialid].type;//type is enough to batch a group of paths
				intersection.uv = uv;

				//record the path for debugging
				if (depth <= MAX_RECORD_DEPTH)
				{
					pathSegment.geomIds[depth - 1] = hit_geom_index;
					pathSegment.outsides[depth - 1] = outside;
				}
			}
		}
		else
		{
			//My code here. This brach won't happen if the stream compaction clears all path whose remaingBounces is 0
			intersection.t = -1.0f;
		}

		//My code here. Write back.
		pathSegments[path_index] = pathSegment;
		intersections[path_index] = intersection;
		intersections_material_id[path_index] = intersection_material_id;
	}
}

__global__ void shadeMaterialOld(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		PathSegment pathSegment = pathSegments[idx];
		if (pathSegment.remainingBounces > 0)
		{
			ShadeableIntersection intersection = shadeableIntersections[idx];

			if (intersection.t > 0.0f)
			{
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
				Material material = materials[intersection.materialId];
				thrust::uniform_real_distribution<float> u01(0.f, 1.f);
				glm::vec2 xi(u01(rng), u01(rng));
				SampleMaterialOld(pathSegment, intersection, material, xi);
				pathSegment.remainingBounces--;
				if (pathSegment.remainingBounces == 0 && pathSegment.hitLight == false) pathSegment.color = glm::vec3(0, 0, 0);
			}
			else
			{
				//if implemented stream compaction, this branch will not be executed
				pathSegment.color = glm::vec3(0.0f);
				pathSegment.remainingBounces = 0;
			}

			pathSegments[idx] = pathSegment;
		}
	}
}

__global__ void shadeMaterialNaive(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		PathSegment pathSegment = pathSegments[idx];
		if (pathSegment.remainingBounces > 0)
		{
			ShadeableIntersection intersection = shadeableIntersections[idx];

			if (intersection.t > 0.0f) 
			{
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
				Material material = materials[intersection.materialId];
				thrust::uniform_real_distribution<float> u01(0.f, 1.f);
				glm::vec2 xi(u01(rng), u01(rng));

				/////////////////////////////////////////////////////////////////////////////////////////
				glm::vec3 wi(0, 0, 0);
				float pdf = 1;
				glm::vec3 color = SampleMaterial(pathSegment.ray, intersection, material, xi, wi, &pdf);
				
				if (isZero(color))//total internal reflection, early terminate
				{
					pathSegment.remainingBounces = 0;
				}
				else 
				if (material.type == 1)//if the material is emissive(light), early terminate
				{
					pathSegment.hitLight = true;
					pathSegment.color = pathSegment.color * color;
					pathSegment.remainingBounces = 0;
				}
				else if (material.type == 3)//if the material is transmissive
				{
					pathSegment.color = pathSegment.color * color / pdf * glm::abs(glm::dot(wi, -intersection.surfaceNormal));//when refract, normal is on the opposite side of wi
					pathSegment.ray.origin = intersection.point - MY_OFFSET * intersection.surfaceNormal;//when refract, normal is on the opposite side of wi
					pathSegment.ray.direction = wi;
					pathSegment.remainingBounces--;
				}
				else//if the material is others
				{
					pathSegment.color = pathSegment.color * color / pdf * glm::abs(glm::dot(wi, intersection.surfaceNormal));
					pathSegment.ray.origin = intersection.point + MY_OFFSET * intersection.surfaceNormal;
					pathSegment.ray.direction = wi;
					pathSegment.remainingBounces--;
				}
				/////////////////////////////////////////////////////////////////////////////////////////

				if (pathSegment.remainingBounces == 0 && pathSegment.hitLight == false) pathSegment.color = glm::vec3(0, 0, 0);
			}
			else 
			{
				//if implemented stream compaction, this branch will not be executed
				pathSegment.color = glm::vec3(0.0f);
				pathSegment.remainingBounces = 0;
			}

			pathSegments[idx] = pathSegment;
		}
	}
}

__global__ void shadeMaterialDirectLight(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	// My code here
	, const Geom * lights
	, int lightsSize
	, const Geom * geoms
	, int geomsSize
	, Triangle * triangles
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		PathSegment pathSegment = pathSegments[idx];
		if (pathSegment.remainingBounces > 0)
		{
			ShadeableIntersection intersection = shadeableIntersections[idx];

			if (intersection.t > 0.0f)
			{
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
				Material material = materials[intersection.materialId];
				thrust::uniform_real_distribution<float> u01(0.f, 1.f);
				glm::vec2 xi(u01(rng), u01(rng));
				glm::vec2 xii(u01(rng), u01(rng));
				glm::vec2 xiii(u01(rng), u01(rng));
				int randomLightIndex = u01(rng) * lightsSize;//half open
				Geom light = lights[randomLightIndex];
				Material lightMaterial = materials[light.materialid];
				glm::vec3 throughput = pathSegment.throughput;

				/////////////////////////////////////////////////////////////////////////////////////////
				glm::vec3 wiLight_SampleLight(0, 0, 0);
				glm::vec3 wiLight_SampleMaterial(0, 0, 0);
				glm::vec3 wiGlobal(0, 0, 0);
				float pdfGlobal = 1;
				float pdfLight_SampleMaterial = 1;
				float pdfLight_SampleLight = 1;
				float pdfMaterial_SampleMaterial = 1;
				float pdfMaterial_SampleLight = 1;
				glm::vec3 Li(0, 0, 0);
				glm::vec3 Li_SampleLight(0, 0, 0);
				glm::vec3 Li_SampleMaterial(0, 0, 0);
				glm::vec3 colorMaterial_SampleMaterial = SampleMaterial(pathSegment.ray, intersection, material, xi, wiLight_SampleMaterial, &pdfMaterial_SampleMaterial);

				// A. Sample light and material
				if (material.type == 1)// hit a light source
				{
					//!!!!!!!!!!!!!!!!!!!!!!!!!IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!
					//Since your path tracer computes the direct lighting a given 
					//intersection receives as its own term, your path tracer must 
					//not include too much light. This means that every ray which 
					//already computed the direct lighting term should not incorporate 
					//the Le term of the light transport equation into its light 
					//contribution. In other words, unless a particular ray came 
					//directly from the camera or from a perfectly specular surface, 
					//Le should be ignored.

					//in direct light integrator, this will always be true
					//if (pathSegment.remainingBounces == maxBounce || pathSegment.hitSpecular)//haven't account for specular bounce //done
					Li = colorMaterial_SampleMaterial;
					pathSegment.remainingBounces = 0;
				}
				else// hit others
				{
					glm::vec3 o = material.type == 3 ?
						intersection.point - MY_OFFSET * intersection.surfaceNormal :
						intersection.point + MY_OFFSET * intersection.surfaceNormal;
					int hitGeomId = -1;

					// I. Compute Li
					if (material.type == 2 || material.type == 3)// Specular
					{
						// 0. Set flag
						pathSegment.hitSpecular = true;
					}
					else// Non specular
					{
						// 0. Set flag
						pathSegment.hitSpecular = false;

						// 1. Sample light
						glm::vec3 colorLight_SampleLight = SampleLight(light, intersection, lightMaterial, xii, wiLight_SampleLight, &pdfLight_SampleLight);
						pdfLight_SampleLight /= lightsSize;
						Ray rLight_SampleLight;
						rLight_SampleLight.origin = o;
						rLight_SampleLight.direction = wiLight_SampleLight;

						//shadow feeler
						if (ShadowFeeler(rLight_SampleLight, geoms, geomsSize, triangles, &hitGeomId) && hitGeomId == light.id)
						{
							//if hit the light, therefore no shadow
							//pathSegment.hitLight = true;
							glm::vec3 colorMaterial_SampleLight = NotSampleMaterial(intersection, material, wiLight_SampleLight, &pdfMaterial_SampleLight);
							colorMaterial_SampleLight *= glm::abs(glm::dot(wiLight_SampleLight, intersection.surfaceNormal));
							if (pdfLight_SampleLight != 0)
							{
								//in direct light integrator, we don't do MIS
								//float weight_SampleLight = PowerHeuristic(1, pdfLight_SampleLight, 1, pdfMaterial_SampleLight);
								//Li_SampleLight = weight_SampleLight * colorMaterial_SampleLight * colorLight_SampleLight / pdfLight_SampleLight;
								Li_SampleLight = colorMaterial_SampleLight * colorLight_SampleLight / pdfLight_SampleLight;
							}
							else
							{
								Li_SampleLight = colorMaterial_SampleLight * colorLight_SampleLight;
							}
						}
						else
						{
							//if hit nothing or hit other geom
							//pathSegment.hitLight = false;
							Li_SampleLight = glm::vec3(0, 0, 0);
						}

						//in direct light integrator, we don't sample material
						// 2. Sample material, since we already sampled one, we are using that result
						//if (pdfLight_SampleLight != 0)
						//{
						//	colorMaterial_SampleMaterial *= glm::abs(glm::dot(wiLight_SampleMaterial, intersection.surfaceNormal));
						//	Ray rLight_SampleMaterial;
						//	rLight_SampleMaterial.origin = o;
						//	rLight_SampleMaterial.direction = wiLight_SampleMaterial;

						//	//shadow feeler
						//	if (ShadowFeeler(rLight_SampleMaterial, geoms, geomsSize, triangles, &hitGeomId) && hitGeomId == light.id)
						//	{
						//		glm::vec3 colorLight_SampleMaterial = NotSampleLight(light, intersection, lightMaterial, wiLight_SampleMaterial, &pdfLight_SampleMaterial);
						//		pdfLight_SampleMaterial /= lightsSize;
						//		if (pdfLight_SampleMaterial != 0)
						//		{
						//			float weight_SampleMaterial = PowerHeuristic(1, pdfMaterial_SampleMaterial, 1, pdfLight_SampleMaterial);
						//			Li_SampleMaterial = weight_SampleMaterial * colorMaterial_SampleMaterial * colorLight_SampleMaterial / pdfMaterial_SampleMaterial;
						//		}
						//		else
						//		{
						//			//do nothing
						//			Li_SampleMaterial = glm::vec3(0, 0, 0);
						//		}
						//	}
						//	else
						//	{
						//		Li_SampleMaterial = glm::vec3(0, 0, 0);
						//	}
						//}

						// 3. Add together
						Li = Li_SampleLight + Li_SampleMaterial;
					}

					// II. Choose new direction for next depth and update path
					glm::vec3 temp = SampleMaterial(pathSegment.ray, intersection, material, xiii, wiGlobal, &pdfGlobal);

					// III. Early termination
					if (isZero(temp))
					{
						// total internal reflection
						pathSegment.remainingBounces = 0;
					}
					else
					{
						pathSegment.remainingBounces--;
						pathSegment.throughput *= temp * glm::abs(glm::dot(wiGlobal, intersection.surfaceNormal)) / pdfGlobal;
						pathSegment.ray.direction = wiGlobal;
						pathSegment.ray.origin = o;//if transmissive handled differently

						//printf("%f,%f,%f\n", pathSegment.throughput.x, pathSegment.throughput.y, pathSegment.throughput.z);
						
						//in direct light integrator, we don't need early termination
						// 6. When using multi-importance sampling, results are added instead of multiplied, so termination can not be based on whether hit a light or not. Should use ruassian roulette.
						//if (maxBounce - pathSegment.remainingBounces > START_RUASSIAN_ROULETTE_AFTER)
						//{
						//	float russian = u01(rng);
						//	float maxThroughput = glm::max(glm::max(pathSegment.throughput.x, pathSegment.throughput.y), pathSegment.throughput.z);

						//	if (maxThroughput < russian)
						//		pathSegment.remainingBounces = 0;//early termination

						//	pathSegment.throughput /= maxThroughput;
						//}

					}

				}

				// B. No matter whether hit a light or a general geomtry, update the final color
				pathSegment.color += Li * throughput;
				/////////////////////////////////////////////////////////////////////////////////////////

			}
			else
			{
				//If implemented stream compaction, this branch will not be executed - WRONG!!!
				//^^^WRONG!!!WRONG!!!WRONG!!!WRONG!!!WRONG!!!WRONG!!!WRONG!!!WRONG!!!WRONG!!!^^^
				//If implemented stream compaction, this branch will still be executed.
				//Because the order is ComputeIntersection->ShadeMaterial->CompactPath,
				//so when using full light integrator, color should not be set to 0 when
				//no intersection is detected, but should leave it as it is. This way the
				//previously accumulated color would still be valid.

				//pathSegment.color = glm::vec3(0.0f);//this is wrong
				pathSegment.remainingBounces = 0;
			}

			//in direct light integrator, we only do one trace
			pathSegment.remainingBounces = 0;//over write this value no matter what it was
			pathSegments[idx] = pathSegment;
		}
	}
}

__global__ void shadeMaterialDirectLightMIS(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	// My code here
	, const Geom * lights
	, int lightsSize
	, const Geom * geoms
	, int geomsSize
	, Triangle * triangles
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		PathSegment pathSegment = pathSegments[idx];
		if (pathSegment.remainingBounces > 0)
		{
			ShadeableIntersection intersection = shadeableIntersections[idx];

			if (intersection.t > 0.0f)
			{
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
				Material material = materials[intersection.materialId];
				thrust::uniform_real_distribution<float> u01(0.f, 1.f);
				glm::vec2 xi(u01(rng), u01(rng));
				glm::vec2 xii(u01(rng), u01(rng));
				glm::vec2 xiii(u01(rng), u01(rng));
				int randomLightIndex = u01(rng) * lightsSize;//half open
				Geom light = lights[randomLightIndex];
				Material lightMaterial = materials[light.materialid];
				glm::vec3 throughput = pathSegment.throughput;

				/////////////////////////////////////////////////////////////////////////////////////////
				glm::vec3 wiLight_SampleLight(0, 0, 0);
				glm::vec3 wiLight_SampleMaterial(0, 0, 0);
				glm::vec3 wiGlobal(0, 0, 0);
				float pdfGlobal = 1;
				float pdfLight_SampleMaterial = 1;
				float pdfLight_SampleLight = 1;
				float pdfMaterial_SampleMaterial = 1;
				float pdfMaterial_SampleLight = 1;
				glm::vec3 Li(0, 0, 0);
				glm::vec3 Li_SampleLight(0, 0, 0);
				glm::vec3 Li_SampleMaterial(0, 0, 0);
				glm::vec3 colorMaterial_SampleMaterial = SampleMaterial(pathSegment.ray, intersection, material, xi, wiLight_SampleMaterial, &pdfMaterial_SampleMaterial);

				// A. Sample light and material
				if (material.type == 1)// hit a light source
				{
					//!!!!!!!!!!!!!!!!!!!!!!!!!IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!
					//Since your path tracer computes the direct lighting a given 
					//intersection receives as its own term, your path tracer must 
					//not include too much light. This means that every ray which 
					//already computed the direct lighting term should not incorporate 
					//the Le term of the light transport equation into its light 
					//contribution. In other words, unless a particular ray came 
					//directly from the camera or from a perfectly specular surface, 
					//Le should be ignored.

					//in direct light MIS, this is always true
					//if (pathSegment.remainingBounces == maxBounce || pathSegment.hitSpecular)//haven't account for specular bounce //done
					Li = colorMaterial_SampleMaterial;
					pathSegment.remainingBounces = 0;
				}
				else// hit others
				{
					glm::vec3 o = material.type == 3 ?
						intersection.point - MY_OFFSET * intersection.surfaceNormal :
						intersection.point + MY_OFFSET * intersection.surfaceNormal;
					int hitGeomId = -1;

					// I. Compute Li
					if (material.type == 2 || material.type == 3)// Specular
					{
						// 0. Set flag
						pathSegment.hitSpecular = true;
					}
					else// Non specular
					{
						// 0. Set flag
						pathSegment.hitSpecular = false;

						// 1. Sample light
						glm::vec3 colorLight_SampleLight = SampleLight(light, intersection, lightMaterial, xii, wiLight_SampleLight, &pdfLight_SampleLight);
						pdfLight_SampleLight /= lightsSize;
						Ray rLight_SampleLight;
						rLight_SampleLight.origin = o;
						rLight_SampleLight.direction = wiLight_SampleLight;

						//shadow feeler
						if (ShadowFeeler(rLight_SampleLight, geoms, geomsSize, triangles, &hitGeomId) && hitGeomId == light.id)
						{
							//if hit the light, therefore no shadow
							//pathSegment.hitLight = true;
							glm::vec3 colorMaterial_SampleLight = NotSampleMaterial(intersection, material, wiLight_SampleLight, &pdfMaterial_SampleLight);
							colorMaterial_SampleLight *= glm::abs(glm::dot(wiLight_SampleLight, intersection.surfaceNormal));
							if (pdfLight_SampleLight != 0)
							{
								float weight_SampleLight = PowerHeuristic(1, pdfLight_SampleLight, 1, pdfMaterial_SampleLight);
								Li_SampleLight = weight_SampleLight * colorMaterial_SampleLight * colorLight_SampleLight / pdfLight_SampleLight;
							}
							else
							{
								Li_SampleLight = colorMaterial_SampleLight * colorLight_SampleLight;
							}
						}
						else
						{
							//if hit nothing or hit other geom
							//pathSegment.hitLight = false;
							Li_SampleLight = glm::vec3(0, 0, 0);
						}

						// 2. Sample material, since we already sampled one, we are using that result
						if (pdfLight_SampleLight != 0)
						{
							colorMaterial_SampleMaterial *= glm::abs(glm::dot(wiLight_SampleMaterial, intersection.surfaceNormal));
							Ray rLight_SampleMaterial;
							rLight_SampleMaterial.origin = o;
							rLight_SampleMaterial.direction = wiLight_SampleMaterial;

							//shadow feeler
							if (ShadowFeeler(rLight_SampleMaterial, geoms, geomsSize, triangles, &hitGeomId) && hitGeomId == light.id)
							{
								glm::vec3 colorLight_SampleMaterial = NotSampleLight(light, intersection, lightMaterial, wiLight_SampleMaterial, &pdfLight_SampleMaterial);
								pdfLight_SampleMaterial /= lightsSize;
								if (pdfLight_SampleMaterial != 0)
								{
									float weight_SampleMaterial = PowerHeuristic(1, pdfMaterial_SampleMaterial, 1, pdfLight_SampleMaterial);
									Li_SampleMaterial = weight_SampleMaterial * colorMaterial_SampleMaterial * colorLight_SampleMaterial / pdfMaterial_SampleMaterial;
								}
								else
								{
									//do nothing
									Li_SampleMaterial = glm::vec3(0, 0, 0);
								}
							}
							else
							{
								Li_SampleMaterial = glm::vec3(0, 0, 0);
							}
						}

						// 3. Add together
						Li = Li_SampleLight + Li_SampleMaterial;
					}

					// II. Choose new direction for next depth and update path
					glm::vec3 temp = SampleMaterial(pathSegment.ray, intersection, material, xiii, wiGlobal, &pdfGlobal);

					// III. Early termination
					if (isZero(temp))
					{
						// total internal reflection
						pathSegment.remainingBounces = 0;
					}
					else
					{
						pathSegment.remainingBounces--;
						pathSegment.throughput *= temp * glm::abs(glm::dot(wiGlobal, intersection.surfaceNormal)) / pdfGlobal;
						pathSegment.ray.direction = wiGlobal;
						pathSegment.ray.origin = o;//if transmissive handled differently

						//printf("%f,%f,%f\n", pathSegment.throughput.x, pathSegment.throughput.y, pathSegment.throughput.z);
						
						//in direct light MIS, we don't need early termination
						//When using multi-importance sampling, results are added instead of multiplied, so termination can not be based on whether hit a light or not. Should use ruassian roulette.
						//if (maxBounce - pathSegment.remainingBounces > START_RUASSIAN_ROULETTE_AFTER)
						//{
						//	float russian = u01(rng);
						//	float maxThroughput = glm::max(glm::max(pathSegment.throughput.x, pathSegment.throughput.y), pathSegment.throughput.z);

						//	if (maxThroughput < russian)
						//		pathSegment.remainingBounces = 0;//early termination

						//	pathSegment.throughput /= maxThroughput;
						//}

					}

				}

				// B. No matter whether hit a light or a general geomtry, update the final color
				pathSegment.color += Li * throughput;
				/////////////////////////////////////////////////////////////////////////////////////////

			}
			else
			{
				//If implemented stream compaction, this branch will not be executed - WRONG!!!
				//^^^WRONG!!!WRONG!!!WRONG!!!WRONG!!!WRONG!!!WRONG!!!WRONG!!!WRONG!!!WRONG!!!^^^
				//If implemented stream compaction, this branch will still be executed.
				//Because the order is ComputeIntersection->ShadeMaterial->CompactPath,
				//so when using full light integrator, color should not be set to 0 when
				//no intersection is detected, but should leave it as it is. This way the
				//previously accumulated color would still be valid.

				//pathSegment.color = glm::vec3(0.0f);//this is wrong
				pathSegment.remainingBounces = 0;
			}

			//in direct light MIS, we only do one trace
			pathSegment.remainingBounces = 0;//over write this value no matter what it was
			pathSegments[idx] = pathSegment;
		}
	}
}

__global__ void shadeMaterialFullLight(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	// My code here
	, const Geom * lights
	, int lightsSize
	, const Geom * geoms
	, int geomsSize
	, Triangle * triangles
	, int maxBounce
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		PathSegment pathSegment = pathSegments[idx];
		if (pathSegment.remainingBounces > 0)
		{
			ShadeableIntersection intersection = shadeableIntersections[idx];

			if (intersection.t > 0.0f)
			{
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
				Material material = materials[intersection.materialId];
				thrust::uniform_real_distribution<float> u01(0.f, 1.f);
				glm::vec2 xi(u01(rng), u01(rng));
				glm::vec2 xii(u01(rng), u01(rng));
				glm::vec2 xiii(u01(rng), u01(rng));
				int randomLightIndex = u01(rng) * lightsSize;//half open
				Geom light = lights[randomLightIndex];
				Material lightMaterial = materials[light.materialid];
				glm::vec3 throughput = pathSegment.throughput;

				/////////////////////////////////////////////////////////////////////////////////////////
				glm::vec3 wiLight_SampleLight(0, 0, 0);
				glm::vec3 wiLight_SampleMaterial(0, 0, 0);
				glm::vec3 wiGlobal(0, 0, 0);
				float pdfGlobal = 1;
				float pdfLight_SampleMaterial = 1;
				float pdfLight_SampleLight = 1;
				float pdfMaterial_SampleMaterial = 1;
				float pdfMaterial_SampleLight = 1;
				glm::vec3 Li(0, 0, 0);
				glm::vec3 Li_SampleLight(0, 0, 0);
				glm::vec3 Li_SampleMaterial(0, 0, 0);
				glm::vec3 colorMaterial_SampleMaterial = SampleMaterial(pathSegment.ray, intersection, material, xi, wiLight_SampleMaterial, &pdfMaterial_SampleMaterial);

				// A. Sample light and material
				if (material.type == 1)// hit a light source
				{
					//!!!!!!!!!!!!!!!!!!!!!!!!!IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!
					//Since your path tracer computes the direct lighting a given 
					//intersection receives as its own term, your path tracer must 
					//not include too much light. This means that every ray which 
					//already computed the direct lighting term should not incorporate 
					//the Le term of the light transport equation into its light 
					//contribution. In other words, unless a particular ray came 
					//directly from the camera or from a perfectly specular surface, 
					//Le should be ignored.
					if(pathSegment.remainingBounces==maxBounce || pathSegment.hitSpecular)//haven't account for specular bounce //done
						Li = colorMaterial_SampleMaterial;
					pathSegment.remainingBounces = 0;
				}
				else// hit others
				{
					glm::vec3 o = material.type == 3 ?
						intersection.point - MY_OFFSET * intersection.surfaceNormal :
						intersection.point + MY_OFFSET * intersection.surfaceNormal;
					int hitGeomId = -1;
					
					// I. Compute Li
					if (material.type == 2 || material.type == 3)// Specular
					{
						// 0. Set flag
						pathSegment.hitSpecular = true;
					}
					else// Non specular
					{
						// 0. Set flag
						pathSegment.hitSpecular = false;

						// 1. Sample light
						glm::vec3 colorLight_SampleLight = SampleLight(light, intersection, lightMaterial, xii, wiLight_SampleLight, &pdfLight_SampleLight);
						pdfLight_SampleLight /= lightsSize;
						Ray rLight_SampleLight;
						rLight_SampleLight.origin = o;
						rLight_SampleLight.direction = wiLight_SampleLight;

						//shadow feeler
						if (ShadowFeeler(rLight_SampleLight, geoms, geomsSize, triangles, &hitGeomId) && hitGeomId == light.id)
						{
							//if hit the light, therefore no shadow
							//pathSegment.hitLight = true;
							glm::vec3 colorMaterial_SampleLight = NotSampleMaterial(intersection, material, wiLight_SampleLight, &pdfMaterial_SampleLight);
							colorMaterial_SampleLight *= glm::abs(glm::dot(wiLight_SampleLight, intersection.surfaceNormal));
							if (pdfLight_SampleLight != 0)
							{
								float weight_SampleLight = PowerHeuristic(1, pdfLight_SampleLight, 1, pdfMaterial_SampleLight);
								Li_SampleLight = weight_SampleLight * colorMaterial_SampleLight * colorLight_SampleLight / pdfLight_SampleLight;
							}
							else
							{
								Li_SampleLight = colorMaterial_SampleLight * colorLight_SampleLight;
							}
						}
						else
						{
							//if hit nothing or hit other geom
							//pathSegment.hitLight = false;
							Li_SampleLight = glm::vec3(0, 0, 0);
						}

						// 2. Sample material, since we already sampled one, we are using that result
						if (pdfLight_SampleLight != 0)
						{
							colorMaterial_SampleMaterial *= glm::abs(glm::dot(wiLight_SampleMaterial, intersection.surfaceNormal));
							Ray rLight_SampleMaterial;
							rLight_SampleMaterial.origin = o;
							rLight_SampleMaterial.direction = wiLight_SampleMaterial;

							//shadow feeler
							if (ShadowFeeler(rLight_SampleMaterial, geoms, geomsSize, triangles, &hitGeomId) && hitGeomId == light.id)
							{
								glm::vec3 colorLight_SampleMaterial = NotSampleLight(light, intersection, lightMaterial, wiLight_SampleMaterial, &pdfLight_SampleMaterial);
								pdfLight_SampleMaterial /= lightsSize;
								if (pdfLight_SampleMaterial != 0)
								{
									float weight_SampleMaterial = PowerHeuristic(1, pdfMaterial_SampleMaterial, 1, pdfLight_SampleMaterial);
									Li_SampleMaterial = weight_SampleMaterial * colorMaterial_SampleMaterial * colorLight_SampleMaterial / pdfMaterial_SampleMaterial;
								}
								else
								{
									//do nothing
									Li_SampleMaterial = glm::vec3(0, 0, 0);
								}
							}
							else
							{
								Li_SampleMaterial = glm::vec3(0, 0, 0);
							}
						}

						// 3. Add together
						Li = Li_SampleLight + Li_SampleMaterial;
					}

					// II. Choose new direction for next depth and update path
					glm::vec3 temp = SampleMaterial(pathSegment.ray, intersection, material, xiii, wiGlobal, &pdfGlobal);

					// III. Early termination
					if (isZero(temp))
					{
						// total internal reflection
						pathSegment.remainingBounces = 0;
					}
					else
					{
						pathSegment.remainingBounces--;
						pathSegment.throughput *= temp * glm::abs(glm::dot(wiGlobal, intersection.surfaceNormal)) / pdfGlobal;
						pathSegment.ray.direction = wiGlobal;
						pathSegment.ray.origin = o;//if transmissive handled differently

						//printf("%f,%f,%f\n", pathSegment.throughput.x, pathSegment.throughput.y, pathSegment.throughput.z);
						// 6. When using multi-importance sampling, results are added instead of multiplied, so termination can not be based on whether hit a light or not. Should use ruassian roulette.
						if (maxBounce - pathSegment.remainingBounces > START_RUASSIAN_ROULETTE_AFTER)
						{
							float russian = u01(rng);
							float maxThroughput = glm::max(glm::max(pathSegment.throughput.x, pathSegment.throughput.y), pathSegment.throughput.z);

							if (maxThroughput < russian)
								pathSegment.remainingBounces = 0;//early termination

							pathSegment.throughput /= maxThroughput;
						}

					}

				}

				// B. No matter whether hit a light or a general geomtry, update the final color
				pathSegment.color += Li * throughput;
				/////////////////////////////////////////////////////////////////////////////////////////

			}
			else
			{
				//If implemented stream compaction, this branch will not be executed - WRONG!!!
				//^^^WRONG!!!WRONG!!!WRONG!!!WRONG!!!WRONG!!!WRONG!!!WRONG!!!WRONG!!!WRONG!!!^^^
				//If implemented stream compaction, this branch will still be executed.
				//Because the order is ComputeIntersection->ShadeMaterial->CompactPath,
				//so when using full light integrator, color should not be set to 0 when
				//no intersection is detected, but should leave it as it is. This way the
				//previously accumulated color would still be valid.

				//pathSegment.color = glm::vec3(0.0f);//this is wrong
				pathSegment.remainingBounces = 0;
			}

			pathSegments[idx] = pathSegment;
		}
	}
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */


void ComputeIntersection(dim3 gridSize, int blockSize1d, int depth, int num_paths, PathSegment * paths, Geom * geoms, Triangle * _triangles, Material * materials, int geomsSize, ShadeableIntersection * intersections, int * intersections_material_id)
{
	// tracing
	computeIntersections << <gridSize, blockSize1d >> > (depth, num_paths, paths, geoms, _triangles, materials, geomsSize, intersections, intersections_material_id);
	checkCUDAError("trace one bounce");
	cudaDeviceSynchronize();
}

void BatchMaterialOne(int num_paths, PathSegment * paths, ShadeableIntersection * intersections, int * intersections_material_id)
{
	//arrage dev_intersections and dev_paths so that intersections with same materials stay together
	auto it = thrust::make_zip_iterator(thrust::make_tuple(intersections, paths));
	thrust::device_ptr<int> dev_intersections_material_id_ptr(intersections_material_id);
	thrust::sort_by_key(thrust::device, dev_intersections_material_id_ptr, dev_intersections_material_id_ptr + num_paths, it);
}

void BatchMaterialTwo(int num_paths, PathSegment * paths, ShadeableIntersection * intersections)
{
	//arrage dev_intersections and dev_paths so that intersections with same materials stay together
	thrust::device_ptr<ShadeableIntersection> dev_intersection_ptr(intersections);
	thrust::device_ptr<PathSegment> dev_paths_ptr(paths);
	thrust::sort_by_key(thrust::device, dev_intersection_ptr, dev_intersection_ptr + num_paths, dev_paths_ptr);
}

void ShadePathNaive(dim3 gridSize, int blockSize1d, int iter, int num_paths, ShadeableIntersection * intersections, PathSegment * paths,  Material * materials)
{
	//My code here
	//naive integrator
	shadeMaterialNaive << <gridSize, blockSize1d >> > (iter, num_paths, intersections, paths, materials);
	//shadeMaterialOld << <gridSize, blockSize1d >> > (iter, num_paths, intersections, paths, materials);
	cudaDeviceSynchronize();
}

void ShadePathDirectLight(dim3 gridSize, int blockSize1d, int iter, int num_paths, ShadeableIntersection * intersections, PathSegment * paths, Material * materials, Geom * lights, int lightsSize, Geom * geoms, int geomsSize, Triangle * triangles)
{
	//My code here
	//direct light integrator
	shadeMaterialDirectLight << <gridSize, blockSize1d >> > (iter, num_paths, intersections, paths, materials, lights, lightsSize, geoms, geomsSize, triangles);
	cudaDeviceSynchronize();
}

void ShadePathDirectLightMIS(dim3 gridSize, int blockSize1d, int iter, int num_paths, ShadeableIntersection * intersections, PathSegment * paths, Material * materials, Geom * lights, int lightsSize, Geom * geoms, int geomsSize, Triangle * triangles)
{
	//My code here
	//direct light mis integrator
	shadeMaterialDirectLightMIS << <gridSize, blockSize1d >> > (iter, num_paths, intersections, paths, materials, lights, lightsSize, geoms, geomsSize, triangles);
	cudaDeviceSynchronize();
}

void ShadePathFullLight(dim3 gridSize, int blockSize1d, int iter, int num_paths, ShadeableIntersection * intersections, PathSegment * paths, Material * materials, Geom * lights, int lightsSize, Geom * geoms, int geomsSize, Triangle * triangles, int maxBounce)
{
	//My code here
	//full light integrator
	shadeMaterialFullLight << <gridSize, blockSize1d >> > (iter, num_paths, intersections, paths, materials, lights, lightsSize, geoms, geomsSize, triangles, maxBounce);
	cudaDeviceSynchronize();
}

void CompactPath(dim3 gridSize, int blockSize1d, int * num_paths, glm::vec3 * image, int * paths_exist, int * paths_indices, PathSegment ** paths, PathSegment ** paths_temp)
{
	//stream compaction here
	//use dev_paths_exist to stream compact dev_paths
	kernelMapToBooleanAndGather << <gridSize, blockSize1d >> > (*num_paths, image, paths_exist, *paths);
	cudaDeviceSynchronize();

	//use inclusive scan so that it is eaiser to get the total number of path
	//this will require -1 when using the indices array
	thrust::inclusive_scan(thrust::device_pointer_cast<int>(paths_exist), thrust::device_pointer_cast<int>(paths_exist) + *num_paths, thrust::device_pointer_cast<int>(paths_indices));
	int num_paths_temp = *num_paths;
	cudaMemcpy(&num_paths_temp, dev_paths_indices + *num_paths - 1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	kernelScatter << <gridSize, blockSize1d >> > (*num_paths, *paths_temp, *paths, paths_exist, paths_indices);
	cudaDeviceSynchronize();

	PathSegment* temp = *paths;
	*paths = *paths_temp;
	*paths_temp = temp;
	*num_paths = num_paths_temp;

	//printf("%d\n", num_paths);
}

void pathtrace(uchar4 *pbo, int frame, int iter,
	bool enable_stream_compact,
	int enable_material_batching,
	bool enable_cache_first_path,
	int integrator_type) 
{
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d((cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x, (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	if (integrator_type == 1 || integrator_type == 2 || integrator_type == 3)//direct light mis integrator and full light integrator, adding results instead of multiplying results
	{
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, glm::vec3(0, 0, 0));
	}
	else
	{
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, glm::vec3(1, 1, 1));
	}
	checkCUDAError("generate camera ray");

	int depth = 0;

	int num_paths = pixelcount;

	int geomSize = hst_scene->geoms.size();

	int lightSize = hst_scene->lights.size();

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;

	while (!iterationComplete) {
		depth++;

		dim3 gridSize1d = (num_paths + blockSize1d - 1) / blockSize1d;

		if (depth == 1 && enable_cache_first_path && paths_cached)
		{
			// load first paths
			cudaMemcpy(dev_paths, dev_paths_cache, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
			cudaMemcpy(dev_intersections, dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else
		{
			// clean shading chunks
			cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
			// compute intersections
			ComputeIntersection(gridSize1d, blockSize1d, depth, num_paths, dev_paths, dev_geoms, dev_triangles, dev_materials, geomSize, dev_intersections, dev_intersections_material_id);
			// batch materials
			if (enable_material_batching == 1)
			{
				//using an extra int array and iterator of a tuple, supposed to use radix sort
				BatchMaterialOne(num_paths, dev_paths, dev_intersections, dev_intersections_material_id);
			}
			else if (enable_material_batching == 2)
			{
				//using an struct comparison, supposed to use merge sort
				BatchMaterialTwo(num_paths, dev_paths, dev_intersections);
			}

			// cache first paths
			if (depth == 1 && enable_cache_first_path && !paths_cached)
			{
				cudaMemcpy(dev_paths_cache, dev_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
				cudaMemcpy(dev_intersections_cache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
				paths_cached = true;
			}
		}

		// shade paths
		if (integrator_type == 0)
		{
			//naive integrator
			ShadePathNaive(gridSize1d, blockSize1d, iter, num_paths, dev_intersections, dev_paths, dev_materials);
		}
		else if (integrator_type == 1)
		{
			//direct light integrator
			ShadePathDirectLight(gridSize1d, blockSize1d, iter, num_paths, dev_intersections, dev_paths, dev_materials, dev_lights, lightSize, dev_geoms, geomSize, dev_triangles);
		}
		else if (integrator_type == 2)
		{
			//direct light mis integrator
			ShadePathDirectLightMIS(gridSize1d, blockSize1d, iter, num_paths, dev_intersections, dev_paths, dev_materials, dev_lights, lightSize, dev_geoms, geomSize, dev_triangles);
		}
		else if (integrator_type == 3)
		{
			//full light integrator
			ShadePathFullLight(gridSize1d, blockSize1d, iter, num_paths, dev_intersections, dev_paths, dev_materials, dev_lights, lightSize, dev_geoms, geomSize, dev_triangles, traceDepth);
		}

		// compact paths
		if (enable_stream_compact)
		{
			CompactPath(gridSize1d, blockSize1d, &num_paths, dev_image, dev_paths_exist, dev_paths_indices, &dev_paths, &dev_paths_temp);
		}

		// terminate this iteration
		if (depth > traceDepth || num_paths <= 0) iterationComplete = true;
	}

	// Assemble this iteration and apply it to the image
	if (!enable_stream_compact)
	{
		dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
		finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);
	}

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
