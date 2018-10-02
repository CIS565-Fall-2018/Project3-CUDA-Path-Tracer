#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "bsdf.cu"
#include "shapes.cu"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)


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
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

#define PDF_EPSILON 0.001
#define RAY_EPSILON 0.0005f

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static ShadeableIntersection* dev_cached_intersections = NULL;

// TODO: static variables for device memory, any extra info you need, etc
// ...

struct RayIntersectionPredicate
{
	__host__ __device__ bool operator()(const ShadeableIntersection& shadeable_intersection)
	{

		return (shadeable_intersection.t == -1);
	}
};

struct RayTerminatePredicate
{
	RayTerminatePredicate() {}

	__host__ __device__ bool operator()(const PathSegment& path_segment)
	{

		return (path_segment.remainingBounces > 0);
	}
};

struct MaterialPredicate
{

	MaterialPredicate() {}

	__host__ __device__ bool operator()(const ShadeableIntersection& first, const ShadeableIntersection& second)
	{
		return (first.materialId < second.materialId);
	}
};

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

	cudaMalloc(&dev_cached_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_cached_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
	cudaFree(dev_cached_intersections);
    // TODO: clean up any extra device memory you created

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) 
	{
		const int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, pathSegments[index].remainingBounces);
		thrust::uniform_real_distribution<float> u01(0, 2.f);

		const float jitterX = u01(rng);// (2.f * tanf(cam.fov.x / 2.f) * u1(rng) / cam.resolution.x);
		const float jitterY = u01(rng);// (2.f * tanf(cam.fov.y / 2.f) * u2(rng) / cam.resolution.y);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * ((cam.pixelLength.x * (jitterX + (float)x - (float)cam.resolution.x * 0.5f)))
			- cam.up * cam.pixelLength.y * (jitterY + (float)y - (float)cam.resolution.y * 0.5f)
			);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
		segment.isRayDead = false;
		segment.isRefractedRay = false;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection* intersections
	, ShadeableIntersection* cacheIntersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment& pathSegment = pathSegments[path_index];

		float t;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		const bool refractedRay = pathSegment.isRefractedRay;

		ShadeableIntersection intersection = intersections[path_index];

		// naive parse through global geoms

		// TODO : Clean this up
		ShadeableIntersection tempIntersection = intersection;
		ShadeableIntersection hitIntersection = tempIntersection;

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];
			t = 0.0f;

			if (geom.type == CUBE)
			{
				t = Shapes::Cube::TestIntersection(&geom, pathSegment.ray, &tempIntersection, refractedRay);
			}
			else if (geom.type == SPHERE)
			{
				t = Shapes::Sphere::TestIntersection(&geom, pathSegment.ray, &tempIntersection, refractedRay);
			}
			else if (geom.type == PLANE)
			{
				t = Shapes::SquarePlane::TestIntersection(&geom, pathSegment.ray, &tempIntersection, refractedRay);
			}
			else if (geom.type == IMPLICIT)
			{
				t = Shapes::Implicit::TestIntersection(&geom, &pathSegment.ray, &tempIntersection, refractedRay);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				hitIntersection = tempIntersection;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
			intersections[path_index].m_didIntersect = false;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;

			intersections[path_index].m_surfaceNormal = hitIntersection.m_surfaceNormal;
			intersections[path_index].m_surfaceTangent = hitIntersection.m_surfaceTangent;
			intersections[path_index].m_surfaceBiTangent = hitIntersection.m_surfaceBiTangent;

			intersections[path_index].m_didIntersect = true;
			intersections[path_index].m_intersectionPointWorld = hitIntersection.m_intersectionPointWorld;
			intersections[path_index].m_tangentToWorld = hitIntersection.m_tangentToWorld;
			intersections[path_index].m_worldToTangent = hitIntersection.m_worldToTangent;
		}

		if(depth == 0)
		{
			cacheIntersections[path_index] = intersections[path_index];
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial (
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
    ShadeableIntersection& intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f && pathSegments[idx].remainingBounces > 0) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
        float lightTerm = glm::dot(intersection.m_surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        pathSegments[idx].color *= u01(rng); // apply some noise because why not
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
    }
  }
}

/*
 * NaiveIntegratorShader
 */
__global__ void NaiveIntegratorShader(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < num_paths)
	{
		ShadeableIntersection& intersection = shadeableIntersections[idx];

		if (intersection.t > 0.0f)
		{

			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			const glm::vec3 materialColor = material.color;

			glm::vec3 finalColor = (materialColor * material.emittance);

			// If the material indicates that the object was a light, "light" the ray
			if(material.emittance > 0.0f || pathSegments[idx].remainingBounces <= 0)
			{
				finalColor *= pathSegments[idx].color;

				pathSegments[idx].color = finalColor;
				pathSegments[idx].remainingBounces = 0;
				pathSegments[idx].isRayDead = true;
			}
			else
			{
				// 1. Calculate WoW
				const glm::vec3 woW = -pathSegments[idx].ray.direction;
				glm::vec3 wiW(0.f);
				glm::vec2 xi(u01(rng), u01(rng));

				float pdf = 1.f;

				// 2. Get the wiw and pdf from the given material
				const glm::vec3 sample_f_color = BSDF::Sample_F(woW, &wiW, &pdf, &xi, &material, &intersection);

				pathSegments[idx].isRefractedRay = material.hasRefractive;

				if(pdf < PDF_EPSILON)
				{
					pathSegments[idx].color = glm::vec3(1.f);
					pathSegments[idx].remainingBounces = 0;
					pathSegments[idx].isRayDead = true;
					return;
				}

				const float dotProduct = glm::dot(wiW, intersection.m_surfaceNormal);
				const float lambertsTerm = glm::abs(dotProduct);

				// 3. Add the color to the path sement 
				// color *= (sample_f * lamberts) / pdf
				pathSegments[idx].color *= (sample_f_color * lambertsTerm) / pdf;

				// 4. Update the ray direction and remove one bounce from path segment
				const glm::vec3 originOffset = intersection.m_surfaceNormal * RAY_EPSILON * (dotProduct < 0 ? -1.f : 1.f);

				pathSegments[idx].ray.origin = intersection.m_intersectionPointWorld + originOffset;
				pathSegments[idx].ray.direction = wiW;
				pathSegments[idx].remainingBounces--;
			}
		}
		else 
		{
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
			pathSegments[idx].isRayDead = true;
		}
	}
}

/*
* DirectLightingShader
*/
__global__ void DirectLightingShader(
	int iter
	, int num_paths
	, int num_geoms
	, int num_lights
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Geom* allGeometry
	, Geom* lightGeometry
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_paths)
	{
		ShadeableIntersection& intersection = shadeableIntersections[idx];

		if (intersection.t > 0.0f)
		{
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			const glm::vec3 materialColor = material.color;

			glm::vec3 finalColor = (materialColor * material.emittance);

			// If the material indicates that the object was a light, "light" the ray
			if(material.emittance > 0.0f || pathSegments[idx].remainingBounces <= 0 || num_lights == 0)
			{
				finalColor *= pathSegments[idx].color;

				pathSegments[idx].color = finalColor;
				pathSegments[idx].remainingBounces = 0;
				pathSegments[idx].isRayDead = true;
			}
			else
			{
				// 1. Initialize all the shit
				const glm::vec3 woW = -pathSegments[idx].ray.direction;
				glm::vec3 wiW_light(0.f);
				glm::vec3 wiW_bsdf(0.f);

				glm::vec2 xi_light(u01(rng), u01(rng));
				glm::vec2 xi_bsdf(u01(rng), u01(rng));

				int randomLight = int(u01(rng) * num_lights);

				float pdf_light = 1.f;
				float pdf_bsdf = 1.f;

				Geom random_light = lightGeometry[randomLight];


				/*// 2. Get the wiw and pdf from the given material
				const glm::vec3 sample_f_color = BSDF::Sample_F(woW, &wiW, &pdf, &xi, &material, &intersection);

				pathSegments[idx].isRefractedRay = material.hasRefractive;

				if(pdf < PDF_EPSILON)
				{
					pathSegments[idx].color = glm::vec3(1.f);
					pathSegments[idx].remainingBounces = 0;
					pathSegments[idx].isRayDead = true;
					return;
				}

				const float dotProduct = glm::dot(wiW, intersection.m_surfaceNormal);
				const float lambertsTerm = glm::abs(dotProduct);

				// 3. Add the color to the path sement 
				// color *= (sample_f * lamberts) / pdf
				pathSegments[idx].color *= (sample_f_color * lambertsTerm) / pdf;

				// 4. Update the ray direction and remove one bounce from path segment
				const glm::vec3 originOffset = intersection.m_surfaceNormal * RAY_EPSILON * (dotProduct < 0 ? -1.f : 1.f);

				pathSegments[idx].ray.origin = intersection.m_intersectionPointWorld + originOffset;
				pathSegments[idx].ray.direction = wiW;
				pathSegments[idx].remainingBounces--;*/
			}
		}
		else 
		{
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
			pathSegments[idx].isRayDead = true;
		}
	}
}



// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, int totalIterations, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;// / float(totalIterations));
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter, int totalIterations) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

    generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	const int num_paths = dev_path_end - dev_paths;
	int curr_paths = num_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) 
	{
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (curr_paths + blockSize1d - 1) / blockSize1d;

		const bool useCachedIntersection = (depth == 0 && iter != 1);

		if(!useCachedIntersection)
		{
			computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
				depth
				, curr_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, dev_cached_intersections
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}
		depth++;

		// Sort by Material
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + curr_paths, dev_paths, MaterialPredicate());

		// 1. Do stream  and remove rays that have no intersection.
		// Not needed for now

		// 2. Perform Color calculation using BSDF for Naive.
		NaiveIntegratorShader << < numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			curr_paths,
			(useCachedIntersection ? dev_cached_intersections : dev_intersections),
			dev_paths,
			dev_materials
			);

		// 3. Remove any rays that have reached maximum bounces.
		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + curr_paths, RayTerminatePredicate());
		cudaDeviceSynchronize();
		curr_paths = (dev_path_end - dev_paths);
		
		// This should be based on result of (3).
		iterationComplete = (depth > traceDepth || curr_paths <= 0);
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, totalIterations, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
