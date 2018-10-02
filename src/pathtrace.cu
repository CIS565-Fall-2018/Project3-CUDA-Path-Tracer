#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include <ctime>

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

int directLighting = 0;
int caching = 0;
int sortMaterials = 1;

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

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static ShadeableIntersection * dev_cached_intersections = NULL;
static ShadeableIntersection * shadow_intersections = NULL;
static PathSegment * shadow_paths = NULL;
static ShadeableIntersection * next_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

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

	cudaMalloc(&shadow_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(shadow_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&shadow_paths, pixelcount * sizeof(PathSegment));
	cudaMemset(shadow_paths, 0, pixelcount * sizeof(PathSegment));

	cudaMalloc(&next_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(next_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, int caching)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		thrust::default_random_engine rngX = makeSeededRandomEngine(x, y, iter);
		thrust::default_random_engine rngY = makeSeededRandomEngine(y, x, iter);
		thrust::uniform_real_distribution<float> u01(0, 1);
		float jitterX = u01(rngX);
		float jitterY = u01(rngY);

		if (caching == 1) {
			jitterX = 0;
			jitterY = 0;
		}

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- (cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f) + (cam.right * cam.pixelLength.x * jitterX))
			- (cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f) + (cam.up * cam.pixelLength.y * jitterY))
			);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
		segment.alive = true;
		segment.hitLight = false;
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
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment &pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == TRIANGLE)
			{
				t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
			intersections[path_index].geomId = -1;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].intersectionPoint = intersect_point;
			intersections[path_index].geomId = hit_geom_index;
		}

		pathSegment.remainingBounces--;
	}
}

__global__ void evaluateBSDFs(int num_paths, ShadeableIntersection * shadeableIntersections, PathSegment * pathSegments, Material * materials) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_paths) {
		ShadeableIntersection &intersection = shadeableIntersections[idx];
		PathSegment &pathSegment = pathSegments[idx];

		// First check if this path is still alive
		if (intersection.t < 0 || pathSegment.remainingBounces <= 0 || materials[intersection.materialId].emittance > 0.0f) {
			pathSegment.alive = false;
		}

		// Now evaluate the color from this intersection if it hit something
		if (intersection.t >= 0) {
			Material &material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			if (material.emittance > 0.0f) {
				pathSegment.color *= (materialColor * material.emittance);
				pathSegment.hitLight = true;
			}
			else {
				/*if (intersection.materialId == 4) {
					float R0 = ((1.55 - 1.0) / (1.55 + 1.0)) * ((1.55 - 1.0) / (1.55 + 1.0));
					float Rtheta = R0 + (1 - R0) * (1 - powf(glm::dot(pathSegment.ray.direction, intersection.surfaceNormal), 5));
					pathSegment.color *= materialColor * Rtheta;
				}
				else {*/
					pathSegment.color *= materialColor;
				//}
			}
		}
		else {
			pathSegment.color = glm::vec3(0, 0, 0);
		}
	}
}

__global__ void computeNewRays(int iter, int frame, int num_paths, ShadeableIntersection * shadeableIntersections, PathSegment * pathSegments, Material * materials) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_paths) {
		PathSegment &pathSegment = pathSegments[idx];

		if (pathSegment.alive) {
			ShadeableIntersection &intersection = shadeableIntersections[idx];
			Material &material = materials[intersection.materialId];

			thrust::default_random_engine rng = makeSeededRandomEngine(frame, idx, iter);

			if (intersection.materialId == 4) {
				//diffuseScatterRay(pathSegment, intersection.intersectionPoint, intersection.surfaceNormal, material, rng);
				//refractScatterRay(pathSegment, intersection.intersectionPoint, pathSegment.ray.direction, intersection.surfaceNormal, material, rng);
				reflectScatterRay(pathSegment, intersection.intersectionPoint, pathSegment.ray.direction, intersection.surfaceNormal, material, rng);
			}
			else {
				diffuseScatterRay(pathSegment, intersection.intersectionPoint, intersection.surfaceNormal, material, rng);
			}
		}
	}
}

// Add the current iteration's output to the overall image if this path is now dead
__global__ void gatherColors(int num_paths, glm::vec3 * image, PathSegment * pathSegments)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < num_paths) {
		PathSegment &pathSegment = pathSegments[idx];

		if (pathSegment.hitLight) {
			image[pathSegment.pixelIndex] += pathSegment.color;
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
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
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
        float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
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

struct cullpredicate
{
	__device__
		bool operator()(const PathSegment &pathSegment)
	{
		return pathSegment.alive;
	}
};

/*
 * Direct lighting methods
 */

//__global__ void directLightingKernel(int iteration, int frame, int num_paths, ShadeableIntersection * shadeableIntersections, PathSegment * pathSegments, Material * materials, Geom * geoms, int geoms_size, glm::vec3 *image) {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (idx < num_paths) {
//		PathSegment &pathSegment = pathSegments[idx];
//		ShadeableIntersection &intersection = shadeableIntersections[idx];
//
//		if (intersection.t > 0) {
//			Material &material = materials[intersection.materialId];
//
//			// Attempt intersection with the area light on the ceiling
//			glm::vec3 randomPointOnLight = glm::vec3(0, 8.5, 0);
//			float t = boxIntersectionTest(geoms[0], pathSegment.ray, tmp_intersect, tmp_normal, outside);
//		}
//	}
//}

__global__ void createShadowPaths(int num_paths, ShadeableIntersection *intersections, PathSegment *shadowPaths) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_paths) {
		glm::vec3 randomPointOnLight = glm::vec3(0, 8.5, 0);

		if (intersections[idx].t > 0) {
			shadowPaths[idx].ray.direction = randomPointOnLight - intersections[idx].intersectionPoint;
			shadowPaths[idx].ray.origin = intersections[idx].intersectionPoint;
		}
	}
}

__global__ void colorBasedOnShadowIntersections(int num_paths, ShadeableIntersection *shadowIntersections, ShadeableIntersection *dev_intersections, Material * materials, PathSegment *pathSegments, glm::vec3 *image) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_paths) {
		PathSegment &pathSegment = pathSegments[idx];

		if (shadowIntersections[idx].t <= 0 || shadowIntersections->geomId != 0) {
			image[pathSegments->pixelIndex] += glm::vec3(1, 1, 1);
		}
		else {
			image[pathSegments->pixelIndex] += glm::vec3(1, 1, 1);
		}
	}
}

__global__ void colorEveryPixel(int num_paths, glm::vec3 *image, PathSegment *pathSegments) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_paths) {
		image[idx] = glm::vec3(255.0f, 0.0f, 0.0f);
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 256;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, frame, traceDepth, dev_paths, caching);
	checkCUDAError("generate camera ray");

	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = pixelcount;
	int iteration = 0;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	if (directLighting == 1) {
		// clean shading chunks
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

		cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));
		cudaMemset(shadow_intersections, 0, num_paths * sizeof(ShadeableIntersection));
		cudaMemset(shadow_paths, 0, num_paths * sizeof(PathSegment));

		// tracing
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (iteration, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();

		// Create all the shadow paths
		createShadowPaths << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, shadow_paths);
		checkCUDAError("create shadow paths");
		cudaDeviceSynchronize();

		// Test these shadow paths with the scene
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (iteration, num_paths, shadow_paths, dev_geoms, hst_scene->geoms.size(), shadow_intersections);
		checkCUDAError("test shadow paths with scene");
		cudaDeviceSynchronize();

		// For all that did not hit a light color that pixel black
		colorBasedOnShadowIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, shadow_intersections, dev_intersections, dev_materials, dev_paths, dev_image);
		cudaDeviceSynchronize();

	}
	else {

		bool iterationComplete = false;

		while (!iterationComplete) {
			dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

			if (caching == 1) {

				// clean shading chunks
				if (frame == 1 || iteration > 0) {
					cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

					// tracing
					computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (iteration, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections);
					checkCUDAError("trace one bounce");
					cudaDeviceSynchronize();
				}
				else {
					cudaMemcpy(dev_intersections, dev_cached_intersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
				}

				if (frame == 1 && iteration == 0) {
					cudaMemcpy(dev_cached_intersections, dev_intersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
				}
			}
			else {
				cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

				// tracing
				computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (iteration, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections);
				checkCUDAError("trace one bounce");
				cudaDeviceSynchronize();
			}

			// Evaluate the bsdfs for each path segments based on the intersections and accumulate the color
			evaluateBSDFs << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_materials);
			checkCUDAError("evaluate bsdfs");
			cudaDeviceSynchronize();

			// Generate new ray for each path segment based on the intersection for each path that is still alive
			computeNewRays << <numblocksPathSegmentTracing, blockSize1d >> > (iteration, frame, num_paths, dev_intersections, dev_paths, dev_materials);
			checkCUDAError("compute new rays");
			cudaDeviceSynchronize();

			// Add to the final image all the rays that have hit a light, use their accumulated color
			gatherColors << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_image, dev_paths);
			checkCUDAError("add finished rays colors to image");
			cudaDeviceSynchronize();

			// Cull out all the rays that either didnt hit anything, have reached max depth, or hit a light, based on alive boolean
			PathSegment *end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, cullpredicate());
			cudaDeviceSynchronize();

			num_paths = end - dev_paths;

			if (num_paths <= 0) {
				iterationComplete = true;
			}
			iteration++;

			// TODO:
			// --- Shading Stage ---
			// Shade path segments based on intersections and generate new rays by
			// evaluating the BSDF.
			// Start off with just a big kernel that handles all the different
			// materials you have in the scenefile.
			// TODO: compare between directly shading the path segments and shading
			// path segments that have been reshuffled to be contiguous in memory.


			 // TODO: should be based off stream compaction results.
		}
	}

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, frame, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
