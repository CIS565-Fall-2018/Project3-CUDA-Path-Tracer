#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <chrono>
#include <ctime>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "../stream_compaction/efficient.h"

#define ERRORCHECK 1
#define CACHE_FIRST_ITERATION 1

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

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

__host__ __device__
void concentricSampleDisk(float* newX, float* newY, thrust::default_random_engine &rng)
{
  // get the sample
  thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
  float x = u01(rng);
  float y = u01(rng);

  // remap to -1 to 1
  float xOffset = 2.f * x - 1.f;
  float yOffset = 2.f * y - 1.f;

  if (xOffset == 0 && yOffset == 0)
  {
    *newX = xOffset;
    *newY = yOffset;
  }

  float theta, r;
  if (std::abs(xOffset) > std::abs(yOffset))
  {
    r = xOffset;
    theta = (PI / 4.f) * (yOffset / xOffset);
  }
  else 
  {
    r = yOffset;
    theta = (PI / 2.f) - ((PI / 4.f) * (xOffset / yOffset));
  }

  *newX = r * std::cos(theta);
  *newY = r * std::sin(theta);
}

__host__ __device__
void modifyRayForDepthofField(Ray* ray, float aperture, float focalDist, thrust::default_random_engine &rng)
{
  float lensX, lensY;

  concentricSampleDisk(&lensX, &lensY, rng);
  lensX *= aperture;
  lensY *= aperture;
  
  float ft = focalDist / fabs(ray->direction.z);
  glm::vec3 pFocus = getPointOnRay((*ray), ft);
  ray->origin += glm::vec3(lensX, lensY, 0.0f);
  ray->direction = glm::normalize(pFocus - ray->origin);
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
static PathSegment * dev_paths_first_iter_cache = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static PathSegment ** dev_paths_ptrs = NULL;
static int * dev_material_ids = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_paths_ptrs, pixelcount * sizeof(PathSegment*));
    if (CACHE_FIRST_ITERATION) { cudaMalloc(&dev_paths_first_iter_cache, pixelcount * sizeof(PathSegment)); }

/*    for (Geom g : scene->geoms)
    {
      if (g.numTriangles > 0)
      {
 //       cudaMalloc(&g.dev_triangles, g.numTriangles * sizeof(Triangle));
 //       checkCUDAError("malloc triangles");
//
 //       cudaMemcpy(g.dev_triangles, g.triangles, g.numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
 //       checkCUDAError("rip memcpy tris");

      }
    }
    */
    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

  	cudaMalloc(&dev_material_ids, pixelcount * sizeof(int));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_paths_ptrs);
    if (CACHE_FIRST_ITERATION) { cudaFree(dev_paths_first_iter_cache); }
    cudaFree(dev_material_ids);

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
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

    segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);

    modifyRayForDepthofField(&segment.ray, 0.5, 10, rng);
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment ** pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment* pathSegment_ptr = pathSegments[path_index];

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
				t = boxIntersectionTest(geom, pathSegment_ptr->ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment_ptr->ray, tmp_intersect, tmp_normal, outside);
			}
      else if (geom.type == SWORD)
      {
 //       float temp_t = boxIntersectionTest(geom, pathSegment_ptr->ray, tmp_intersect, tmp_normal, outside);
 //       if (temp_t > 0.0f && t_min > temp_t && outside)
//        {
          glm::vec3 baryPosition;
          for (int j = 0; j < geom.numTriangles; ++j)
          {
            if (glm::intersectRayTriangle(pathSegment_ptr->ray.origin,
              pathSegment_ptr->ray.direction,
              geom.dev_triangles[j].v1,
              geom.dev_triangles[j].v2,
              geom.dev_triangles[j].v3,
              baryPosition))
            {
              tmp_normal = geom.dev_triangles[j].n;
              tmp_intersect = baryPosition;
              t = baryPosition.z;
              break;
            }
          }
 //       }
      }
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
        if (outside)
        {
          t_min = t;
          hit_geom_index = i;
          intersect_point = tmp_intersect;
          normal = tmp_normal;
        }
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[pathSegment_ptr->pixelIndex].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[pathSegment_ptr->pixelIndex].t = t_min;
			intersections[pathSegment_ptr->pixelIndex].materialId = geoms[hit_geom_index].materialid;
			intersections[pathSegment_ptr->pixelIndex].surfaceNormal = normal;
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
  , int depth
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment ** pathSegments
	, Material * materials
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[pathSegments[idx]->pixelIndex];
    if (intersection.t > 0.0f && pathSegments[idx]->remainingBounces > 0) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
//      long ms = std::chrono::system_clock::now().time_since_epoch().count;
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx]->color *= (materialColor * material.emittance);
        pathSegments[idx]->remainingBounces = 0;
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
        float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        pathSegments[idx]->color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        pathSegments[idx]->color *= u01(rng); // apply some noise because why not
        pathSegments[idx]->remainingBounces--;
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx]->color = glm::vec3(0.0f);
      pathSegments[idx]->remainingBounces = 0;
    }
  }
}


__global__ void shadeRealMaterial (
  int iter
  , int depth
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment ** pathSegments
	, Material * materials
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[pathSegments[idx]->pixelIndex];
    
    if (intersection.t > 0.0f && pathSegments[idx]->remainingBounces > 0) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
      thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx]->color *= (materialColor * material.emittance);
        pathSegments[idx]->remainingBounces = 0;
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
        glm::vec3 intersectionPoint = getPointOnRay(pathSegments[idx]->ray, intersection.t);
        scatterRay(pathSegments[idx], intersection.t, intersectionPoint, intersection.surfaceNormal, material, rng);
        pathSegments[idx]->remainingBounces--;
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx]->color = glm::vec3(0.0f);
      pathSegments[idx]->remainingBounces = 0;
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

// very simple function to get pointers to all the paths to prevent having to copy so much during stream compaction
__global__ void getPointersToPaths(int nPaths, PathSegment** dev_paths_ptrs, PathSegment* dev_paths)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nPaths)
  {
    dev_paths_ptrs[idx] = &(dev_paths[idx]);
  }
}


// very simple kernel to set up our thrust sort
__global__ void getMaterialIDArray(int nPaths, int* dev_materialIDs, 
  ShadeableIntersection* dev_intersections, PathSegment** dev_paths_ptrs)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nPaths)
  {
    dev_materialIDs[idx] = dev_intersections[dev_paths_ptrs[idx]->pixelIndex].materialId;
  }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
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

  if (CACHE_FIRST_ITERATION)
  {
    // save the very first iteration into the other buffer
    if (iter == 1)
    {
      generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths_first_iter_cache);
      checkCUDAError("generate camera ray");
    }

    // memcpy the cache buffer into the dev_paths buffer
    cudaMemcpy(dev_paths, dev_paths_first_iter_cache, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
  }
  else
  {
    // if we aren't caching then just generate rays into dev_paths always
    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");
  }

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
  dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
  getPointersToPaths << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_paths_ptrs, dev_paths);

  bool iterationComplete = false;
	while (!iterationComplete) {

    // clean shading chunks
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // tracing
    numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
    computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (depth, num_paths, dev_paths_ptrs, dev_geoms, hst_scene->geoms.size(), dev_intersections);
    checkCUDAError("trace one bounce");
    cudaDeviceSynchronize();
    depth++;

    /* 
    --- Shading Stage ---
    Shade path segments based on intersections and generate new rays by evaluating the BSDF.
    Start off with just a big kernel that handles all the different materials you have in the scenefile. 
    */

    thrust::device_ptr<int> dev_materialIDs_thrust(dev_material_ids);
    thrust::device_ptr<PathSegment*> dev_paths_thrust(dev_paths_ptrs);
    getMaterialIDArray << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_material_ids, dev_intersections, dev_paths_ptrs);
    thrust::sort_by_key(dev_materialIDs_thrust, dev_materialIDs_thrust + num_paths, dev_paths_thrust);

    shadeRealMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (iter, depth, num_paths, dev_intersections, dev_paths_ptrs, dev_materials);
    checkCUDAError("shade real material");

    // now we call the stream compaction
    num_paths = StreamCompaction::Efficient::compact(num_paths, dev_paths_ptrs, dev_paths_ptrs);
    checkCUDAError("Stream Compaction");

    if (num_paths <= 0)
    {
      iterationComplete = true;
    }
 	}

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

  ///////////////////////////////////////////////////////////////////////////

  // Send results to OpenGL buffer for rendering
  sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  checkCUDAError("pathtrace");
}
