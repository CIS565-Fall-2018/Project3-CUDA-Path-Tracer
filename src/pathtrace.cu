#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

// Toggleable features (0 for false, 1 for true)
#define STREAM_COMPACTION 1
#define CACHE_FIRST_BOUNCE 0
#define SORT_BY_MATERIALS 1

#define AA !CACHE_FIRST_BOUNCE
#define DEPTH_OF_FIELD 1
#define BOUNDING_VOLUME_CULLING 1

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
static ShadeableIntersection * dev_intersections_first = NULL;
static int* dev_indicesCompact = NULL;
static int* dev_matIndicesSorted = NULL;

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

	cudaMalloc(&dev_indicesCompact, pixelcount * sizeof(int));

	cudaMalloc(&dev_intersections_first, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections_first, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_matIndicesSorted, pixelcount * sizeof(int));

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);

	cudaFree(dev_indicesCompact); // indices used to access paths/intersections (needed if stream compaction is implemented)
	cudaFree(dev_intersections_first); // buffer to store iteration 1 & depth 0 intersections for caching
	cudaFree(dev_matIndicesSorted); // key buffer used to sort dev_indicesCompact according to the material ID

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, int* indicesCompact)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		float epsilonX = 0;
		float epsilonY = 0;
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		thrust::uniform_real_distribution<float> u01(0, 1);
#if AA
		epsilonX = u01(rng);
		epsilonY = u01(rng);
#endif
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + epsilonX)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + epsilonY)
		);

#if DEPTH_OF_FIELD
		float lensRad = cam.lensRadius;
		float focalDist = cam.focalDistance;
		glm::vec3 pLens = glm::vec3(lensRad * calculateConcentricSampleDisk(u01(rng), u01(rng)), 0.0f);
		float ft = focalDist / glm::abs(segment.ray.direction.z);
		glm::vec3 pFocus = segment.ray.direction * ft;
		
		segment.ray.origin += pLens;
		segment.ray.direction = glm::normalize(pFocus - pLens);
#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
		indicesCompact[index] = index;
	}
}

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
	, int* indicesCompact
	, int* indicesMaterial
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		int idx = indicesCompact[path_index];
		PathSegment pathSegment = pathSegments[idx];

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
			else if (geom.type == MESH) {
#if BOUNDING_VOLUME_CULLING
				if (!meshBoundingVolumeIntersectionTest(geom, pathSegment.ray)) {
					// if the ray completely misses the bounding volume, then skip the mesh by jumping to the next
					// geom (skipping by nbTriangles)
					i += geom.nbTriangles;
				}
#endif
				continue;
			}
			else if (geom.type == TRI) {
				t = triIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}

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
			intersections[idx].t = -1.0f;
		}
		else
		{
			// The ray hits something
			intersections[idx].t = t_min;
			intersections[idx].materialId = geoms[hit_geom_index].materialid;
			indicesMaterial[path_index] = intersections[idx].materialId; // store material ID for sorting
			intersections[idx].surfaceNormal = normal;
		}
	}
}

__global__ void shadeRealMaterial(
	int iter
	, int depth
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, int* indicesCompact) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		int compactIndex = indicesCompact[idx];

#if STREAM_COMPACTION
		bool deadPath = compactIndex == -1 || pathSegments[compactIndex].remainingBounces <= 0;
#else
		bool deadPath = pathSegments[compactIndex].remainingBounces <= 0;
#endif	

		if (!deadPath) {
			PathSegment* path = &pathSegments[compactIndex];
			ShadeableIntersection intersection = shadeableIntersections[compactIndex];
			if (intersection.t > 0.0f) {
				// if the intersection exists...
				// Set up the RNG
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, compactIndex, depth);
				thrust::uniform_real_distribution<float> u01(0, 1);

				Material material = materials[intersection.materialId];
				glm::vec3 materialColor = material.color;

				// If the material indicates that the object was a light, "light" the ray
				if (material.emittance > 0.0f) {
					path->color *= (materialColor * material.emittance);
					path->remainingBounces = 0;
				}
				// Otherwise, do some pseudo-lighting computation. This is actually more
				// like what you would expect from shading in a rasterizer like OpenGL.
				else {
					scatterRay(*path,
						getPointOnRay(path->ray, intersection.t),
						intersection.surfaceNormal,
						material,
						rng);
					path->remainingBounces -= 1;
				}
				// If there was no intersection, color the ray black.
				// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
				// used for opacity, in which case they can indicate "no opacity".
				// This can be useful for post-processing and image compositing.
			}
			else {
				path->color = glm::vec3(0.0f);
				path->remainingBounces = 0;
			}

#if STREAM_COMPACTION
			// set the index to -1 if the path terminates
			indicesCompact[idx] = path->remainingBounces <= 0 ? -1 : indicesCompact[idx];
#endif	
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

struct is_path_complete {
	__host__ __device__
		bool operator()(const int index) {
		return index == -1;
	}
};

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

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, dev_indicesCompact);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing((num_paths + blockSize1d - 1) / blockSize1d);
		
		// which device array of intersections to use in the shading kernel
		ShadeableIntersection* intersectionsToUse = dev_intersections;

#if CACHE_FIRST_BOUNCE
		if (depth == 0) {
			// might reuse cache if depth == 0
			if (iter == 1) {
				// compute 1st bounce and cache it
				computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth
					, num_paths
					, dev_paths
					, dev_geoms
					, hst_scene->geoms.size()
					, dev_intersections
					, dev_indicesCompact
					, dev_matIndicesSorted
					);
				checkCUDAError("trace one bounce");
				cudaMemcpy(dev_intersections_first, dev_intersections,
					pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			else {
				// reuse cache
				intersectionsToUse = dev_intersections_first;
			}
		}
		else {
			// cannot reuse cache if depth > 0
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, dev_indicesCompact
				, dev_matIndicesSorted
				);
			checkCUDAError("trace one bounce");
		}
#else
		 computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			 depth
			 , num_paths
			 , dev_paths
			 , dev_geoms
			 , hst_scene->geoms.size()
			 , dev_intersections
			 , dev_indicesCompact
			 , dev_matIndicesSorted
			 );
		 checkCUDAError("trace one bounce");
#endif
		cudaDeviceSynchronize();
		

	  // --- Shading Stage ---
	  // Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.

#if SORT_BY_MATERIALS
		thrust::sort_by_key(thrust::device, dev_matIndicesSorted, dev_matIndicesSorted + num_paths, dev_indicesCompact);
#endif
		shadeRealMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
		  iter,
		  depth,
		  num_paths,
		  intersectionsToUse,
		  dev_paths,
		  dev_materials,
		  dev_indicesCompact
		  );
		depth++;

		 if (depth >= traceDepth) {
			// iteration ends due to depth
			iterationComplete = true;
		 }
#if STREAM_COMPACTION
		else {
			// stream compaction
			int* new_indices_end = thrust::remove_if(thrust::device, dev_indicesCompact, dev_indicesCompact + num_paths, is_path_complete());
			num_paths = new_indices_end - dev_indicesCompact; // shrink number paths
			iterationComplete = (num_paths <= 0);
		}
#endif
  }

	// Assemble this iteration and apply it to the image
	num_paths = dev_path_end - dev_paths; // reset num paths to pixel count for a correct final gather
	dim3 numBlocksPixels((pixelcount + blockSize1d - 1) / blockSize1d);
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	checkCUDAError("pathtrace");
}
