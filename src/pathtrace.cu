#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

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

static ShadeableIntersection * dev_cached_intersections = NULL;

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

    float x_offseted = x;
    float y_offseted = y;

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, segment.remainingBounces);
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (ANTI_ALIAS) {
      x_offseted += u01(rng);
      y_offseted += u01(rng);
    }

    segment.ray.direction = glm::normalize(cam.view
      - cam.right * cam.pixelLength.x * (x_offseted - (float)cam.resolution.x * 0.5f)
      - cam.up * cam.pixelLength.y * (y_offseted - (float)cam.resolution.y * 0.5f)
    );

    if (LENS_RADIUS > 0) {
      //Sample point on lens
      glm::vec3 lens_origin = squareToDiskConcentric(glm::vec2(u01(rng), u01(rng)));
      lens_origin *= LENS_RADIUS;

      //Compute point on plane of focus
      float t_val = glm::abs(FOCAL_DISTANCE / segment.ray.direction.z);
      glm::vec3 focal_point = t_val * segment.ray.direction;

      //Update ray for effect of lens
      segment.ray.origin += lens_origin;
      segment.ray.direction = glm::normalize(focal_point - lens_origin);
    }

    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
  }
}

// TODO: computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
  int depth, int num_paths,
  PathSegment * pathSegments, Geom * geoms,
  int geoms_size, ShadeableIntersection * intersections) {

  int path_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (path_index < num_paths) {
    PathSegment pathSegment = pathSegments[path_index];

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    // naive parse through global geoms

    for (int i = 0; i < geoms_size; i++) {
      Geom & geom = geoms[i];

      if (geom.type == CUBE) {
        t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
      }
      else if (geom.type == SPHERE) {
        t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
      }
      // TODO: add more intersection tests here... triangle? metaball? CSG?

      // Compute the minimum t from the intersection tests to determine what
      // scene geometry object was hit first.
      if (t > 0.0f && t_min > t) {
        t_min = t;
        hit_geom_index = i;
        intersect_point = tmp_intersect;
        normal = tmp_normal;
      }
    }

    if (hit_geom_index == -1) {
      intersections[path_index].t = -1.0f;
    }
    else {
      //The ray hits something
      intersections[path_index].t = t_min;
      intersections[path_index].materialId = geoms[hit_geom_index].materialid;
      intersections[path_index].surfaceNormal = normal;
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
__global__ void shadeFakeMaterial(
  int iter, int num_paths,
  ShadeableIntersection * shadeableIntersections,
  PathSegment * pathSegments, Material * materials) {

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
    }
    else {
      pathSegments[idx].color = glm::vec3(0.0f);
    }
  }
}

__global__ void shadeMaterial(
  int iter, int num_paths,
  ShadeableIntersection * shadeableIntersections,
  PathSegment * pathSegments, Material * materials) {

  // naive implementation - no direct lighting and no mis

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!(idx < num_paths)) {
    return;
  }
  __syncthreads();

  if (pathSegments[idx].remainingBounces < 1) {
    return;
  }
  __syncthreads();

  thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
  thrust::uniform_real_distribution<float> u01(0, 1);

  ShadeableIntersection intersection = shadeableIntersections[idx];

  // if no intersection, color the ray black.
  if (intersection.t < 0.f) {
    pathSegments[idx].color = glm::vec3(0.0f);
    pathSegments[idx].remainingBounces = 0;
    return;
  }
  __syncthreads();

  Material material = materials[intersection.materialId];

  // If the material indicates that the object was a light, "light" the ray
  if (material.emittance > 0.f) {
    pathSegments[idx].color *= (material.color * material.emittance);
    pathSegments[idx].remainingBounces = 0;
    return;
  }
  __syncthreads();

  Ray old_ray = pathSegments[idx].ray;
  glm::vec3 old_intersection = old_ray.origin + intersection.t * old_ray.direction;
  //old_intersection = old_intersection + EPSILON * intersection.surfaceNormal;

  // scatterRay fills in vals for bounced ray and updates color based on material appropriately
  scatterRay(pathSegments[idx], old_intersection, intersection.surfaceNormal, material, rng);
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < nPaths) {
    PathSegment iterationPath = iterationPaths[index];
    image[iterationPath.pixelIndex] += iterationPath.color;
  }
}

// For material sorting
struct sort_by_material {
  __host__ __device__
  bool operator() (const ShadeableIntersection &first_isx, const ShadeableIntersection &second_isx) {
    return first_isx.materialId < second_isx.materialId;
  }
};

// For compaction
struct split_by_completed {
  __host__ __device__
  bool operator() (const PathSegment &segment) {
    return segment.remainingBounces > 0;
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

  /*********************************************/
  /********** BEGIN PATHTRACE LOOPING **********/
  /*********************************************/

  generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
  checkCUDAError("generate camera ray");

  int depth = 0;
  PathSegment* dev_path_end = dev_paths + pixelcount;
  int num_paths = dev_path_end - dev_paths;

#ifdef SORT_MATERIAL
  thrust::device_ptr<ShadeableIntersection> thrust_intersections(dev_intersections);
  thrust::device_ptr<PathSegment> thrust_paths(dev_paths);
#endif

  // --- PathSegment Tracing Stage --- //
  // Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
  while (!iterationComplete) {
    // clean shading chunks
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // tracing
    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

    // compute intersection
    computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
      depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections);

    checkCUDAError("trace one bounce");
    cudaDeviceSynchronize();
    ++depth;

#ifdef CACHE_FIRST
    // Handling intersections for first bounce
    if (depth == 0) {
      if (iter == 1) {
        cudaMemcpy(dev_cached_intersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
      } else {
        cudaMemcpy(dev_intersections, dev_cached_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
      }
    }
#endif
   
#ifdef SORT_MATERIAL
    thrust::sort_by_key(thrust_intersections, thrust_intersections, thrust_paths + num_paths, sort_by_material());
#endif
    //cudaDeviceSynchronize();

    // --- Shading Stage ---
    // Shade path segments based on intersections and generate new rays by
    // evaluating the BSDF.

    // Handling shading
    shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
      iter, num_paths, dev_intersections, dev_paths, dev_materials);

    num_paths = dev_path_end - dev_paths;

#ifdef STREAM_COMPACTION
      // CHECKITOUT - To test anti-aliasing, change depth >= 1, and move the camera around. You'll see jagged edges become smoother
    PathSegment* pivot_index = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, split_by_completed());
    num_paths = pivot_index - dev_paths;
    if (num_paths < 1) {
      depth = traceDepth + 1;
    }
#endif
    //cudaDeviceSynchronize();

    iterationComplete = depth > traceDepth;

    ++depth;

  } //end: while iterating bounces

  num_paths = dev_path_end - dev_paths;

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
  finalGather << <numBlocksPixels, blockSize1d >> >(num_paths, dev_image, dev_paths);

  /*******************************************/
  /********** END PATHTRACE LOOPING **********/
  /*******************************************/

  // Send results to OpenGL buffer for rendering
  sendImageToPBO << <blocksPerGrid2d, blockSize2d >> >(pbo, cam.resolution, iter, dev_image);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image,
    pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  checkCUDAError("pathtrace");
}
