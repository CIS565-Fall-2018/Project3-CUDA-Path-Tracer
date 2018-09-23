#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "warpfunctions.h"
#include "materials.h"
#include "lights.h"

// #define SORT_INTERSECTIONS_BY_MATERIAL_ID



#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess == err)
  {
    return;
  }

  fprintf(stderr, "CUDA error");
  if (file)
  {
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

thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
  int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
  return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
                               int iter, glm::vec3* image)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y)
  {
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

static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Geom* dev_geom_lights = NULL;
static Material* dev_materials = NULL;
static int* dev_path_material_ids = NULL;
static int* dev_path_indices = NULL;
static PathSegment* dev_paths = NULL;
static PathSegment* dev_paths_b = NULL;
static ImageInfo* dev_imageInfo = NULL;
static glm::vec3* dev_texels = NULL;
thrust::device_ptr<PathSegment> dev_thrust_paths;
static ShadeableIntersection* dev_intersections = NULL;
static ShadeableIntersection* dev_intersections_b = NULL;
static thrust::device_ptr<int> dev_thrust_path_indices;
static thrust::device_ptr<int> dev_thrust_path_material_ids;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene* scene)
{
  hst_scene = scene;
  const Camera& cam = hst_scene->state.camera;
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
  cudaMalloc(&dev_paths_b, pixelcount * sizeof(PathSegment));
  cudaMalloc(&dev_path_material_ids, pixelcount * sizeof(int));
  cudaMalloc(&dev_path_indices, pixelcount * sizeof(int));

  dev_thrust_path_material_ids = thrust::device_ptr<int>(dev_path_material_ids);
  dev_thrust_path_indices = thrust::device_ptr<int>(dev_path_indices);

  cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
  
  cudaMalloc(&dev_imageInfo, scene->imageInfo.size() * sizeof(ImageInfo));
  cudaMemcpy(dev_imageInfo, scene->imageInfo.data(), scene->imageInfo.size() * sizeof(ImageInfo), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_texels, scene->allTexels.size() * sizeof(glm::vec3));
  cudaMemcpy(dev_texels, scene->allTexels.data(), scene->allTexels.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  std::vector<Geom> lights;
  for(const auto& geom : scene->geoms)
  {
    if (scene->materials[geom.materialid].emittance <= 0)
    {
      continue;
    }

    lights.push_back(geom);
  }

  scene->m_numLights = lights.size();

  cudaMalloc(&dev_geom_lights, lights.size() * sizeof(Geom));
  cudaMemcpy(dev_geom_lights, lights.data(), lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  cudaMalloc(&dev_intersections_b, pixelcount * sizeof(ShadeableIntersection));
  cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
  cudaMemset(dev_intersections_b, 0, pixelcount * sizeof(ShadeableIntersection));

  checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
  cudaFree(dev_image); // no-op if dev_image is null
  cudaFree(dev_paths);
  cudaFree(dev_geoms);
  cudaFree(dev_materials);
  cudaFree(dev_intersections);
  cudaFree(dev_geom_lights);
  cudaFree(dev_path_material_ids);
  cudaFree(dev_path_indices);
  cudaFree(dev_paths_b);
  cudaFree(dev_imageInfo);
  cudaFree(dev_texels);
  cudaFree(dev_intersections_b);

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

  if (x < cam.resolution.x && y < cam.resolution.y)
  {
    int index = x + (y * cam.resolution.x);
    PathSegment& segment = pathSegments[index];

    segment.ray.origin = cam.position;
    segment.color = glm::vec3(0.0f, 0.0f, 0.0f);
    segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);

    // TODO: implement antialiasing by jittering the ray
    segment.ray.direction = glm::normalize(cam.view
      - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
      - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
    );

    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
    segment.rayFromSpecular = false;
  }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
  int depth
  , int num_paths
  , PathSegment* pathSegments
  , Geom* geoms
  , int geoms_size
  , ShadeableIntersection* intersections
)
{
  int path_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (path_index < num_paths)
  {
    const PathSegment& pathSegment = pathSegments[path_index];
    int pixelIndex = pathSegment.pixelIndex;
    intersections[path_index] = Intersections::Do(pathSegment.ray, geoms, geoms_size);
  }
}

__device__ float PowerHeuristic(int nf, Float fPdf, int ng, Float gPdf)
{
  Float f = nf * fPdf;
  Float g = ng * gPdf;
  return (f * f) / (f * f + g * g);
}


__device__ Color3f GetTextureData(const ImageInfo& info, const glm::vec2& uv, glm::vec3* texels)
{
  int startTexelIdx = info.startIdx;

  int X = glm::min(info.width * uv.x, info.width - 1.0f);
  int Y = glm::min(info.height * uv.y, info.height - 1.0f); // flipped from stb image
  const int coord = startTexelIdx + X + info.height * Y;
  return texels[coord];
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
__global__ void shadeRays(
  int iter,
  int maxDepth,
  int num_paths,
  int num_lights,
  int geoms_size,
  ShadeableIntersection* shadeableIntersections,
  PathSegment* pathSegments,
  Material* materials,
  Geom* lights,
  Geom* geoms,
  glm::vec3* texels,
  ImageInfo* imageInfos
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_paths)
  {
    return;
  }

  PathSegment& targetSegment = pathSegments[idx];
  const int pixelIndex = targetSegment.pixelIndex;

  // Didn't hit anything or hit something behind
  const ShadeableIntersection intersection = shadeableIntersections[idx];
  if (intersection.t <= 0.0f)
  {
    targetSegment.remainingBounces = 0;
    return;
  }

  // if the intersection exists...
  // Set up the RNG
  // LOOK: this is how you use thrust's RNG! Please look at
  // makeSeededRandomEngine as well.
  thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
  thrust::uniform_real_distribution<float> u01(0, 1);

  const Material material = materials[intersection.materialId];
  const bool sampledSpecular = material.type == SPECULAR || material.type == ROUGH_SPECULAR;

  // If the material indicates that the object was a light, "light" the ray
  if (material.emittance > 0.0f)
  {
    // Max Depth - Hit Light Directly
    if (maxDepth == pathSegments[idx].remainingBounces || targetSegment.rayFromSpecular)
    {
      targetSegment.color += (targetSegment.throughput * material.color * material.emittance);
    }

    // Terminate Ray
    targetSegment.remainingBounces = 0;
    return;
  }

  const glm::vec3 woW = -targetSegment.ray.direction;
  const glm::vec3 wo = intersection.worldToTangent * woW;
  glm::vec3 directWiW;
  float directPdf = 0.0f;

  glm::vec3 indirectWi;
  glm::vec3 indirectWiW = glm::vec3(pixelIndex);
  float indirectPdf = 0.0f;
  Color3f indirectFrTerm;

  Color3f finalColor = Color3f(0.0f);

  const int randomIdx = (int)(u01(rng) * num_lights);
  Geom* activeLight = &lights[randomIdx];
  const Material lightMaterial = materials[activeLight->materialid];

  Color3f diffuseMaterialColor = material.color;
  if (material.diffuseMapId >= 0)
  {
    diffuseMaterialColor = GetTextureData(imageInfos[material.diffuseMapId], intersection.uv, texels);
  }

  Intersection lightIntr;

  float directFactor = 0.0f;
  float indirectFactor = 0.0f;

  if (!sampledSpecular) {

    if (material.type == DIFFUSE)
    {
      indirectFrTerm = BRDF::Lambert::Sample_f(diffuseMaterialColor, wo, &indirectWi, &indirectPdf, u01(rng), u01(rng));
    }

    indirectWiW = intersection.tangentToWorld * indirectWi;

    const Color3f directLi = Lights::Arealight::Sample_Li(lightMaterial.color * lightMaterial.emittance, intersection.intersectPoint, u01(rng), u01(rng), activeLight, &directWiW, &directPdf, &lightIntr);
    directPdf = directPdf / static_cast<float>(num_lights);

    if (directPdf > EPSILON)
    {
      const Ray shadowRay = Intersections::SpawnRay(intersection.intersectPoint, intersection.surfaceNormal, directWiW);
      const auto shadowIntr = Intersections::Do(shadowRay, geoms, geoms_size);

      if (shadowIntr.geom != nullptr)
      {
        // ID compare
        if (shadowIntr.geom->id == activeLight->id)
        {
          const float directCosTerm = std::abs(glm::dot(intersection.surfaceNormal, directWiW));
          const glm::vec3 directWi = intersection.worldToTangent * directWiW;

          if (material.type == DIFFUSE)
          {
            const Color3f directFrTerm = BRDF::Lambert::f(diffuseMaterialColor, wo, directWi);
            directFactor = PowerHeuristic(1, directPdf, 1, BRDF::Lambert::Pdf(wo, directWi));
            finalColor += ((directFrTerm * directLi * directCosTerm * directFactor) / directPdf);
          }
        }
      }
    }

    if (indirectPdf > EPSILON)
    {
      float lightPdf = Lights::Arealight::Pdf_Li(intersection.intersectPoint, intersection.surfaceNormal, indirectWiW, activeLight);
      if (lightPdf > EPSILON) {
        lightPdf = lightPdf / num_lights;
        indirectFactor = PowerHeuristic(1, indirectPdf, 1, lightPdf);
      }

      Ray indirectRay = Intersections::SpawnRay(intersection.intersectPoint, intersection.surfaceNormal, indirectWiW);
      Intersection indirectIsect;

      const float indirectCosTerm = std::abs(glm::dot(intersection.surfaceNormal, indirectWiW));

      const auto indirectIntr = Intersections::Do(indirectRay, geoms, geoms_size);

      Color3f indirectLiTerm = Color3f(0.0f);

      if (indirectIntr.geom != nullptr)
      {
        if (indirectIntr.geom->id == activeLight->id) {
          indirectLiTerm = Lights::Arealight::L(lightMaterial.color * lightMaterial.emittance, indirectIntr.surfaceNormal, -indirectWiW);
        }

        finalColor += ((indirectFrTerm * indirectLiTerm * indirectCosTerm * indirectFactor) / indirectPdf);
      }
    }
  }

  // Add MIS Color
  targetSegment.color += (finalColor * targetSegment.throughput);

  targetSegment.remainingBounces--;

  if (targetSegment.remainingBounces <= 0)
  {
    // No Need to compute next ray
    return;
  }

  Vector3f bounceWi;
  Vector3f bounceWiW;
  float bouncePdf;
  Color3f bounceFrTerm;

  targetSegment.rayFromSpecular = false;

  if (material.type == DIFFUSE)
  {
    bounceFrTerm = BRDF::Lambert::Sample_f(diffuseMaterialColor, wo, &bounceWi, &bouncePdf, u01(rng), u01(rng));
  }
  else if (material.type == SPECULAR)
  {
    bounceFrTerm = BRDF::Specular::Sample_f(diffuseMaterialColor, wo, &bounceWi, &bouncePdf, FRESNEL_NOOP);
    targetSegment.rayFromSpecular = true;
  }
  else if (material.type == ROUGH_SPECULAR)
  {
    bounceFrTerm = BRDF::Microfacet::Sample_f(material.color, wo, &bounceWi, &bouncePdf, FRESNEL_NOOP, u01(rng), u01(rng), material.roughness, material.roughness);
    targetSegment.rayFromSpecular = true;
  }

  bounceWiW = intersection.tangentToWorld * bounceWi;

  const float bounceCosTerm = std::abs(glm::dot(intersection.surfaceNormal, bounceWiW));

  if (bouncePdf < EPSILON) {
    // Terminate Ray
    targetSegment.remainingBounces = 0;
    return;
  }

  targetSegment.throughput = (targetSegment.throughput * bounceFrTerm * bounceCosTerm) / bouncePdf;
  targetSegment.ray = Intersections::SpawnRay(intersection.intersectPoint, intersection.surfaceNormal, bounceWiW);

  // Russian Roulette
  const float maxVal = glm::max(glm::max(static_cast<float>(targetSegment.throughput[0]), targetSegment.throughput[1]), targetSegment.throughput[2]);

  if (u01(rng) < (1.0f - maxVal)) {
    targetSegment.remainingBounces = 0;
    return;
  }
  
  targetSegment.throughput /= maxVal;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < nPaths)
  {
    PathSegment iterationPath = iterationPaths[index];
    image[iterationPath.pixelIndex] += iterationPath.color;
  }
}

__global__ void fillMaterialArray(int nPaths, int* materialIds, int* pathIndices, ShadeableIntersection* shadeableIntersections)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < nPaths)
  {
    const ShadeableIntersection& shadeIntr = shadeableIntersections[index];
    materialIds[index] = shadeIntr.materialId;
    pathIndices[index] = index;
  }
}

__global__ void reshufflePathSegments(int nPaths, int* pathIndices, ShadeableIntersection* shadeableIntersections, PathSegment* segments,  ShadeableIntersection* oIntrs, PathSegment* oSegments)
{
  const int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < nPaths)
  {
    const int idx = pathIndices[index];
    oIntrs[index] = shadeableIntersections[idx];
    oSegments[index] = segments[idx];
  }
}


struct IsValidPath
{
  __host__ __device__ bool operator() (const PathSegment& segment)
  {
    return segment.remainingBounces > 0;
  }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
  const int traceDepth = hst_scene->state.traceDepth;
  const Camera& cam = hst_scene->state.camera;
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

  generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
  checkCUDAError("generate camera ray");

  int depth = 0;
  PathSegment* dev_path_end = dev_paths + pixelcount;
  int num_paths = dev_path_end - dev_paths;

  const int all_path_count = num_paths;

  // --- PathSegment Tracing Stage ---
  // Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
  while (!iterationComplete)
  {
    // clean shading chunks
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // tracing
    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
    computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>>(
      depth,
      num_paths,
      dev_paths,
      dev_geoms,
      hst_scene->geoms.size(),
      dev_intersections
    );
    checkCUDAError("trace one bounce");

#ifdef SORT_INTERSECTIONS_BY_MATERIAL_ID
    fillMaterialArray<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_path_material_ids, dev_path_indices, dev_intersections);
    thrust::sort_by_key(dev_thrust_path_material_ids, dev_thrust_path_material_ids + num_paths, dev_thrust_path_indices);
    reshufflePathSegments<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_path_indices, dev_intersections, dev_paths, dev_intersections_b, dev_paths_b);

    PathSegment* tempA = dev_paths;
    ShadeableIntersection* tempB = dev_intersections;
    dev_paths = dev_paths_b;
    dev_intersections = dev_intersections_b;
    dev_paths_b = tempA;
    dev_intersections_b = tempB;
#endif

    dev_thrust_paths = thrust::device_ptr<PathSegment>(dev_paths);

    shadeRays<<<numblocksPathSegmentTracing, blockSize1d>>>(
      iter,
      traceDepth,
      num_paths,
      hst_scene->m_numLights,
      hst_scene->geoms.size(),
      dev_intersections,
      dev_paths,
      dev_materials,
      dev_geom_lights,
      dev_geoms,
      dev_texels,
      dev_imageInfo
    );

    const auto middleItr = thrust::partition(dev_thrust_paths, dev_thrust_paths + num_paths, IsValidPath());
    iterationComplete = dev_paths == middleItr.get();
    num_paths = middleItr.get() - dev_paths;
  }

  cudaDeviceSynchronize();

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
  finalGather<<<numBlocksPixels, blockSize1d>>>(all_path_count, dev_image, dev_paths);

  ///////////////////////////////////////////////////////////////////////////

  // Send results to OpenGL buffer for rendering
  sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image,
             pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  checkCUDAError("pathtrace");
}
