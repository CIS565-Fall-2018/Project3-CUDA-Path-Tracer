#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "shapeFunctions.h"
#include "bsdf.h"
#include "lightFunctions.h"

bool doMaterialSort = false;
bool doFirstCache = false;
bool doAntiAlias = true;
bool doCompact = true;
IntegratorType integrator = IntegratorType::NAIVE;

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
// TODO: static variables for device memory, any extra info you need, etc
// ...
static int *dev_materialIds = NULL;
static int *dev_indices = NULL;
static ShadeableIntersection * dev_firstIntersections = NULL;
static PathSegment * dev_firstPaths = NULL; 
static Geom * dev_lights = NULL;

// Attempt to make a better material sort than thrust
// But it was not better...
/*
static ShadeableIntersection * dev_intersectionsCopy = NULL;
static PathSegment * dev_pathsCopy = NULL;
*/

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
    cudaMalloc(&dev_materialIds, pixelcount * sizeof(int));

    cudaMalloc(&dev_indices, pixelcount * sizeof(int));

    cudaMalloc(&dev_firstPaths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_firstIntersections, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    // Attempt to make a better material sort than thrust
    // But it was not better...
    /*
    cudaMalloc(&dev_pathsCopy, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_intersectionsCopy, pixelcount * sizeof(ShadeableIntersection));
    */

    doMaterialSort = scene->materialSort;
    doFirstCache = scene->firstCache;
    doAntiAlias = scene->antiAlias;
    doCompact = scene->streamCompact;

    if (doFirstCache && doAntiAlias) {
        doFirstCache = false;
    }

    switch (scene->integrator) {
        case 'N':
            integrator = IntegratorType::NAIVE;
            break;
        case 'D':
            integrator = IntegratorType::DIRECT;
            break;
        case 'F':
            integrator = IntegratorType::FULL;
            break;
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_materialIds);
    cudaFree(dev_indices);
    cudaFree(dev_firstPaths);
    cudaFree(dev_firstIntersections);

    // Attempt to make a better material sort than thrust
    // But it was not better...
    /*
    cudaFree(dev_pathsCopy);
    cudaFree(dev_intersectionsCopy);
    */

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, PathSegment* pathSegmentsFirst, bool cache, bool aa, IntegratorType integratorType)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);

        if (cache && iter > 1) {
            pathSegments[index] = pathSegmentsFirst[index];
            return;
        }

        PathSegment & segment = pathSegments[index];

        segment.ray.origin = cam.position;

        switch (integratorType) {
            case IntegratorType::NAIVE:
            case IntegratorType::DIRECT:
                segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
                break;
            case IntegratorType::FULL:
                segment.color = glm::vec3(0.0f, 0.0f, 0.0f);
                segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
                break;
        }

        // implement antialiasing by jittering the ray
        float jitX = 0;
        float jitY = 0;
        if (aa) {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, pathSegments[index].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);
            jitX = u01(rng);
            jitY = u01(rng);
        }

        
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * ((cam.pixelLength.x * (jitX + (float)x - (float)cam.resolution.x * 0.5f)))
            - cam.up * cam.pixelLength.y * (jitY + (float)y - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;

        if (cache) {
            pathSegmentsFirst[index] = pathSegments[index];
        }
    }
}



// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth
    , int iter
    , int num_paths
    , PathSegment * pathSegments
    , Geom * geoms
    , int geoms_size
    , ShadeableIntersection * intersections
    , ShadeableIntersection * intersectionsFirst
    , bool cache
    )
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        if (cache && depth == 0 && iter > 1) {
            intersections[path_index] = intersectionsFirst[path_index];
            return;
        }
        PathSegment pathSegment = pathSegments[path_index];
        ShadeableIntersection& intersection = intersections[path_index];
        
        sceneIntersect(pathSegment.ray, geoms, geoms_size, intersection);

        if (cache && depth == 0) {
            intersectionsFirst[path_index] = intersections[path_index];
        }
    }
}

/* Attempt to have a kernel compute intersections with all geometry for a single ray


__global__ void computeSingleRayIntersection(
    PathSegment * pathSegment
    , Geom * geoms
    , int geoms_size
    , ShadeableIntersection * intersection
    , bool cache
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < geoms_size)
    {
        intersection->t = -1;

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec3 tangent;
        glm::vec3 bitangent;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec3 tmp_tangent;
        glm::vec3 tmp_bitangent;

        bool hit = false;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            hit = false;
            ShadeableIntersection tmp_intersection;
            Geom & geom = geoms[i];

            if (geom.type == CUBE)
            {
                //t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_tangent, tmp_bitangent, outside);
                if (Cube::Intersect(pathSegment.ray, geom, &tmp_intersection)) {
                    hit = true;
                }
            }
            else if (geom.type == SPHERE)
            {
                //t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_tangent, tmp_bitangent, outside);
                if (Sphere::Intersect(pathSegment.ray, geom, &tmp_intersection)) {
                    hit = true;
                }
            }
            else if (geom.type == SQUAREPLANE)
            {
                //t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_tangent, tmp_bitangent, outside);
                if (SquarePlane::Intersect(pathSegment.ray, geom, &tmp_intersection)) {
                    hit = true;
                }
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (hit) {
                if (tmp_intersection.t < intersection.t || intersection.t < 0)
                {
                    intersection.t = tmp_intersection.t;
                    intersection.materialId = geoms[i].materialid;
                    intersection.geomId = i;
                    intersection.surfaceNormal = tmp_intersection.surfaceNormal;
                    intersection.tangent = tmp_intersection.tangent;
                    intersection.bitangent = tmp_intersection.bitangent;
                }
            }
        }
    }
}
*/

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

__global__ void naiveKernel(
    int iter
    , int num_paths
    , ShadeableIntersection * shadeableIntersections
    , PathSegment * pathSegments
    , Material * materials
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...

            // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            // Lighting computation
            else {

                Vector3f woW = -pathSegments[idx].ray.direction;
                Vector3f wiW;
                float pdf = 0;

                Color3f color = BSDF::Sample_f(woW, &wiW, &pdf, material, intersection, rng);

                if (pdf < PDF_EPSILON) {
                    pathSegments[idx].color = glm::vec3(0.f);
                    pathSegments[idx].remainingBounces = 0;
                    return;
                }

                pathSegments[idx].ray.origin = GetNewRayOrigin(wiW, intersection.surfaceNormal, intersection.point); 
                pathSegments[idx].ray.direction = wiW;
                pathSegments[idx].remainingBounces--;
                pathSegments[idx].color *= color * AbsDot(wiW, glm::normalize(intersection.surfaceNormal)) / pdf;

            }
        }
        // If there was no intersection, color the ray black.
        // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
        // used for opacity, in which case they can indicate "no opacity".
        // This can be useful for post-processing and image compositing.
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

__global__ void directLightingKernel(
    int iter
    , int num_paths
    , ShadeableIntersection * shadeableIntersections
    , PathSegment * pathSegments
    , Material * materials
    , Geom * lights
    , int numLights
    , Geom * geoms
    , int geoms_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...

            Material material = materials[intersection.materialId];

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (material.color * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            // Lighting computation
            else {

                // Set up the RNG
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);

                pathSegments[idx].color = DirectLightingSample(intersection, pathSegments[idx], material, materials, lights, numLights, geoms, geoms_size, rng);
                pathSegments[idx].remainingBounces = 0;
            }
        }
        // If there was no intersection, color the ray black.
        // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
        // used for opacity, in which case they can indicate "no opacity".
        // This can be useful for post-processing and image compositing.
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

__global__ void fullLightingKernel(
    int iter
    , int num_paths
    , ShadeableIntersection * shadeableIntersections
    , PathSegment * pathSegments
    , Material * materials
    , Geom * lights
    , int numLights
    , Geom * geoms
    , int geoms_size
    , const int maxDepth
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...

            Material material = materials[intersection.materialId];

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                if (pathSegments[idx].remainingBounces == maxDepth) {
                    pathSegments[idx].color += (material.color * material.emittance);
                }
                pathSegments[idx].remainingBounces = 0;
            }
            // Lighting computation
            else {

                // Set up the RNG
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);

                // Get a direct lighting sample
                Color3f Ld = DirectLightingSample(intersection, pathSegments[idx], material, materials, lights, numLights, geoms, geoms_size, rng);

                // Global illumination
                Vector3f woW = -pathSegments[idx].ray.direction;
                Vector3f wiW;
                float pdf = 0;
                Color3f f = BSDF::Sample_f(woW, &wiW, &pdf, material, intersection, rng);

                if (pdf < PDF_EPSILON) {
                    pathSegments[idx].remainingBounces = 0; 
                    return;
                }

                pathSegments[idx].color += Ld * pathSegments[idx].throughput;
                pathSegments[idx].throughput *= (f * AbsDot(intersection.surfaceNormal, wiW)) / pdf;

                // Russian Roulette
                thrust::uniform_real_distribution<float> u01(0, 1);
                float tpMax = maxValue(pathSegments[idx].throughput);
                if (u01(rng) < (1.f - tpMax)) {
                    pathSegments[idx].remainingBounces = 0;
                }
                else {
                    pathSegments[idx].throughput /= tpMax;
                    pathSegments[idx].remainingBounces--;
                    pathSegments[idx].ray.direction = wiW;
                    pathSegments[idx].ray.origin = GetNewRayOrigin(wiW, intersection.surfaceNormal, intersection.point);
                }
            }
        }
        // If there was no intersection, color the ray black.
        // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
        // used for opacity, in which case they can indicate "no opacity".
        // This can be useful for post-processing and image compositing.
        else {
            //pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

// Attempt to make a better material sort than thrust
// But it was not better...
/*
__global__ void copyMaterialId(int nPaths, int* copyTo, int* indices, ShadeableIntersection *intersections, ShadeableIntersection *intersectCopy, PathSegment* segments, PathSegment* segmentsCopy) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        copyTo[index] = intersections[index].materialId;
        indices[index] = index;
        intersectCopy[index] = intersections[index];
        segmentsCopy[index] = segments[index];
    }
}
*/

// Attempt to make a better material sort than thrust
// But it was not better...
/*
__global__ void shuffleIntersectionsSegments(int nPaths, int* indices, PathSegment* pathSegments, PathSegment* pathSegmentCopy, ShadeableIntersection *intersections, ShadeableIntersection *intersectionsCopy) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        pathSegments[index] = pathSegmentCopy[indices[index]];
        intersections[index] = intersectionsCopy[indices[index]];
    }
}
*/

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

    int depth = 0;

    generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths, dev_firstPaths, doFirstCache, doAntiAlias, integrator);
    checkCUDAError("generate camera ray");

    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete) {

        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth
            , iter
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_intersections
            , dev_firstIntersections
            , doFirstCache
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        // Sort by materialId
        if (doMaterialSort) {
            thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, intersectMaterialCompare());

            // Attempt to make a better material sort than thrust
            // But it was not better...
            //copyMaterialId << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_materialIds, dev_indices, dev_intersections, dev_intersectionsCopy, dev_paths, dev_pathsCopy);
            //thrust::sort_by_key(thrust::device, dev_materialIds, dev_materialIds + num_paths, dev_indices);
            //shuffleIntersectionsSegments << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_indices, dev_paths, dev_pathsCopy, dev_intersections, dev_intersectionsCopy);
        }

        switch (integrator) {
            case IntegratorType::NAIVE:
                
                naiveKernel<<<numblocksPathSegmentTracing, blockSize1d>>> (
                iter,
                num_paths,
                dev_intersections,
                dev_paths,
                dev_materials
                );
                
                break;

            case IntegratorType::DIRECT:

                directLightingKernel << <numblocksPathSegmentTracing, blockSize1d >> > (
                    iter,
                    num_paths,
                    dev_intersections,
                    dev_paths,
                    dev_materials,
                    dev_lights,
                    hst_scene->lights.size(),
                    dev_geoms,
                    hst_scene->geoms.size()
                    );

                break;

            case IntegratorType::FULL:
                fullLightingKernel << <numblocksPathSegmentTracing, blockSize1d >> > (
                    iter,
                    num_paths,
                    dev_intersections,
                    dev_paths,
                    dev_materials,
                    dev_lights,
                    hst_scene->lights.size(),
                    dev_geoms,
                    hst_scene->geoms.size(),
                    traceDepth
                    );

                break;
        }


        // Use thrust::partition to put dead paths at end of dev_paths
        // https://stackoverflow.com/questions/37013191/is-it-possible-to-create-a-thrusts-function-predicate-for-structs-using-a-given
        if (doCompact) {
            dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, pathBounceZero());
            num_paths = (dev_path_end - dev_paths);
        }

        if (depth > traceDepth || num_paths <= 0) {
            iterationComplete = true;
        }

    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

void toggleMaterialSort() {
    doMaterialSort = !doMaterialSort;
}
