#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define SORT_BY_MATERIAL 0
#define CACHE_FIRST_BOUNCE 1
#define ANTI_ALIASING 1
#define OBJ_BOUND_CULLING 0

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

struct no_more_bounce
{
    __host__ __device__
        bool operator()(const PathSegment& x)
    {
        return x.remainingBounces <= 0;
    }
};

struct has_more_bounce
{
    __host__ __device__
        bool operator()(const PathSegment& x)
    {
        return x.remainingBounces > 0;
    }
};

struct material_comparator
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& x, const ShadeableIntersection& y)
    {
        return (x.materialId > y.materialId);
    }
};

static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
#if CACHE_FIRST_BOUNCE
static ShadeableIntersection* dev_cached_first_intersection = NULL;
#endif
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

// basically it's copying all the geoms, mats, intersection to global memory
void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

#if CACHE_FIRST_BOUNCE
    cudaMalloc(&dev_cached_first_intersection, pixelcount * sizeof(ShadeableIntersection));
#endif

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
#if CACHE_FIRST_BOUNCE
    cudaFree(dev_cached_first_intersection);
#endif
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
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

#if ANTI_ALIASING
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, x, y);
        thrust::uniform_real_distribution<float> uPN(-0.3f, 0.3f);

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + uPN(rng))
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + +uPN(rng))
        );
#else
        
        // Use the camera's three directions and screen space to calculate
        // the initial directions for each pixel
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
#endif

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
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

    if (path_index >= num_paths) return;

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
    // Per path goes through all the geoms
    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        }
        if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        }
        // TODO: add more intersection tests here... triangle? metaball? CSG?
#if OBJ_BOUND_CULLING
        if (geom.type == OBJ_BOX)
        {

        }
#else
        if (geom.type == TRIANGLE)
        {
            // do triangle first
            t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        }
#endif


        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (t > 0.0f && t < t_min)
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
        return;
    }
    // The ray hits something
    // ShadeableIntersection
    intersections[path_index].t = t_min;
    intersections[path_index].materialId = geoms[hit_geom_index].materialid;
    intersections[path_index].surfaceNormal = normal;
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

__forceinline__
__host__ __device__ void BxDF_perfect_specular(glm::vec3& direction_out, glm::vec3& color_out,
    const glm::vec4& dice, const Material& material, const PathSegment& pathSegment,
    const ShadeableIntersection& intersection) {
    direction_out = glm::reflect(pathSegment.ray.direction, intersection.surfaceNormal);
    color_out = pathSegment.color * material.specular.color;
}

__forceinline__
__host__ __device__ void BxDF_perfect_refractive(glm::vec3& direction_out, glm::vec3& color_out,
    const glm::vec4& dice, const Material& material, const PathSegment& pathSegment,
    const ShadeableIntersection& intersection) {
    float indexOfRefraction = material.indexOfRefraction;
    if (glm::dot(intersection.surfaceNormal, pathSegment.ray.direction) < 0.f) {
        indexOfRefraction = 1 / indexOfRefraction;
    }
    direction_out = glm::refract(pathSegment.ray.direction, intersection.surfaceNormal,
        indexOfRefraction);
    color_out = pathSegment.color * material.specular.color;
}

__forceinline__
__host__ __device__ void BxDF_specular_and_refractive(glm::vec3& direction_out, glm::vec3& color_out,
    const glm::vec4& dice, const Material& material, const PathSegment& pathSegment,
    const ShadeableIntersection& intersection) {


    float indexOfRefraction = material.indexOfRefraction;
    float cosine = glm::dot(glm::normalize(intersection.surfaceNormal), glm::normalize(pathSegment.ray.direction));
    if (cosine < 0.f) {
        indexOfRefraction = 1 / indexOfRefraction;
    }
    float R = (1 - indexOfRefraction) / (1 + indexOfRefraction);
    R = R * R;

    // one minus cosine
    float omc = 1 + cosine;
    float fresnel_term = R + (1 - R) * omc * omc * omc * omc * omc;

    if (dice.y < fresnel_term) {
        // go reflective
        direction_out = glm::reflect(pathSegment.ray.direction, intersection.surfaceNormal);
    }
    else {
        // go refractive
        direction_out = glm::refract(pathSegment.ray.direction, intersection.surfaceNormal,
            indexOfRefraction);
    }

    color_out = pathSegment.color * material.specular.color;
}

__forceinline__
__host__ __device__ void BxDF_diffuse(glm::vec3& direction_out, glm::vec3& color_out,
    const glm::vec4& dice, const Material& material, const PathSegment& pathSegment,
    const ShadeableIntersection& intersection) {

    direction_out = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, dice.y, dice.z);
    color_out = pathSegment.color * material.color;
}

__global__ void shadeKernel(
    int depth,
    int iter
    , int num_paths
    , ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;
    // Fetch global, hide latency
    PathSegment pathSegment = pathSegments[idx];
    ShadeableIntersection intersection = shadeableIntersections[idx];
    Material material = materials[intersection.materialId];

    // Set up the RNG
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec4 dice = glm::vec4(u01(rng), u01(rng), u01(rng), u01(rng));
    // out vectors
    glm::vec3 direction_out = glm::vec3(1.f);
    glm::vec3 color_out = glm::vec3(1.f);

    // TODO(zichuanyu) temp, remove when done compaction
    if (pathSegment.remainingBounces <= 0) {
        return;
    }

    // If there was no intersection, color the ray black.
    if (intersection.t <= 0.0f) {
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = 0;
        pathSegments[idx] = pathSegment;
        return;
    }

    // If the material indicates that the object was a light, "light" the ray
    // no more bounce
    if (material.emittance > 0.0f)
    {
        pathSegment.color *= (material.color * material.emittance);
        pathSegment.remainingBounces = 0;
        pathSegments[idx] = pathSegment;
        return;
    }

    // TODO(zichuanyu)
    // make this kernel big
    // direction only deals with direction
    // color only color

    // handle PathSegment: color, ray, remainingBounces; no need to do with pixelIndex

    glm::vec3 offset_direction = intersection.surfaceNormal;

    if (dice.x < material.hasReflective) {
        // perfect specular
        BxDF_perfect_specular(direction_out, color_out, dice, material, pathSegment, intersection);
    }
    else if (dice.x < material.hasRefractive + material.hasReflective) {
        // refractive
        BxDF_specular_and_refractive(direction_out, color_out, dice, material, pathSegment, intersection);
        offset_direction = glm::normalize(direction_out); 
    }
    else {
        // diffuse
        BxDF_diffuse(direction_out, color_out, dice, material, pathSegment, intersection);
    }

    // offset the origin
    glm::vec3 new_ray_roigin = pathSegment.ray.direction * intersection.t
        + pathSegment.ray.origin + offset_direction * 0.0001f;

    --pathSegment.remainingBounces;
    pathSegment.color = color_out;
    pathSegment.ray.direction = glm::normalize(direction_out);
    pathSegment.ray.origin = new_ray_roigin;

    // write back to postions
    pathSegments[idx] = pathSegment;
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

    // TODO: perform one iteration of path tracing

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
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
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE_FIRST_BOUNCE && !ANTI_ALIASING
        // iter != 1, depth == 0, use
        // iter == 1, depth == 0, cache
        // iter == 1, depth != 0, normal
        // iter != 1, depth != 0, normal

        if (iter != 1 && depth == 0) {
            // use
            cudaMemcpy(dev_intersections, dev_cached_first_intersection,
                pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            checkCUDAError("use cache");
        }
        else {
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_intersections
                );
            if (iter == 1 && depth == 0) {
                // cache
                cudaMemcpy(dev_cached_first_intersection, dev_intersections,
                    pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
                checkCUDAError("cache cache");
            }
            // normal, do nothing
        }
#else
        // only compute intersections, dev_paths is input, dev_intersections is output
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_intersections
            );
#endif

        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

#if SORT_BY_MATERIAL
        // This step brings less branches in one warp
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, material_comparator());
#endif

        // GUESS: dev_intersections is input, dev_paths is output

        shadeKernel << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
            );

        // compaction
        dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, has_more_bounce());
        num_paths = dev_path_end - dev_paths;
        if (num_paths <= 0) {
            iterationComplete = true;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

