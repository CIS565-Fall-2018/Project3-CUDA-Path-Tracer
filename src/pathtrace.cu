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
#include "defines.h"

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
static glm::vec3 *dev_tex = NULL;
#if KDTREE
static KDNode *dev_kdtree = NULL;
#endif
// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

#if KDTREE
  	cudaMalloc(&dev_geoms, scene->sortedGeoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->sortedGeoms.data(), scene->sortedGeoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
#else
	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
#endif

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

	// Allocate texture data
	cudaMalloc(&dev_tex, scene->textureData.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_tex, scene->textureData.data(), scene->textureData.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	#if KDTREE
	cudaMalloc(&dev_kdtree, scene->kdtree.size() * sizeof(KDNode));
	cudaMemcpy(dev_kdtree, scene->kdtree.data(), scene->kdtree.size() * sizeof(KDNode), cudaMemcpyHostToDevice);
	#endif
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

	cudaFree(dev_tex);
	#if KDTREE
	cudaFree(dev_kdtree);
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
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(-0.5, 0.5);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
			);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}
__forceinline__
__host__ __device__
float getAlpha(float y, float py, float qy) {
	return (y - py) / (qy - py);
}

__forceinline__
__host__ __device__
glm::vec3 slerp(float alpha, glm::vec3 az, glm::vec3 bz) {
	return glm::vec3((1 - alpha) * az.r + alpha * bz.r,
		(1 - alpha) * az.g + alpha * bz.g,
					 (1 - alpha) * az.b + alpha * bz.b);
}

__forceinline__
__host__ __device__
glm::vec3 fetchColor(glm::vec3 *textureData, const Material &m, int x, int y, bool texture = true) {
	int pix = y * (texture ? m.texWidth : m.norWidth) + x;
	return textureData[pix + (texture ? m.textureOffset : m.normalOffset)];
}

__global__ void computeIntersectionsKD(int depth, int num_paths, PathSegment *pathSegments,
									   Geom *geoms, KDNode *kdtree, int geoms_size,
									   ShadeableIntersection * intersections,
									   Material *mats, glm::vec3 *textureData) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths) {
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		float boundsT;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		glm::vec2 uv(0);
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv;

		// Traverse through kd tree
		int toVisitOffset = 0, currentNodeIndex = 0;
		int nodesToVisit[64];
		while (true) {
			KDNode node = kdtree[currentNodeIndex];
			boundsT = boundsIntersectionTest(node.bounds, pathSegment.ray);
			if (boundsT != -1.f) {
				if (node.numPrims > 0) {
					for (int i = 0; i < node.numPrims; ++i) {
						Geom &geom = geoms[node.primOffset + i];
						if (geom.type == CUBE) {
							t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmp_uv);
						} else if (geom.type == SPHERE) {
							t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmp_uv);
						} else if (geom.type == TRIANGLE) {
							t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmp_uv);
						}
						if (t > 0.0f && t_min > t) {
							t_min = t;
							hit_geom_index = node.primOffset + i;
							intersect_point = tmp_intersect;
							normal = tmp_normal;
							uv = tmp_uv;
						}
					}
					if (toVisitOffset == 0) { break; }
					currentNodeIndex = nodesToVisit[--toVisitOffset];
				} else {
					nodesToVisit[toVisitOffset++] = node.secondChildOffset;
					currentNodeIndex = currentNodeIndex + 1;
				}
			} else {
				if (toVisitOffset == 0) { break; }
				currentNodeIndex = nodesToVisit[--toVisitOffset];
			}
		}

		if (hit_geom_index == -1) {
			intersections[path_index].t = -1.0f;
		} else {
			//The ray hits something
			Material material = mats[geoms[hit_geom_index].materialid];
			if (material.normalOffset != -1) {
				float w = material.norWidth - 1;
				float h = material.norHeight - 1;
				float u = uv[0];
				float v = uv[1];
				v = v < 0.5 ? v + 2 * (0.5 - v) : v - 2 * (v - 0.5);
				float coordU = w * u;
				float coordV = h * v;
				if (coordV > material.norHeight || coordV < 0) {
					printf("v out of bounds of texture %d x %d: %f with uv %f\n", material.norWidth, material.norHeight, coordV, v);
				}
				glm::vec3 first = slerp(getAlpha(coordU, glm::ceil(coordU), glm::floor(coordV)),
										fetchColor(textureData, material, glm::ceil(coordU), glm::ceil(coordV), false),
										fetchColor(textureData, material, glm::floor(coordU), glm::ceil(coordV), false));
				glm::vec3 second = slerp(getAlpha(coordU, glm::ceil(coordU), glm::floor(coordV)),
										 fetchColor(textureData, material, glm::ceil(coordU), glm::floor(coordV), false),
										 fetchColor(textureData, material, glm::floor(coordU), glm::floor(coordV), false));
				glm::vec3 texCol = slerp(getAlpha(coordV, glm::ceil(coordV), glm::floor(coordV)), first, second);
				texCol = fetchColor(textureData, material, glm::floor(coordU), glm::floor(coordV), false);
				glm::vec3 mapNor = texCol * glm::vec3(2.f) - glm::vec3(1.f);
				// tangent space stuff
				glm::vec3 nor;
				glm::vec3 tan;
				glm::vec3 bit;
				Geom geom = geoms[hit_geom_index];
				computeTBN(geom, intersect_point, &nor, &tan, &bit);

				nor = glm::normalize(multiplyMV(geom.transform, glm::vec4(nor, 0.f)));
				tan = glm::normalize(multiplyMV(geom.transform, glm::vec4(tan, 0.f)));
				bit = glm::normalize(multiplyMV(geom.transform, glm::vec4(bit, 0.f)));
				glm::mat3 TBN = glm::mat3(tan, bit, nor);
				normal = TBN * mapNor;
			}

			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].uv = uv;
		}
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
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		glm::vec2 uv(0);
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE) {
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmp_uv);
			} else if (geom.type == SPHERE) {
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmp_uv);
			} else if (geom.type == TRIANGLE) {
				t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmp_uv);
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
				uv = tmp_uv;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].uv = uv;
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

// Warp functions
__forceinline__
__host__ __device__
glm::vec3 squareToDiskConcentric(const glm::vec2 &sample) {
	glm::vec2 warp = sample * glm::vec2(2.f) - glm::vec2(1.f);
	if (warp.x == 0 && warp.y == 0) {
		return glm::vec3(0.f);
	}
	float theta, r;
	if (std::abs(warp.x) > std::abs(warp.y)) {
		r = warp.x;
		theta = (PI / 4.f) * (warp.y / warp.x);
	} else {
		r = warp.y;
		theta = (PI / 2.f) - (PI / 4.f) * (warp.x / warp.y);
	}
	return glm::vec3(r * std::cos(theta), r * std::sin(theta), 0);
}

__forceinline__
__host__ __device__
glm::vec3 squareToHemisphereCosine(const glm::vec2 &sample) {
	glm::vec3 diskSample = squareToDiskConcentric(sample);
	float z = glm::sqrt(glm::max(0.f, 1.f - diskSample.x * diskSample.x - diskSample.y * diskSample.y));
	return glm::vec3(diskSample.x, diskSample.y, z);
}

__forceinline__
__host__ __device__
float squareToHemisphereCosinePdf(const glm::vec3 &sample) {
	float d = glm::sqrt(sample.x * sample.x + sample.y * sample.y);
	return INV_PI * glm::cos(glm::asin(d));
}

__forceinline__
__host__ __device__
float AbsDot(const glm::vec3 &v1, const glm::vec3 &v2) {
	return glm::abs(glm::dot(v1, v2));
}

// BSDF Functions
// Pass in wo and a material, isect, and sample
// fills in wi, pdf, f

__host__ __device__
void bxdfLambertDiffuse(const glm::vec3 &wo, Material &mat, ShadeableIntersection &isect, thrust::default_random_engine &rng,
						glm::vec3 *wi, glm::vec3 *f, float *pdf) {
	*wi = calculateRandomDirectionInHemisphere(isect.surfaceNormal, rng);
	//if (wo.z < 0) { wi->z *= -1; }
	*pdf = AbsDot(isect.surfaceNormal, *wi) * INV_PI;
	*f = mat.color;
}

__host__ __device__
void bxdfPerfectSpecular(const glm::vec3 &wo, Material &mat, ShadeableIntersection &isect, thrust::default_random_engine &rng,
						 glm::vec3 *wi, glm::vec3 *f, float *pdf) {
	*wi = glm::reflect(-wo, isect.surfaceNormal);
	*pdf = 1.f;
	*f = mat.color;
}

__global__
void shadeMaterial(int iter, int numPaths,
				   ShadeableIntersection *shadeableIntersections,
				   PathSegment *pathSegments,
				   Material *materials, glm::vec3 *textureData, int depth)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numPaths || pathSegments[idx].remainingBounces <= 0) { return; }
	ShadeableIntersection intersection = shadeableIntersections[idx];
	if (intersection.t > 0.0f) { // if the intersection exists...
		// Set up the RNG
		// LOOK: this is how you use thrust's RNG! Please look at
		// makeSeededRandomEngine as well.
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
		thrust::uniform_real_distribution<float> u01(0, 1);

		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;

		if (material.textureOffset != -1) {
			// texture exists for this material
			float w = material.texWidth - 1;
			float h = material.texHeight - 1;
			float u = intersection.uv[0];
			float v = intersection.uv[1];
			v = v < 0.5 ? v + 2 * (0.5 - v) : v - 2 * (v - 0.5);
			float coordU = w * u;
			float coordV = h * v;
			if (coordV > material.texHeight || coordV < 0) {
				printf("v out of bounds of texture %d x %d: %f with uv %f\n", material.texWidth, material.texHeight, coordV, v);
			}
			glm::vec3 first = slerp(getAlpha(coordU, glm::ceil(coordU), glm::floor(coordV)),
									fetchColor(textureData, material, glm::ceil(coordU), glm::ceil(coordV)), 
									fetchColor(textureData, material, glm::floor(coordU), glm::ceil(coordV)));
			glm::vec3 second = slerp(getAlpha(coordU, glm::ceil(coordU), glm::floor(coordV)),
									 fetchColor(textureData, material, glm::ceil(coordU), glm::floor(coordV)),
									 fetchColor(textureData, material, glm::floor(coordU), glm::floor(coordV)));
			glm::vec3 texCol = slerp(getAlpha(coordV, glm::ceil(coordV), glm::floor(coordV)), first, second);
			texCol = fetchColor(textureData, material, glm::floor(coordU), glm::floor(coordV));
			pathSegments[idx].color *= texCol;
		}

		// If the material indicates that the object was a light, "light" the ray
		if (material.emittance > 0.0f) {
			pathSegments[idx].color *= (materialColor * material.emittance);
			pathSegments[idx].remainingBounces = 0;
		} else {
			// Evaluate the BRDF (done in scatterRay)
			scatterRay(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, material, rng);
			pathSegments[idx].remainingBounces--;
		}
	} else {
		pathSegments[idx].color = glm::vec3(0.0f);
		pathSegments[idx].remainingBounces = 0;
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

struct endPath {
	__host__ __device__ bool operator()(const PathSegment &pathSegment) {
		return pathSegment.remainingBounces > 0;
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

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
	while (!iterationComplete) {

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// tracing
	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

	#if KDTREE
	computeIntersectionsKD << <numblocksPathSegmentTracing, blockSize1d >> > (
		depth, num_paths, dev_paths, dev_geoms, dev_kdtree, hst_scene->geoms.size(),
		dev_intersections, dev_materials, dev_tex
		);
	#else
	computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
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


	// TODO:
	// --- Shading Stage ---
	// Shade path segments based on intersections and generate new rays by
	// evaluating the BSDF.
	// Start off with just a big kernel that handles all the different
	// materials you have in the scenefile.
	// TODO: compare between directly shading the path segments and shading
	// path segments that have been reshuffled to be contiguous in memory.

	//shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
	//  iter,
	//  num_paths,
	//  dev_intersections,
	//  dev_paths,
	//  dev_materials
	//);
	shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
		iter,
		num_paths,
		dev_intersections,
		dev_paths,
		dev_materials, dev_tex, depth
	);
	checkCUDAError("shading");

	cudaDeviceSynchronize();

#if STREAM_COMPACT
	dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, endPath());
	auto numPaths = (dev_path_end - dev_paths);

	iterationComplete = numPaths == 0 || depth == traceDepth;
#else
	iterationComplete = depth == traceDepth; // TODO: should be based off stream compaction results.
#endif
	}

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
