CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Zach Corse
  * LinkedIn: https://www.linkedin.com/in/wzcorse/
  * Personal Website: https://wzcorse.com
  * Twitter: @ZachCorse
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 32GB, NVIDIA GeForce GTX 970M (personal computer)

## README

![gif](images/flocking.gif)

Introduction
------------

I'm constructing a pathtracer on the GPU using CUDA. Pathtracing models physically-based light interactions in a scene by casting rays into the scene, testing for intersections, then shading using a variety of physically-based scattering models. My particular implementation handles box, sphere, and triangle intersections, and can simulate diffuse, reflective, and refractive surfaces. What differentiates these models is their BxDFs, which are characterized by unique probability distribution functions that encode the probability that a ray is re-directed in a given direction as a function of angle of incidence and surface normal. Rays in the scene are launched with a "bounce depth", which is simply the number of bounces they have to reach an emissive source (light) before they are terminated with the color black. If they do reach a light, the pixels they were launched from gets the color they've accumulated by interacting with various surfaces in the scene.

Various optimizations speed the processing of parallel rays, and I've included several here. The first handles the early termination of rays. Within a given iteration (a max of 8 bounces by default), a ray might terminate early. This happens for one of two possible reasons: the first is the ray hits an emissive source. This means its path to the camera is complete and can therefore be terminated early. The second is the ray does not intersect any geometry in the scene. This means the photon ray will have no further interactions in the scene and will not subsequently reach a light (it is lost to the aether). It would be wasteful to launch dead rays in the various kernels called within bounceOneRay() (which runs up to ray bounce depth times within a given iteraction). Stream compaction reorders rays in a buffer according to a simple check -- is the ray dead or not? If it isn't, it is moved to the front of the buffer. We maintain a pointer to the transition index between alive and dead rays then only launch rays in subsequent iterations up to this index.

A second optimization handles variable shading computation times, which vary according to the material the ray intersects. Within a warp, if one ray intersects a diffuse surface and another intersects a refractive one, the time it takes to compute the ray's new direction and color varies considerably (diffuse is simpler). It might therefore be prudent to first sort rays by the materials they've intersected, such that warps handling simpler interactions (eg diffuse) can be freed earlier and directed to other tasks. Consequently, my implementation includes a toggleable radix sort by key (material id). As shown below, *in general* this does not provide the speedup one might expect.

A final feature / optimization I include checks ray intersections with arbitrary OBJ mesh bounding boxes before testing more computationally intensive triangle intersections. If a ray does intersect the mesh's bounding box, it is then naively checked against all triangles in the mesh. A more computationally efficient approach would be store the mesh in a KD-tree, but I haven't managed to implement this just yet. Finally, I use [tiny_obj_loader][1] to handle the broad span of OBJ mesh qualities found in open source libraries.

[1]: https://github.com/syoyo/tinyobjloader

Features
------------

Present features included in my pathtracer. See below for sample renders and performance analysis!

1. Basic Pathtracing
  * Diffuse BxDF scattering
  * Scene construction
  * Sphere and Box intersections
2. Terminated Ray Stream Compaction
  * Per iteration
  * Only launches ray threads that have not yet terminated
3. Material sorting
  * Sorts ray intersections pre-shading
  * Warp-aware
4. Reflection BxDF
  * Perfect specular reflection
5. Refraction BxDF
  * Includes Fresnel effects using Schlick's approximation
6. Anti-aliasing
  * Camera rays are jittered, enhancing scene convergence
  * Improves convergence at the cost of first bounce caching
7. OBJ mesh loading with tiny_obj_loader
  * Supports all mesh types
  * Includes mesh bounding box intersection test optimization
  * Uses ray-triangle intersection methods
  * Barycentric normal interpolation
  * Supports all BxDFs mentioned above
  * Does not include textures ... yet
  * Does not include KD-tree acceleration ... yet
  * Without acceleration, practical limitations on mesh size
 
Sample Renders
------------

![pic1](renders/sample.png)
*Standard scene. Cornell box with diffuse materials.*

![pic2](renders/sample.png)
*More complex scene with diffuse materials.*

![pic3](renders/sample.png)
*Ideal specularly reflective spheres.*

![pic4](renders/sample.png)
*Refractive spheres using Schlick's approximation.*

![pic5](renders/sample.png)
*Arbitrary mesh loading.*

Stream Compaction Analysis
------------

Material Sorting Analysis
------------

Anti-Aliasing Comparison
------------

With AA                    |  Solarized Ocean
:-------------------------:|:-------------------------:
![pic6](graphs/sample.png) |  ![pic7](graphs/sample.png)














