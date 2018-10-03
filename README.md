CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Jie Meng
  * [LinkedIn](https://www.linkedin.com/in/jie-meng/), [twitter](https://twitter.com/JieMeng6).
* Tested on: Windows 10, i7-7700HQ @ 2.80GHz, 16GB, GTX 1050 4GB (My personal laptop)


![](img/DIAMOND3000.png)


Path Tracing 101
================

In general, path tracing is a render technique that simulates the interactions between rays from light sources and objects in the scene.

![](img/raytracing/png)

Like illustrated in the image above, we essentially: 
1. Shoot rays rays from the positions of each pixel, through camera, to the scene;
2. If a ray hits an object, then based on the material of the object's surface:
* 1. The ray bounces off randomly if the object is of diffusive material, which basically creates a new ray starts from the intersection with direction randomly distributed in a hemisphere;
* 2. The ray bounces off only through the reflective direction if the object is of reflective material;
* 3. The ray enters the object if the object is refractive, its direction changes according to the object's index of refraction (eta).
3. Repeat step 2. where rays keep bouncing off from a surface to another surface. 
4. A ray is terminated if:
* 1. It reaches a light source;
* 2. It hits nothing, means it will not do any contribution to the result image;
* 3. It has bounced off enough times (or so-called depth), you can think this as a ray's energy is all consumed. 
5. When all rays are terminated, we gather and average their colors and create a image based on it.

From the path-tracing procedures described above, we can get the sense that 'Path Tracing' is trying to simulate the stochastic phenomenon when light hits objects, which is also known and described by [BSDFs](https://en.wikipedia.org/wiki/Bidirectional_scattering_distribution_function), by shooting a lot of rays a lot of times into the scene, and approximate the continuous probability distribution by enough large number of discrete sampling. The best about Path Tracing is that the rays coming about from different pixels are independent, so we can use the power of GPU to run multiple threads, each carrying a ray, simultaneously on GPU.

![](img/bsdf.png)


Highlighted Renderings
==================

Bunny 

![](img/TWOBUNNY2200.png)

Colorful diamonds:

![](img/DIAMOND3000.png)

![](img/DIAMOND1000.png)


Other models

![](img/GOAT1000.png)

![](img/WOLF600.png)

Different BSDFs
==================

The material of an object is represented by its surface BSDFs:

 * Walls usually appears to be purely diffusive, meaning the incident light is uniformly scattered in all directions, so there is no "shiny" visual effect from any angle;
 * Mental materials can be approximated by pure specular BSDF: the incident light entirely bounces off from the reflective direction;
 * Glass and water are described by both reflective and refractive effects. Incident rays are splited into two parts: one goes out in reflective direction, another enters the material. The energy split scheme is described by its [Fresnel Equations](https://en.wikipedia.org/wiki/Fresnel_equations), which usually can be approximated by [Schlick's approximation](https://en.wikipedia.org/wiki/Schlick's_approximation).


![](img/BXDF2000.png)

The above image shows four material configurations: pure diffusive, pure reflective, reflective & refractive and reflective & refractive with higher index of refraction.

![](img/BXDF5001.png)

Same scene with smaller number of iterations and slightly different view angle.

![](img/REFRAC3/png)

A almost pure refractive sphere. Does it reminds you of [something](https://www.google.com/search?q=nintendo+switch&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiFmvWHlundAhVNxVkKHWeUC8cQ_AUIESgE&biw=918&bih=364#imgrc=g27scV2cRmd15M:)? 


Arbitray Mesh Loading
==================

Meshes are .obj files loaded by [tiny_obj_loader](https://github.com/syoyo/tinyobjloader) credited to [Syoyo Fujita](https://github.com/syoyo)

Desired .obj files are in wavefront OBJ file format that contains at least: vertices positions, vertices normals and faces

Mesh rendering can be done with or without [Bounding Box](http://www.idav.ucdavis.edu/education/GraphicsNotes/Bounding-Box/Bounding-Box.html), Bounding Box is a simple optimization method that can restrict only the rays hitting the bounding box to actually check intersections with the mesh. Performance improvement on this can be find in the Analysis [part](#bounding-box-bulling-for-mesh-loading)

![](img/objs.png)

Anti-Aliasing
==================

[Anti-Aliasing](https://en.wikipedia.org/wiki/Spatial_anti-aliasing) is a technique widely used to show sharp edges in renderings. Without AA, sharp edges would appears sawtooth shape on image because the pixels themselves have size. AA solution is path tracing in easy: within each pixel, we jitter the origin of the ray, so that the pixel value has variations, wich is basically what AA needs. 

![](img/AA0.png)

The following pictures show the effect of Anti-aliasing.

| AA ON | AA OFF|
|:-----:|:-------:|
|![](img/AA11.png)|![](img/AA22.png)|

As seen from the enlarged view above: without AA, the sharpe edges become staircases after zoomed in; with AA enabled, the enlarged image still remain plausible.

Optimization Analysis
==================

In order to fully exploit the power of GPU, we have various possible way to optimize the path tracing scheme:

## Stream Compaction path segments

After each bounce, some of the rays would terminate because they hit the light source or void. For the rays that are terminated (while other rays are still running), we could just terminate the thread, or equivalently, run less threads in next iteration.

Inside the code we keep track of the rays, and use stream compaction to compact all the active threads together after every iteration. In the beginning of each iteration, only active rays (or paths) need to be started and computed again. The follow charts shows the performance difference between with and without rays compaction.

## First Bounce Cache

For Path tracing without Anti-aliasing, in the start of every iteration, rays are shot out from the same position as the first iteration. So we could store the resulted intersection information of first bounces from first iteration as a cache, and reuse them in the start of every iteration after that. This way we could save the computation time for first bounces.

## Sort by Material Types

## Bounding Box Culling for Mesh Loading



