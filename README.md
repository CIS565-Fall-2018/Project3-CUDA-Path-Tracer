CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Jie Meng
  * [LinkedIn](https://www.linkedin.com/in/jie-meng/), [twitter](https://twitter.com/JieMeng6).
* Tested on: Windows 10, i7-7700HQ @ 2.80GHz, 16GB, GTX 1050 4GB (My personal laptop)




Path Tracing 101
================
In general, path tracing is a render technique that simulates the interactions between rays from light sources and objects in the scene.

//Desire image for path tracing
Like illustrated in the image above, we essentially: 
* 1. Shoot rays rays from the positions of each pixel, through camera, to the scene;
* 2. If a ray hits an object, then based on the material of the object's surface:
** 2.1. The ray bounces off randomly if the object is of diffusive material, which basically creates a new ray starts from the intersection with direction randomly distributed in a hemisphere;
** 2.2. The ray bounces off only through the reflective direction if the object is of reflective material;
** 2.3. The ray enters the object if the object is refractive, its direction changes according to the object's index of refraction (eta).
* 3. Repeat step 2. where rays keep bouncing off from a surface to another surface. 
* 4. A ray is terminated if:
** 4.1. It reaches a light source;
** 4.2. It hits nothing, means it will not do any contribution to the result image;
** 4.3. It has bounced off enough times (or so-called depth), you can think this as a ray's energy is all consumed. 
* 5. When all rays are terminated, we gather and average their colors and create a image based on it.

