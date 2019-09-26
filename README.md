CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Eric Chiu
* Tested on: Windows 10 Education, Intel(R) Xeon(R) CPU E5-1630 v4 @ 3.60GHz 32GB, NVIDIA GeForce GTX 1070 (SIGLAB)


![](./img/procedural-shape-1.png)


## Description

This project implements a physically based pathtracer using CUDA programming and GPU hardware. Features include stream compaction, first bounce caching, material sorting, anti-aliasing, refraction with Frensel effects, and procedural shapes.

## Stream Compaction

Rendering without stream compaction was faster than rendering with stream compaction until the trace depth hit the 100s. This is probably because the time to perform stream compaction was more than the time it saved in the long run when the trace depth was under 100. When the trace depth was over 100, the time to perform stream compaction was marginal compared to the time it saved in the long run.


![](./img/stream-compaction.png)


## First Bounce Caching

First bounce caching depends on the number of intersections found in the first bounce. As the image resolution increases, the more likely intersections are computed in the first bounce. We can see here that caching the first bounce slightly boosts performance, but not a lot. This is probably because the first bounce is only a small percentage of the number of bounces the pathtracer has to perform.


![](./img/first-bounce-caching.png)


## Material Sorting

Sorting rays and path segments based on different materials surprisingly made the performance slower. The idea was to sort rays and path segments so that ones with the same material will be contiguous in memory. However, after further analysis, it seems like the time it takes to material sort outweights the amount of time it saves in later operations.


![](./img/material-sorting.png)


## Anti-Aliasing

Anti-aliasing can be achieved by offsetting the rays by a small amount when it is first shot out of the screen. The pixel will then be an average of the offset rays, creating a smooth effect. We can see the effect take place in the two images below, on the edge on the sphere.


![](./img/aliased.png)

![](./img/anti-aliased.png)


## Refraction

We used Frensel effects with Schlick's approximation to simulate glass and refractive material. In the image below, we see spheres with diffuse, refractive, and reflective materials. Caustics, light focused in one area, is one result of refraction.


![](./img/refraction.png)


## Procedural Shapes

Procedural shapes, specificially implicit surfaces, can be procedurally generated using signed distance functions. When a ray is shot, we can solve a distance function to determine whether the ray hit a implicit surface or not. If we do hit an implicit surface, we perform shading like for any other object. If we do not hit an implicit surface, we continue on with pathtracing. The two implicit surfaces implemented are shown below.


![](./img/procedural-shape-1.png)

![](./img/procedural-shape-2.png)
