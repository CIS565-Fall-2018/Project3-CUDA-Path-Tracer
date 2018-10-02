CUDA Path Tracer
======================

* Salaar Kohari
  * LinkedIn ([https://www.linkedin.com/in/salaarkohari](https://www.linkedin.com/in/salaarkohari))
  * Website ([http://salaar.kohari.com](http://salaar.kohari.com))
  * University of Pennsylvania, CIS 565: GPU Programming and Architecture
* Tested on: Windows 10, Intel Xeon @ 3.7GHz 32GB, GTX 1070 8GB (SIG Lab)

(GIF of raytracer in real-time)

### Introduction
The GPU path tracer I implemented produces accurate renders in real-time. The rays are scattered using visually accurate diffuse, reflection, and refraction lighting properties. Techniques such as stream compaction and particular memory allocation help speed up the algorithm as new features are added. Other features of the path tracer include arbitrary mesh loading and anti-aliasing.

Some terms will be important for understanding the analysis. Each ray cast from the camera has a maximum number of **bounces** carrying the light before it terminates. When every pixel's non-deterministic path reaches the maximum bounces or does not collide with anything in the scene, one **iteration** is completed. Performance analysis will focus on number of bounces and average iteration time as features are added.

## Algorithm
1. First step
2. Second step
3. Third Step

## Images
(Screenshots of diffuse, reflective, refractive cubes)

### Analysis
Num_paths at each bounce, up to 12 bounces
Iteration time with/without stream compaction
Iteration time with/without anti-aliasing
SC+AA, AA, SC
Iteration time for cornell box vs obj cube vs obj cube bounding box
