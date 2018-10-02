CUDA Path Tracer
======================

* Salaar Kohari
  * LinkedIn ([https://www.linkedin.com/in/salaarkohari](https://www.linkedin.com/in/salaarkohari))
  * Website ([http://salaar.kohari.com](http://salaar.kohari.com))
  * University of Pennsylvania, CIS 565: GPU Programming and Architecture
* Tested on: Windows 10, Intel Xeon @ 3.7GHz 32GB, GTX 1070 8GB (SIG Lab)

![](img/pathtrace.gif)

### Introduction
My GPU path tracer produces accurate renders in real-time. The rays are scattered using visually accurate diffuse, reflection, and refraction lighting properties. Techniques such as stream compaction and particular memory allocation help speed up the iteration time. Other features of the path tracer include arbitrary mesh loading and anti-aliasing.

Some terms will be important for understanding the analysis. Each ray cast from the camera has a maximum number of **bounces** carrying the light before it terminates. When every pixel's non-deterministic path reaches the maximum bounces or does not collide with anything in the scene, one **iteration** is completed. Performance analysis will focus on number of bounces and average iteration time for various features.

## Algorithm
1. First step
2. Second step
3. Third Step

## Images
(Screenshots of diffuse, reflective, refractive)

### Analysis
- Num_paths at each bounce, up to 12 bounces
640000
523159
362017
279091
222367
180137
146772
120227
98634
80858
66523
54544

- SC+AA, AA, SC+CI, SC
28.17 (SC+CI)

- Iteration time for cornell box vs obj cube vs obj cube bounding box
