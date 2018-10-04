CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Xinyu Lin
[Linkedin](https://www.linkedin.com/in/xinyu-lin-138352125/)
* Tested on: Windows 10, Intel(R) Core(TM) i7-6700HQ CPU@2.60GHz, 16GB, GTX960M(Private Computer)

![](img/File_001.png)

# Features:
- **Basic features**
  - A shading kernel with BSDF evaluation for:
    - Ideal diffuse shader.
    - perfect specular reflective surface.
  - Stream compaction for terminating unwanted thread from thread pool using thrust::partition
  - material sorting using thrust::sort_by_key
  - caching first bounce information for future iteration use
- **Advanced features**
  - refreaction with Frensel effects using Schlick's approximation 
  - physically based depth of field
  - stochastic sampled antialiasing
  - Arbitrary mesh loading and rendering
  
Diffuse | Specular | Glass
------|------|------
![](img/diffuse.png) | ![](img/specular.png) | ![](img/glass.png)
