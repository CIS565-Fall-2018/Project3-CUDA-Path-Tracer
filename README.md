CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture**

**Anantha Srinivas**
[LinkedIn](https://www.linkedin.com/in/anantha-srinivas-00198958/), [Twitter](https://twitter.com/an2tha)

**Tested on:**
* Windows 10, i7-8700 @ 3.20GHz 16GB, GTX 1080 8097MB (Personal)
* Built for Visual Studio 2017 using the v140 toolkit
---

Introduction
---

Implemented Features
---

1. A lot of structural changes to the code to reflect the architecture as described in **Physically Based Rendering, Second Edition: From Theory To Implementation** by Pharr, Matt and Humphreys, Greg.

2. Naive Integrator.

3. The path tracer supports the following materials:
* Diffuse materials (BTDF)
* Specular (Perfectly specular and glossy) (BRDF)
* Refractive material (BRDF)

4. Rays between each iteration are compacted to remove any dead rays. This is done to improve the performance.

5. The Path segments are sorted by material type. This ensure that almost all warps executed on the GPU have similar execution time.

6. The first intersection is cached for subsequent iterations. This saves redundant calculation on the GPU.

7. Implemented Refraction (the refractive index is assumed to be 1.52, which should be later taken as input from the scene file)

8. Stocahstic sampled AntiAliasing. This jitters the ray direction from camera by a small amount. Over multiple interations, this results in a blurred effect.

9. Implemented Procedural geometry or implicit surfaces. Current supported implicit surfaces include - box, sphere, torus, capped cylinder. The intersection with the geometry is determined using ray marching.

Analysis
---

Comparison in runtimes between CPU vs GPU based Path tracer (for Naive Integrator)



References
---
[PBRT] Physically Based Rendering, Second Edition: From Theory To Implementation. Pharr, Matt and Humphreys, Greg. 2010.