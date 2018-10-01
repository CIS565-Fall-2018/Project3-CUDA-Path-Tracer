CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Ziad Ben Hadj-Alouane
  * [LinkedIn](https://www.linkedin.com/in/ziadbha/), [personal website](https://www.seas.upenn.edu/~ziadb/)
* Tested on: Windows 10, i7-8750H @ 2.20GHz, 16GB, GTX 1060

# Project Goal
<p align="center">
  <img src="https://github.com/ziedbha/Project3-CUDA-Path-Tracer/blob/master/images/top.png"/>
</p>

In this project, I implemented a Path Tracer using the CUDA parallel computing platform. Path Tracing is a computer graphics Monte Carlo method of rendering images of three-dimensional scenes such that the global illumination is faithful to reality. 

# Path Tracing Intro
<p align="center">
  <img width="300" height="400" src="https://github.com/ziedbha/Project3-CUDA-Path-Tracer/blob/master/images/explanation.png"/>
</p>
In simple terms, a path tracer fires rays from each pixel, which would bounce off in many directions depending on the objects in the scene. If the ray is fortunate enough, it would hit an illuminating surface (lightbulb, sun, etc...), and would cause the first object it hit to be illuminated, much like how light travels towards our eyes.

## Issues
Path tracing is computationally expensive since for each pixel, our rays might hit numerous geometries along the way. Checking intersections with each geometry is expensive, which is why we employ optimization techniques (culling, caching, stream compaction, material sorting) as well as use a parallel computing platform (CUDA) to take advantage of the GPUs cores.

## Stream Compaction, Caching, & Sorting Optimizations
### Stream Compaction

### Caching

### Material Sorting

## 3D Model Importing (.obj)

## Anti-Aliasing
| Without Anti-Aliasing | With Anti-Aliasing | 
| ------------- | ----------- |
| ![](images/aa_none.png) | ![](images/aa.png) |

In computer graphics, anti-aliasing is a software technique used to diminish stair-step like lines that should be smooth instead. This usually occurs when the resolution isn't high enough. In path tracing, we apply anti-aliasing by firing the ray with additional noise. If the ray is supposed to color pixel (x,y), we sample more colors around that pixel, and average them out to color pixel (x,y).

The picture below explains the approach. My implementation is represented by the image on the right.

<p align="center">
  <img width="300" height="100" src="https://github.com/ziedbha/Project3-CUDA-Path-Tracer/blob/master/images/aa_exp.png"/>
</p>

## Depth of Field

| Without Depth of Field | With Depth of Field | 
| ------------- | ----------- |
| ![](images/dof_none.png) | ![](images/dof.png) |

In optics, depth of field is the distance about the plane of focus where objects appear acceptably sharp in an image. In path tracing, this is achieved by adding noise to the ray direction about the focal point, as explained in this [article](https://medium.com/@elope139/depth-of-field-in-path-tracing-e61180417027).




## Materials Support
### Diffuse
### Specular Reflective
### Specular Refractive
### Transmissive using Schlick's Approximation

# Bloopers & Bugs

Many more features can extend my implementation (such as texture importing, normal mapping, subsurface scattering, etc...).

# Build Instructions
1. Install [CMake](https://cmake.org/install/)
2. Install [Visual Studio 2015](https://docs.microsoft.com/en-us/visualstudio/welcome-to-visual-studio-2015?view=vs-2015) with C++ compativility
3. Install [CUDA 8](https://developer.nvidia.com/cuda-80-ga2-download-archive) (make sure your GPU supports this)
4. Clone this Repo
5. Create a folder named "build" in the root of the local repo
6. Navigate to the build folder and run "cmake-gui .." in a CLI
7. Configure the build with Visual Studio 14 2015 Win64, then generate the solution
8. Run the solution using Visual Studio 2015. Build cis_565_path_tracer.
9. Run cis_565_path_tracer with command line arguments: ../scenes/name_of_scene.txt
