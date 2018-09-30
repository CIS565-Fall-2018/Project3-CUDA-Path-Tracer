CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

## Path tracing with CUDA on the GPU
### Connie Chang
  * [LinkedIn](https://www.linkedin.com/in/conniechang44), [Demo Reel](https://www.vimeo.com/ConChang/DemoReel)
* Tested on: Windows 10, Intel Xeon CPU E5-1630 v4 @ 3.70 GHz, GTX 1070 8GB (SIG Lab)

![](builds/direct.F.2018-09-29_00-48-35z.5000samp.png)  
A render with full lighting

## Introduction
The goal of this project is to write a path tracer that runs on the GPU using CUDA. Each thread of the GPU followed a ray, and updated color calculations as it bounced in the scene. Three different integration algorithms were implemented: Naive, Direct Lighting, and Full Lighting. In addition, this project supports three light scattering models: Diffuse Lambert, reflective, and refractive. The latter two use a Fresnel dielectric computation to accurately simulate the ratio of reflection to refraction. The three scattering methods can be combined in any way to create interesting materials. To make the final image look nicer, anti-aliasing was added by jittering the rays as they left the camera.  
  
Some optimizations were added to speed up the render. To reduce warp divergence, dead rays are partitioned out. Therefore, the dead rays do not occupy any threads and the live rays are bundled together in warps. Furthermore, there is an option to cache the first bounce of the first ray casted from each pixel, reducing the computations required for the first bounce on subsequent samples. However, this cache does not work with anti-aliasing. Lastly, there is another option to sort materials such that rays hitting the same material are bundled together. Unfortunately, this did not provide much of an optimization.  

## Implementation Details

## Usage and Scene File Description
An example scene file looks like this:
```
INTEGRATOR F
MATERIALSORT 0
FIRSTCACHE 0
ANTIALIAS 1

// Emissive material (light)
MATERIAL 0
RGB         1 1 1
SPECEX      0
SPECRGB     0 0 0
DIFFUSE     1
REFL        0
REFR        0
REFRIOR     0
EMITTANCE   5

// Diffuse white
MATERIAL 1
RGB         .98 .98 .98
SPECEX      0
SPECRGB     0 0 0
DIFFUSE     1
REFL        0
REFR        0
REFRIOR     0
EMITTANCE   0

// Diffuse red
MATERIAL 2
RGB         .85 .35 .35
SPECEX      0
SPECRGB     0 0 0
DIFFUSE     1
REFL        0
REFR        0
REFRIOR     0
EMITTANCE   0

// Diffuse green
MATERIAL 3
RGB         .35 .85 .35
SPECEX      0
SPECRGB     0 0 0
DIFFUSE     1
REFL        0
REFR        0
REFRIOR     0
EMITTANCE   0

// Specular white
MATERIAL 4
RGB         .98 .98 .98
SPECEX      0
SPECRGB     .98 .98 .98
DIFFUSE     0
REFL        1
REFR        1
REFRIOR     1.6
EMITTANCE   0

// Camera
CAMERA
RES         800 800
FOVY        45
ITERATIONS  5000
DEPTH       8
FILE        direct
EYE         0.0 5 10.5
LOOKAT      0 5 0
UP          0 1 0


// Ceiling light
OBJECT 0
squareplane
material 0
TRANS       0 9.9 0
ROTAT       90 0 0
SCALE       3 3 1

// Floor
OBJECT 1
cube
material 1
TRANS       0 0 0
ROTAT       0 0 0
SCALE       10 .01 10

// Ceiling
OBJECT 2
cube
material 1
TRANS       0 10 0
ROTAT       0 0 90
SCALE       .01 10 10

// Back wall
OBJECT 3
cube
material 1
TRANS       0 5 -5
ROTAT       0 90 0
SCALE       .01 10 10

// Left wall
OBJECT 4
cube
material 2
TRANS       -5 5 0
ROTAT       0 0 0
SCALE       .01 10 10

// Right wall
OBJECT 5
cube
material 3
TRANS       5 5 0
ROTAT       0 0 0
SCALE       .01 10 10

// Sphere
OBJECT 6
sphere
material 4
TRANS       -1 4 -1
ROTAT       0 0 0
SCALE       1.5 1.5 1.5
```

The first section sets some flags for the render. The first line tells the path tracer which integrator to use. 'N' is for Naive, 'D' is for Direct Lighting, and 'F' is for Full Lighting. The second flag turns material sorting on or off, with 1 being on and 0 being off. Next is the flag for caching the first bounce. Last is the flag for turning on anti-aliasing. Note that if anti-aliasing is turned on, the first bounce cache will be turned off regardless of what the scene file says. The following configuration uses the Full Lighting integrator with anti-aliasing, while material sort and first bounce cache are turned off.  
```
INTEGRATOR F
MATERIALSORT 0
FIRSTCACHE 0
ANTIALIAS 1
```

Next, the materials are described. The first line tells the program that it is about to read a material and gives it a unique ID number. Then, we have color ranging from 0 - 1, the specular exponent, and the specular color. The next three are flags for whether this material has diffuse, reflective, or refractive components. REFRIOR is the index of refraction. Lastly, EMITTANCE is how much light this material emits if it emits any at all. The following example has a material ID of 4, color and specular color of (0.98, 0.98, 0.98), and a specular exponent of 0. It does not have a diffuse component, but has both reflection and refraction, creating a glass-like material. It's index of refraction is 1.6, and it does not emit any light.  
```
MATERIAL 4
RGB         .98 .98 .98
SPECEX      0
SPECRGB     .98 .98 .98
DIFFUSE     0
REFL        1
REFR        1
REFRIOR     1.6
EMITTANCE   0
```

Then, there is the camera description. The first line tells the program that it's about to read camera information. That is followed by the camera resolution, field of view vertically, the number of samples per pixel, the depth of each ray sample, and the file name to store the image. Finally, it has the position of the camera, the point it is looking at, and the direction of its up vector.  
```
CAMERA
RES         800 800
FOVY        45
ITERATIONS  5000
DEPTH       8
FILE        direct
EYE         0.0 5 10.5
LOOKAT      0 5 0
UP          0 1 0
```

The last section describes the geometry in the scene. The first line says we are reading a geometry OBJECT and gives it a unique ID number. The second line says what kind of geometry this is. This project supports squareplane, cube, and sphere. Next is this geometry's material, using the material ID. The last three lines are the transformations applied to this geometry.  
```
OBJECT 0
squareplane
material 0
TRANS       0 9.9 0
ROTAT       90 0 0
SCALE       3 3 1
```

## Performance
