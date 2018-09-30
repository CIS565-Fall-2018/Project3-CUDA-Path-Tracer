CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Angelina Risi
  * [LinkedIn](www.linkedin.com/in/angelina-risi)
  * [Twitter](https://twitter.com/Angelina_Risi)
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 8GB, GTX 960M 4096MB (Personal Laptop)

### (TODO: Your README)

![Material Sample Scene](img/SampleScene1.2018-09-30_01-04-41z.2220samp.png)
![Refelction and Refraction](img/ReflectRefractTest684sample.png)


#### Depth-of Field  
  
![Depth of Field](img/big_DOF.2018-09-30_02-33-09z.770samp.png)
  
Depth of field is the effect in which objects further away from the point where the eye is focused become blurry, less well-defined, than those closer to the focal point. This is done by jittering the cast rays' origins in an "aperture" radius and recalculating the direction toward the originally aimed focal point. This method causes greater distortion of the ray from the target point the further it is from the focal point.  
To customize depth-of-field effects there are two main variables: the focal length and aperture radius. Currently the focal length, which determines the focal point of each ray, is determined in code as the difference between the camera position and "Look-At" point, while the aperture radius is defined as a constant (located in path_helpers.h). Increasing or decreasing the focal length moves the curved "focal plane" while changing the aperture size affects the range of jitter, making the image more or less blurry outside of the focal plane.  
Due to the nature of this feature, it cannot be properly implemented while caching the first bounce.  
This feature adds a little more overhead in generating new rays from the camera at the begining of each iteration by adding more instructions per thread and memory access. This feature is toggleable from a defined boolean in path_helpers.h. In a hypothetical CPU implementation, the instructions would essentially be the same, but done sequentially in a loop. This means the cpu would need to generate 2 times the number of pixels random numbers, in sequence, and apply a pair of them to ach ray and recalculate the direction. This would be quite inefficent even for this seemingly small task versus parallel implementations. Since the code itself is short, very few more GPU optimizations can be imagined, but perhaps speeding up memory access such as through using shared memory would be useful.

#### Materials  
  
##### Perfect Specular (Reflective)  
  
(insert image here)
  
Perfectly reflective surfaces reflect the incident ras perfectly around the surface normals of the object. In code, the specular color is sampled and the reflected direction calculated from the incident and normal vectors. This creates a mirror-like effect on the object surface.
