CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Angelina Risi
  * [LinkedIn](www.linkedin.com/in/angelina-risi)
  * [Twitter](https://twitter.com/Angelina_Risi)
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 8GB, GTX 960M 4096MB (Personal Laptop)

### (TODO: Your README)

<img src="https://github.com/risia/Project3-CUDA-Path-Tracer/blob/master/img/SampleScene1.2018-09-30_01-04-41z.2220samp.png" width="400"></img><img src="https://github.com/risia/Project3-CUDA-Path-Tracer/blob/master/img/ReflectRefractTest684sample.png" width="400"></img>


### Depth-of Field  
  
![Depth of Field](img/big_DOF.2018-09-30_02-33-09z.770samp.png)
  
Depth of field is the effect in which objects further away from the point where the eye is focused become blurry, less well-defined, than those closer to the focal point. This is done by jittering the cast rays' origins in an "aperture" radius and recalculating the direction toward the originally aimed focal point. This method causes greater distortion of the ray from the target point the further it is from the focal point.  
To customize depth-of-field effects there are two main variables: the focal length and aperture radius. Currently the focal length, which determines the focal point of each ray, is determined in code as the difference between the camera position and "Look-At" point, while the aperture radius is defined as a constant (located in path_helpers.h). Increasing or decreasing the focal length moves the curved "focal plane" while changing the aperture size affects the range of jitter, making the image more or less blurry outside of the focal plane.  
Due to the nature of this feature, it cannot be properly implemented while caching the first bounce.  
This feature adds a little more overhead in generating new rays from the camera at the begining of each iteration by adding more instructions per thread and memory access. This feature is toggleable from a defined boolean in path_helpers.h. In a hypothetical CPU implementation, the instructions would essentially be the same, but done sequentially in a loop. This means the cpu would need to generate 2 times the number of pixels random numbers, in sequence, and apply a pair of them to ach ray and recalculate the direction. This would be quite inefficent even for this seemingly small task versus parallel implementations. Since the code itself is short, very few more GPU optimizations can be imagined, but perhaps speeding up memory access such as through using shared memory would be useful.

### Materials  

#### Absorbancy  
  
Material absorbancy is modelled in refractive or subsurface scattering materials using an exponential function of path length through the material and material absorbancy. Since I did not implement imperfect specular I reused the materials' SPECEX parameter as the absorbance coefficient. Multiplying e<sup>-aL</sup> by the accumulated path color models portions of the light energy being absorbed by the material instead of completely passing through. (insert pictures with and without absorbancy)  
    
#### Perfect Specular (Reflective)  
  
(insert image here)
  
Perfectly reflective surfaces reflect the incident ras perfectly around the surface normals of the object. In code, the specular color is sampled and the reflected direction calculated from the incident and normal vectors. This creates a mirror-like effect on the object surface.
  
#### Perfect Refractive  
  
Refractive materials transmit light through them, but if the refractive index of the material differs from that of the surrounding material (in this case air) the transmitted ray is "bent" in a new angle determined by the ratio of refractive indices and the angle of the incident ray with the surface normal.  
  
#### Imperfect Refractive  (Partially reflective)
  
Materials with both reflective and refractive elements can be modelled by calculating the portions of the incident ray reflected and transmitted and then choosing which path to pursue using this as a probability. The coloring of the material is determined by adding the portion reflected multiplied by specular color with the portion refracted multiplied by diffuse color.  
  
#### Subsurface Scattering  
  
Subsurface scattering is a material effect in which light diffuses into a material and is scattered by particles in the material. This results in a "milkiness" of the material, such as in candle wax or skin. The absorbancy and "scatter length" of the material are important in determining he color in this case. Currently the program uses a fixed defined scatter length, the average length between "scatter" events in the material, but in real materials this variable would depend on its crystal structure and components. A smaller scatter length means more scattering events on average through the material, so fewer photons would make it through in the number of bounces allowed, and there would be more light attenuation from the absorbancy of material due to the increased total path through the material.  
Potentially one could remove the allowed bounce count decrease when scattering in the material to allow more accuracy, but this would increase compute time per iteration and make it more variable.  
This image made it possible for me to wrap my head around the process:
![Subsurface Scatter](https://i.stack.imgur.com/tOp55.jpg)
Source: [Stack Exchange](https://computergraphics.stackexchange.com/questions/5214/a-recent-approach-for-subsurface-scattering) (I did not reference the actual code)

  
  

