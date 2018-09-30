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
