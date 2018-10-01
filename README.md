CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Lan Lou
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 16GB, GTX 1070 8GB (Personal Laptop)

### Sample render

a scene including different materials : **pure reflective**, **diffuse**, and **refraction with reflect** based on Schlick's approximation

![](https://github.com/LanLou123/Project3-CUDA-Path-Tracer/raw/master/img/cornell.2018-09-30_16-46-42z.4007samp.png)

### Glass Dragon 

```resolution``` : 1000X1000 ```iteration``` : 2000 ```render time``` : 25min  ```vertex count``` : 12.5k 

![](https://github.com/LanLou123/Project3-CUDA-Path-Tracer/raw/master/dragonKD2000iter.png )

### Ineractions

![](https://github.com/LanLou123/Project3-CUDA-Path-Tracer/raw/master/img/ddfddddddd.gif )

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
  
  ![](https://github.com/LanLou123/Project3-CUDA-Path-Tracer/raw/master/img/doffix.JPG)
  
  - stochastic sampled antialiasing
  
AA off | AA on
------|------
![](https://github.com/LanLou123/Project3-CUDA-Path-Tracer/raw/master/img/aaoff.JPG)|![](https://github.com/LanLou123/Project3-CUDA-Path-Tracer/raw/master/img/aaon.JPG)
  - arbitrary mesh loading and rendering
    - used [tinyObj loader](http://syoyo.github.io/tinyobjloader/)
    - used glm's triangle intersection method
    - bounding bolumn intersection toggle
### Dragon 

```resolution``` : 1000X1000 ```iteration``` : 2000 ```render time``` : 25min for glass, 28min for specular ```vertex count``` : 12.5k 

Glass dragon | Specular dragon
------|------
![](https://github.com/LanLou123/Project3-CUDA-Path-Tracer/raw/master/dragonKD2000iter.png )|![](https://github.com/LanLou123/Project3-CUDA-Path-Tracer/raw/master/img/dragonspecular2000iter.png )

### Sword

a great sword from dark souls series with simple specular material
![](https://github.com/LanLou123/Project3-CUDA-Path-Tracer/raw/master/img/sword2.png)

  - KD tree for mesh loading acceleration
  
# Performance analysis:

## cache first bounce:

- We want to cache the first iteration's information because before the first bounce, all the information including ray origin ray directions are all the same in spite of different iterations.
- according to the chart bellow, by caching the first bounce, we slightly improved the efficiency

time cost to 500 iterations:

time(ms)	|cached	| no cache
--------------|---------|-------
test scene|	31861.30 |	34134.12

## sort material:
- For different rays, depending on the material of object they hit, the rays will perform differently, and it's always better to put rays that hit the same material closely in the same block instead of those having different materials, this will prevent some unneccesary stalling condition caused by rays' different behaviour(different thread running time).
- In terms of my own test, with sorting toggled on, the program always experience slower performace than before, I guess this is because the number of materials is not big enough to cover the efficiency lost from the sorting operation.

## bounding box for mesh
- it's quite obvious that by adding bounding box to mesh we can prevent unneccessary triangle checks from happenning, resulting in some performance improvement.
- here's my test result (naive)time cost to 100 iterations

time(sec)	|added	| not added
--------------|---------|-------
dragon|	597.20 |	623.11

## KD tree acceleration : 

time cost to 100 iterations

time(secs)	|dragon(12.5 kverts)	|sword (1k verts)
--------------|---------|-------
KD accelerated|	121|	32
Nive|	603|	50

![](https://github.com/LanLou123/Project3-CUDA-Path-Tracer/raw/master/img/kdeffect.JPG )

  - I choosed to implement a KD tree for the acceleration of obj loading, the following are the main features of the tree
    - For the spatial division, I made the current tree node to divide along the longest axis of the node's bounding box, since it minimize the waste of extra divisions,
    - Another thing about division is the magnitude of it, I choosed to use the metric that when the left child node and right child node is having more than half of their triangles in common, we do the division.
    - After we build the tree on cpu, we can trasfer the tree's data to GPU
    - In GPU, since recursion and stl library is unavailable, I instead passed two extra buffers to GPU : first one ```dev_KDtreenode``` which is the array of kdtree nodes each having idx of it's children ,it's parents' index ,bouding box and triangle idx in gpu triangle list, second one : ```dev_gputriidxlst``` which is an array especially for storing triangle indices mapping from kdtree to the actual triangle buffer for each node.
    - as for the target node(containing intersected triangle) searching algorithm, I used a mutation of an iterative in-order binary tree search method and C - style stack, I don't think this is a good solution, but it works and give me no small performance boost, but anyway, I will change to use a better method in the future.

# References
- [KD Trees for Faster Ray Tracing ](https://blog.frogslayer.com/kd-trees-for-faster-ray-tracing-with-triangles/) 
- [ConcentricSampleDisk function](https://pub.dartlang.org/documentation/dartray/0.0.1/core/ConcentricSampleDisk.html)
- [GPU gem3](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_pref01.html)
- [Schlick's approximation wiki](https://en.wikipedia.org/wiki/Schlick's_approximation)
- some iterative solutions for binary search tree 
