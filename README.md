CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Yichen Shou
  * [LinkedIn](https://www.linkedin.com/in/yichen-shou-68023455/), [personal website](http://www.yichenshou.com/)
* Tested on: Windows 10, i7-2600KU @ 3.40GHz 16GB RAM, NVIDIA GeForce GTX 660Ti 8GB (Personal Desktop)

## Project Showcase

![](img/compactExplanation.PNG)

## Project Overview

- what is path tracer
- why do it on the gpu

## Features Implemented
- path tracing
- reflective materials
- refraction w/ caustics
- anti aliasing w/ pictures
- custom obj loaded -> teapot

- for extra features write about
Overview write-up of the feature
Performance impact of the feature
If you did something to accelerate the feature, what did you do and why?
Compare your GPU version of the feature to a HYPOTHETICAL CPU version (you don't have to implement it!) Does it benefit or suffer from being implemented on the GPU?
How might this feature be optimized beyond your current implementation?

## Performance analysis
---tested on 1200 x 1200 picture of a teapot with 3 spheres. 10 objs and 10 materials total. averaged over 10 iterations not counting the first one
- no optimization - 1447 to 1290
- material sorting 1675 to 1290
- first bounce cache 1450 to 1000
- obj loading bounding box vs. no bounding box 1250 to 980

- all diffuse  1050-900
all reflect - 770 - 620
all refract- 520 - 370

## sources used
- tinyObj loader
- PBR
- 565 slides
- youtube vid about schlick
this site(https://people.sc.fsu.edu/~jburkardt/data/obj/obj.html) for obj files