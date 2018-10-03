CUDA Path Tracer
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Xiao Zhang
  * [LinkedIn](https://www.linkedin.com/in/xiao-zhang-674bb8148/)
* Tested on: Windows 10, i7-7700K @ 4.20GHz 16.0GB, GTX 1080 15.96GB (my own PC)

Analysis (blocksize1d is set to 128 unchanged, image order is direct light integrator, full light integrator and naive integrator)
======================

## 1. Mat scene 800x800 pixel 200 ppx 8 recursion 

* overview

![](img/my_mat.jpg)

* statistics

![](img/1.JPG)

* images

A. direct light integrator
  
![](img/my_mat.2018-10-03_06-45-23z.2018-10-03_06-45-34z.201_spp.integrator1_compact0_batch0_cache0.png)

B. full light integrator
  
![](img/my_mat.2018-10-03_06-47-02z.2018-10-03_06-47-18z.201_spp.integrator3_compact0_batch0_cache0.png)

C. naive integrator
  
![](img/my_mat.2018-10-03_06-40-15z.2018-10-03_06-40-28z.201_spp.integrator0_compact0_batch0_cache0.png)

## 2. Two light scene 800x800 pixel 200 ppx 64 recursion

* overview

![](img/my_scene.jpg)

* statistics

![](img/2.JPG)

* images

A. direct light integrator
  
![](img/my_scene.2018-10-03_03-31-37z.2018-10-03_03-32-39z.201_spp.integrator1_compact0_batch0_cache0.png)

B. full light integrator
  
![](img/my_scene.2018-10-03_03-24-29z.2018-10-03_03-25-47z.201_spp.integrator3_compact0_batch0_cache0.png)

C. naive integrator
  
![](img/my_scene.2018-10-03_03-12-40z.2018-10-03_03-13-50z.201_spp.integrator0_compact0_batch0_cache0.png)

## 3. Two light scene 800x800 pixel 200 ppx 8 recursion

* overview

![](img/my_scene_8.jpg)

* statistic

![](img/3.JPG)

* images

A. direct light integrator
  
![](img/my_scene_8.2018-10-03_05-21-17z.2018-10-03_05-21-28z.201_spp.integrator1_compact0_batch0_cache0.png)

B. full light integrator
  
![](img/my_scene_8.2018-10-03_04-10-04z.2018-10-03_04-10-21z.201_spp.integrator3_compact0_batch0_cache0.png)

C. naive integrator
  
![](img/my_scene_8.2018-10-03_04-04-10z.2018-10-03_04-04-23z.201_spp.integrator0_compact0_batch0_cache0.png)

## 4. Rex scene 800x800 pixel 200 ppx 8 recursion (28974 triangles in total)

* overview

![](img/my_scene_rex_8.jpg)

* statistic

![](img/4.JPG)

* images

A. direct light integrator
  
![](img/my_scene_rex_8.2018-10-03_05-17-07z.2018-10-03_05-17-29z.201_spp.integrator1_compact0_batch0_cache0.png)

B. full light integrator
  
![](img/my_scene_rex_8.2018-10-03_04-44-58z.2018-10-03_04-51-50z.201_spp.integrator3_compact0_batch0_cache0.png)

C. naive integrator

![](img/my_scene_rex_8.2018-10-03_04-28-46z.2018-10-03_04-31-37z.201_spp.integrator0_compact0_batch0_cache0.png)

## 5. Reflective rex sceen 800x800 pixel 200 ppx 8 recursion (28974 triangles in total)

* overview

![](img/my_scene_rex_r.jpg)

* statistic

![](img/5.JPG)

* images

A. direct light integrator
  
![](img/my_scene_rex_r_8.2018-10-03_06-10-12z.2018-10-03_06-10-31z.201_spp.integrator1_compact0_batch0_cache0.png)

B. full light integrator
  
![](img/my_scene_rex_r_8.2018-10-03_05-44-30z.2018-10-03_05-49-48z.201_spp.integrator3_compact0_batch0_cache0.png)

C. naive integrator
  
![](img/my_scene_rex_r_8.2018-10-03_05-27-43z.2018-10-03_05-30-36z.201_spp.integrator0_compact0_batch0_cache0.png)

## 6. Comparison with CPU path tracer

* overview

