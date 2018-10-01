CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Name: Bowen Yang
  * [LinkedIn](https://www.linkedin.com/in/%E5%8D%9A%E6%96%87-%E6%9D%A8-83bba6148)
  * [GitHub](https://github.com/Grillnov)
  * [Facebook](https://www.facebook.com/yang.bowen.7399)
  * [Steam](https://steamcommunity.com/id/grillnov)
* Tested on: Windows 10 x64, i7-6800K @ 3.40GHz 32GB, GTX 1080 8GB (Personal computer at home)

# Description

Implement a path tracer with CUDA acceleration enabled, capable of rendering globally-illuminated, photo-realistic images at high speeds. The basecode already has the auxiliary services like I/O, keyboard interrrupts, and OpenGL calls. What we're supposed to do is to implement the core of it.

# Part 1: Basics

![](img/Plain.png)
The picture pretty much says it all.

## Performance analysis so far
We choose this default configuration as the benchmark of our performance analysis:
*Anti-aliasing: OFF*
*Depth of Field: OFF*
*1st Cache: OFF*
*Material sort: OFF*

### Performance analysis phase 1: The cached rays

![](img/Cache.png)
As we can see here the caching of the 1st arrays created slightly more overhead and consequently had negative effect on the performance.

### Performance analysis phase 2: The sorting

Thoretically we can expect some performance boost from sorting all the intersections by their material keys to make them continuously scattered in memory, so that the spatial locality becomes better in succeeding phases that calculate the scatters, etc.

However many of our classmates are suffering from severe performance loss due to this sorting operation. Some even faced a 90x execution time increase.

After careful referring the thrust library I found out that, unless sorting primitive integers, thrust::sort() or thrust::sort_by_key() will call its merge-sort subprocedure, not the radix-sort one, although for our purposes radix-sort is the optimal solution. So I headed out and implemented one on my own since it's way too hard to inherit the boolean functors in thrust library.

![](img/Sort.png)

Still some performance loss, but a lot better than using sort in thrust library directly.

# Part 2: Detail of the radix sort implementation



# Part 3: Anti-Aliasing

Jitter the rays with an amplitute taken from thrust::uniform_real_distribution, with a range of (-JITTER_RANGE, JITTER_RANGE). Since we're doing multiple iterations of sampling anyway, the multisampling is automatically

![](img/AA.png)
With jitter = 0.008f.

![](img/AAmessed.png)
It's a little tricky to tackle with the jitter amplitute or you'll end up like this.

Some comparison between AA switched on/off.
![](img/AACloseup.png)

![](img/PlainCloseup.png)
# Part 4: Motion Blur

![](img/MotionBlur.png)
# Part 5: Depth of Field

![](img/DOF.png)
![](img/CloseUpDOF.png)

[Link to the reference for sampling the jitter on the lens](https://pub.dartlang.org/documentation/dartray/0.0.1/core/ConcentricSampleDisk.html)
