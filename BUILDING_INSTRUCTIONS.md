CUDA Path Tracer - Building Instructions
======================

Please have a look at pathtrace.cu and intersections.h for flags.

Certain flags enable features. I initially had some stack overflow issues so I decided to stop certain features while I was debugging or implementing them. So, those flags are still active.

For the android scene, you might need to enable Plastic Material using the flags.

If you face similar issues related to overuse of registers, I would recommend either you downgrade the number of threads per block or disable some unused branches.

The final renders are in final_images. There are a ton of scenes available, the names should mostly help you figure out what scenes you need to render.

I had issues with compiling on my Laptop but Desktop ran fine! Not sure what is the issue.


#### KDTree

If you look through my code, you might see a KD Tree. Apparently it work in Debug mode, but doesn't run in Release mode :(

I decided to let that feature go, but I guess you can always try it out in Debug mode if you want to. It was hard to debug and I wasted a full day on it. So, I decided to implement other features instead!