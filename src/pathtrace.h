#pragma once

#include <vector>
#include "scene.h"

#define COMPACT 0
#define SORTING 0
#define CACHING 0
#define TIMING 0
#define SORTTIMING 0

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
