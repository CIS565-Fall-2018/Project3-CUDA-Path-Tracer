#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void sort_by_material(int num_paths, ShadeableIntersection * shadeableIntersections, PathSegment * pathSegments);
