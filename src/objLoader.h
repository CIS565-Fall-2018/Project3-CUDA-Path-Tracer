#pragma once
#include <string>
#include "glm/glm.hpp"
#include "sceneStructs.h"

void loadObj(std::string inputfile, int& startTriangleIndex, int& endTriangleIndex, std::vector<Triangle>* triangles);