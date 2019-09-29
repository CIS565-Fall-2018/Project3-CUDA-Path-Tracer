#pragma once

#include "utilities.h"
#include "sceneStructs.h"
#include <iostream>

using namespace std;

class BVHKDTree
{
private:
	BVHKDTree();
public:
	~BVHKDTree();
	static int buildTree(vector<Triangle> &triangles, const int axis, const int indexOffset, const int start, const int end, glm::vec3* min = nullptr, glm::vec3* max = nullptr);
};