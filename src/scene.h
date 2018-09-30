#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
	int loadMesh(string filename, std::vector<Geom> &tris);
    int loadCamera();

	KDHelperNode *buildKDTreeCPU(std::vector<Geom> &geoms, int depth, int maxDepth);
	int flattenKDTree(KDHelperNode *root, std::vector<Geom> &sortedGeoms, std::vector<KDNode> &kdtree, int *offset);
	int countNodes(KDHelperNode *node);
	float cost(float split, const std::vector<Geom> &geoms, int axis, float min, float max);
	float split(const std::vector<Geom> &geoms, int axis, float min, float max);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
	std::vector<Geom> sortedGeoms; // for flattened kd tree
	std::vector<KDNode> kdtree;
    std::vector<Material> materials;
	std::vector<glm::vec3> textureData;
    RenderState state;
};

__host__ __device__
Bounds Union(const Bounds &b1, const Bounds &b2);
Bounds GetBounds(Geom geo);
Bounds GetBounds(std::vector<Geom> &geoms);
int longestAxis(Bounds &b);
glm::vec3 getMidpoint(Bounds &b);