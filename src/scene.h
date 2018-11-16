#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

#include"KDtreeNode.h"


using namespace std;


struct GPUKDtreeNode {
	int leftidx;
	int rightidx;
	int depth;
	int GPUtriangleidxinLst;
	int trsize = 0;
	int curidx;
	int parentidx = -1;
	bool isleafnode = false;
	glm::vec3 maxB;
	glm::vec3 minB;
};

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
	bool buildKDtree();
public:
    Scene(string filename);
    ~Scene();

	int globaltricount = 0;
	int nodecount = -1;


	void BuildTreeGPU(KDtreeNode* nn,int cc);
	KDtreeNode *rootnode = new KDtreeNode();
	int meshcount;
	std::vector<mesh> meshs;
    std::vector<Geom> geoms;
	std::vector<Triangle> triangles;
    std::vector<Material> materials;

	std::vector<int> triangleidxforGPU;
	std::vector<GPUKDtreeNode> KDtreeforGPU;
	std::vector<float> wavelen;

    RenderState state;
};
