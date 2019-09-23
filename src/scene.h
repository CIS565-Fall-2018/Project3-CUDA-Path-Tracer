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
    int loadCamera();
	void loadMesh();
public:
    Scene(string filename);
    ~Scene();

	int N_tris;	// number of mesh vertices

	std::vector<Triangle> tris;

	glm::vec3 box_min;
	glm::vec3 box_max;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
