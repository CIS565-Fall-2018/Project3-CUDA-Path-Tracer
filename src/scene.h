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
    int loadMaterial(const string& materialid);
    int loadGeom(const string& sceneName, const string& objectid);
    int loadCamera();
	//My code here
	int loadMesh(const string& sceneName, const string& fileName, vector<Triangle>& mesh_triangles);
	void printTriangles(const vector<Triangle> &triangles);

public:
    Scene(const string& filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

	//My code here
	std::vector<Triangle> triangles;
	std::vector<Geom> lights;
};

