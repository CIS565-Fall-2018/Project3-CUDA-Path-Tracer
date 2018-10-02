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
    int loadTriangles(const string& filename, const Geom& parent);
    void updateBound(float* bound_out, const glm::vec3& vert);
    void printBound(const float* bound);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Geom> global_triangles;
    std::vector<Material> materials;
    RenderState state;
};
