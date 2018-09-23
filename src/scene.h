#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

struct ImageInfo
{
  int width{ 0 };
  int height{ 0 };
  int repeatX{ 1 };
  int repeatY{ 1 };
  int startIdx{ 0 };
};

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    void LoadImage(std::string path, ImageInfo& info);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<ImageInfo> imageInfo;
    RenderState state;

    std::vector<glm::vec3> allTexels;

    int m_numLights;
};
