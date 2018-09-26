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

struct Triangle
{
  glm::vec3 planeNormal;

  glm::vec3 p1;
  glm::vec3 p2;
  glm::vec3 p3;

  glm::vec3 n1;
  glm::vec3 n2;
  glm::vec3 n3;

  glm::vec2 uv1;
  glm::vec2 uv2;
  glm::vec2 uv3;
};

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    void loadMesh(const string& meshPath, Geom& geom);
public:
    Scene(string filename);
    ~Scene();

    void LoadImage(std::string path, ImageInfo& info);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<ImageInfo> imageInfo;
    RenderState state;

    std::vector<glm::vec3> allTexels;

    std::vector<Triangle> meshTriangles;

    int m_numLights;
};
