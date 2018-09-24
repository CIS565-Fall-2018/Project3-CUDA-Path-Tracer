#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

Scene::Scene(string filename) {
  stbi_set_flip_vertically_on_load(true);

    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
}

void Scene::LoadImage(std::string path, ImageInfo& info)
{
  info.startIdx = allTexels.size();
  int width;
  int height;
  int channels;

  uint8_t* imagePixels = (uint8_t *)stbi_load(path.c_str(),
    &width,
    &height,
    &channels,
    STBI_rgb);

  info.width = width;
  info.height = height;

  const unsigned bytePerPixel = channels;

  // Source: https://stackoverflow.com/questions/48235421/get-rgb-of-a-pixel-in-stb-image
  for (int idx = 0; idx < width; ++idx)
  {
    for (int idy = 0; idy < height; ++idy)
    {
      uint8_t* pixelOffset = imagePixels + (idx + height * idy) * bytePerPixel;
      float r = pixelOffset[0] / 255.0f;
      float g = pixelOffset[1] / 255.0f;
      float b = pixelOffset[2] / 255.0f;
      allTexels.emplace_back(r, g, b);
    }
  }
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;
        newGeom.id = id;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
            else if (strcmp(line.c_str(), "squareplane") == 0) {
              cout << "Creating new squareplane..." << endl;
              newGeom.type = SQUAREPLANE;
            }
            else if (strcmp(tokens[0].c_str(), "mesh") == 0) {
              cout << "Creating new mesh..." << endl;
              newGeom.type = MESH;
              newGeom.meshStartIndex = meshTriangles.size();
              loadMesh(tokens[1], newGeom);
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x
							, 2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

void Scene::loadMesh(const string& meshPath, Geom& geom)
{
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string err;
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, meshPath.c_str(), "../meshes/", true);

  if (!err.empty()) {
    std::cerr << err << std::endl;
  }

  if (!ret) {
    printf("Failed to load/parse .obj.\n");
    return;
  }

  if (shapes.size() > 1 || shapes.size() == 0) {
    printf("Only 1 shape supported per OBJ.\n");
    return;
  }

  const auto& meshIndices = shapes[0].mesh.indices;
  geom.numTriangles = shapes[0].mesh.indices.size() / 3;

  meshTriangles.reserve(meshTriangles.size() + (meshIndices.size() / 3));

  for (unsigned int j = 0; j < meshIndices.size(); j += 3)
  {
    const int v1Idx = meshIndices[j].vertex_index;
    const int n1Idx = meshIndices[j].normal_index;
    const int t1Idx = meshIndices[j].texcoord_index;

    const int v2Idx = meshIndices[j + 1].vertex_index;
    const int n2Idx = meshIndices[j + 1].normal_index;
    const int t2Idx = meshIndices[j + 1].texcoord_index;

    const int v3Idx = meshIndices[j + 2].vertex_index;
    const int n3Idx = meshIndices[j + 2].normal_index;
    const int t3Idx = meshIndices[j + 2].texcoord_index;

    Triangle tri;
    tri.p1 = glm::vec3(attrib.vertices[v1Idx], attrib.vertices[v1Idx + 1], attrib.vertices[v1Idx + 2]);
    tri.p2 = glm::vec3(attrib.vertices[v2Idx], attrib.vertices[v2Idx + 1], attrib.vertices[v2Idx + 2]);
    tri.p3 = glm::vec3(attrib.vertices[v3Idx], attrib.vertices[v3Idx + 1], attrib.vertices[v3Idx + 2]);

    tri.n1 = glm::vec3(attrib.normals[n1Idx], attrib.normals[n1Idx + 1], attrib.normals[n1Idx + 2]);
    tri.n2 = glm::vec3(attrib.normals[n2Idx], attrib.normals[n2Idx + 1], attrib.normals[n2Idx + 2]);
    tri.n3 = glm::vec3(attrib.normals[n3Idx], attrib.normals[n3Idx + 1], attrib.normals[n3Idx + 2]);

    tri.uv1 = glm::vec2(attrib.texcoords[t1Idx], attrib.texcoords[t1Idx + 1]);
    tri.uv2 = glm::vec2(attrib.texcoords[t2Idx], attrib.texcoords[t2Idx + 1]);
    tri.uv3 = glm::vec2(attrib.texcoords[t3Idx], attrib.texcoords[t3Idx + 1]);

    meshTriangles.push_back(tri);
  }
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;
        newMaterial.type = DIFFUSE;

        //load static properties
        for (int i = 0; i < 50; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);

            if (line.empty())
            {
              break;
            }

            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "ROUGHNESS") == 0) {
              newMaterial.roughness = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "DIFFUSE") == 0) {
              newMaterial.type = DIFFUSE;
            } else if (strcmp(tokens[0].c_str(), "SPECULAR") == 0) {
              newMaterial.type = SPECULAR;
            } else if (strcmp(tokens[0].c_str(), "ROUGH_SPECULAR") == 0) {
              newMaterial.type = ROUGH_SPECULAR;
            }
            else if (strcmp(tokens[0].c_str(), "ROUGH_DIFFUSE") == 0) {
              newMaterial.type = ROUGH_DIFFUSE;
            }
            else if (strcmp(tokens[0].c_str(), "DIFFUSE_MAP") == 0 && tokens.size() == 4) {
              ImageInfo info;
              info.repeatX = atoi(tokens[2].c_str());
              info.repeatY = atoi(tokens[3].c_str());

              newMaterial.diffuseMapId = imageInfo.size();

              LoadImage(tokens[1], info);
              imageInfo.push_back(info);
            }
            else if (strcmp(tokens[0].c_str(), "BUMP_MAP") == 0 && tokens.size() == 4) {
              ImageInfo info;
              info.repeatX = atoi(tokens[2].c_str());
              info.repeatY = atoi(tokens[3].c_str());

              newMaterial.bumpMapId = imageInfo.size();

              LoadImage(tokens[1], info);
              imageInfo.push_back(info);
            }
            else if (strcmp(tokens[0].c_str(), "NORMAL_MAP") == 0 && tokens.size() == 4) {
              ImageInfo info;
              info.repeatX = atoi(tokens[2].c_str());
              info.repeatY = atoi(tokens[3].c_str());

              newMaterial.normalMapId = imageInfo.size();

              LoadImage(tokens[1], info);
              imageInfo.push_back(info);
            }
            else if (strcmp(tokens[0].c_str(), "EMISSIVE_MAP") == 0 && tokens.size() == 4) {
              ImageInfo info;
              info.repeatX = atoi(tokens[2].c_str());
              info.repeatY = atoi(tokens[3].c_str());

              newMaterial.emissiveMapId = imageInfo.size();

              LoadImage(tokens[1], info);
              imageInfo.push_back(info);
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}
