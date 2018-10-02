#include <iostream>
#include "scene.h"
#include "tiny_obj_loader/loader_example.cc"
#include "tiny_obj_loader/tiny_obj_loader.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

Scene::Scene(string filename) {
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
            }
            else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms_.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    }
    else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        string obj_file_name;
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            }
            else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
            else if (strcmp(tokens[0].c_str(), "obj_file") == 0) {
                obj_file_name = tokens[1];
                cout << "Creating new object from obj file..." << endl;
                newGeom.type = OBJ_BOX;
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
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        // if obj file, construct more triangles
        // must before push_back newGeom
        if (newGeom.type == OBJ_BOX) {
            // after this, the bounding box is already in the geoms vector
            int res = loadTriangles(obj_file_name, newGeom);
            if (res < 0) {
                cout << "ERROR: fail to load obj file: " << obj_file_name << endl;
                return -1;
            }
        }

        if (newGeom.type != OBJ_BOX) {
            geoms_.push_back(newGeom);
        }
        return 1;
    }
}

int Scene::loadTriangles(const string& filename, const Geom& parent) {
    std::cout << "Loading " << filename << std::endl;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    string basepath = "../mesh";
    const char* filename_char_ptr = filename.c_str();
    const char* basepath_char_ptr = basepath.c_str();

    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err,
        filename_char_ptr, basepath_char_ptr, true);

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        printf("Failed to load/parse .obj.\n");
        return -1;
    }

    auto all_vertices = attrib.vertices;
    auto all_normals = attrib.normals;
    std::cout << "ALL VERT " << all_vertices.size() << std::endl;

    std::cout << "ALL NORMAL " << all_normals.size() << std::endl;

    // use size to mark the range of triangle geoms
    int startIdx = triangles_.size(), endIdx;

    // x min, x max, y min, y max, z min, z max
    float bound[6] = { FLT_MAX, FLT_MIN, FLT_MAX, FLT_MIN, FLT_MAX, FLT_MIN };

    for (auto shape : shapes) {
        auto indices = shape.mesh.indices;
        for (int i = 0; i < indices.size() / 3; ++i) {
            // i th triangle

            // copy all parent info
            Triangle triangle;

            int vert1Id = indices[i * 3 + 0].vertex_index;
            int vert2Id = indices[i * 3 + 1].vertex_index;
            int vert3Id = indices[i * 3 + 2].vertex_index;
            int normalId = normalId = indices[i * 3 + 0].normal_index;

            /*normalId = indices[i * 3 + 0].normal_index;
            std::cout << "NORMAL IDX: " << normalId << std::endl;

            normalId = indices[i * 3 + 1].normal_index;
            std::cout << "NORMAL IDX: " << normalId << std::endl;

            normalId = indices[i * 3 + 2].normal_index;
            std::cout << "NORMAL IDX: " << normalId << std::endl;*/

            glm::vec3 v1, v2, v3;

            v1.x = all_vertices[vert1Id * 3 + 0];
            v1.y = all_vertices[vert1Id * 3 + 1];
            v1.z = all_vertices[vert1Id * 3 + 2];
            
            v2.x = all_vertices[vert2Id * 3 + 0];
            v2.y = all_vertices[vert2Id * 3 + 1];
            v2.z = all_vertices[vert2Id * 3 + 2];
            
            v3.x = all_vertices[vert3Id * 3 + 0];
            v3.y = all_vertices[vert3Id * 3 + 1];
            v3.z = all_vertices[vert3Id * 3 + 2];

            /*glm::vec3 display;
            display = triangle.triangleInfo.normal;

            std::cout << "Triangle normal " << i << std::endl;
            std::cout << triangle.triangleInfo.normal.x << " " << triangle.triangleInfo.normal.y << " " << triangle.triangleInfo.normal.z << std::endl;
            std::cout << std::endl;

            display = triangle.triangleInfo.v1;
            std::cout << "Triangle v1 " << i << std::endl;
            std::cout << display.x << " " << display.y << " " << display.z << std::endl;
            std::cout << std::endl;

            display = triangle.triangleInfo.v2;
            std::cout << "Triangle v2 " << i << std::endl;
            std::cout << display.x << " " << display.y << " " << display.z << std::endl;
            std::cout << std::endl;

            display = triangle.triangleInfo.v3;
            std::cout << "Triangle v3 " << i << std::endl;
            std::cout << display.x << " " << display.y << " " << display.z << std::endl;
            std::cout << std::endl;

            std::cout << "=====================================" << std::endl;*/

            glm::mat4 mat = parent.transform;
            glm::mat4 mat2 = parent.inverseTransform;
            glm::mat4 mat3 = parent.invTranspose;

            v1 = glm::vec3(mat * glm::vec4(v1, 1.f));
            v2 = glm::vec3(mat * glm::vec4(v2, 1.f));
            v3 = glm::vec3(mat * glm::vec4(v3, 1.f));

            // update bound
            updateBound(bound, v1);
            updateBound(bound, v2);
            updateBound(bound, v3);

            triangle.normal = glm::normalize(glm::cross((v3 - v1), (v2 - v1)));

            triangle.v1 = v1;
            triangle.v2 = v2;
            triangle.v3 = v3;
            
            triangle.materialid = parent.materialid;

            triangles_.push_back(triangle);
        }
    }
    endIdx = triangles_.size() - 1;

    // PrintInfo(attrib, shapes, materials);

    // Begin to construct bounding box
    printBound(bound);
    Geom boundingBox;
    boundingBox.type = OBJ_BOX;
    boundingBox.translation = glm::vec3((bound[0] + bound[1]) / 2,
                                        (bound[2] + bound[3]) / 2,
                                        (bound[4] + bound[5]) / 2);
    boundingBox.rotation = glm::vec3(0, 0, 0);
    boundingBox.scale = glm::vec3((bound[1] - bound[0]),
                                  (bound[3] - bound[2]),
                                  (bound[5] - bound[4]));
    boundingBox.transform = utilityCore::buildTransformationMatrix(
        boundingBox.translation, boundingBox.rotation, boundingBox.scale);
    boundingBox.inverseTransform = glm::inverse(boundingBox.transform);
    boundingBox.invTranspose = glm::inverseTranspose(boundingBox.transform);
    boundingBox.materialid = 1;
    boundingBox.triangleIdx.start = startIdx;
    boundingBox.triangleIdx.end = endIdx;

    // push to geoms
    geoms_.push_back(boundingBox);

    return 1;
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
        }
        else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "UP") == 0) {
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

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    }
    else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.color = color;
            }
            else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            }
            else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

void Scene::updateBound(float* bound_out, const glm::vec3& vert) {
    bound_out[0] = min(bound_out[0], vert.x);
    bound_out[1] = max(bound_out[1], vert.x);

    bound_out[2] = min(bound_out[2], vert.y);
    bound_out[3] = max(bound_out[3], vert.y);

    bound_out[4] = min(bound_out[4], vert.z);
    bound_out[5] = max(bound_out[5], vert.z);
}


void Scene::printBound(const float* bound) {
    std::cout << "=====Bounding Box=======" << std::endl;
    std::cout << "x Min: " << bound[0] << std::endl;
    std::cout << "x Max: " << bound[1] << std::endl;
    std::cout << "y Min: " << bound[2] << std::endl;
    std::cout << "y Max: " << bound[3] << std::endl;
    std::cout << "z Min: " << bound[4] << std::endl;
    std::cout << "z Max: " << bound[5] << std::endl;
    std::cout << "=====Bounding Box=======" << std::endl;
}
