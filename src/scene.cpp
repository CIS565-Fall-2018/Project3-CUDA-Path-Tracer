#include <iostream>
#include "scene.h"
#include "tiny_obj/tiny_obj_loader.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>



#define USE_OBJ_LOADER 0
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


int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(tokens[0].c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            } else if (strcmp(tokens[0].c_str(), "mesh") == 0) {
				cout << "Createing new mesh from obj files.." << endl;
				string obj_filepath = tokens[1].c_str();
				cout << "Get file path! obj file path is: " << obj_filepath << endl;
				if (loadOBJ(obj_filepath))
				{
					newGeom.num_tri = triangles.size();
					cout << "load obj files success! " << endl;
				}
				else
				{
					cout << "load obj files fail! " << endl;

				}
				
				newGeom.type = MESH;
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

    //load static properties + focaldistance + lensradius
    for (int i = 0; i < 7; i++) {
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
		} else if (strcmp(tokens[0].c_str(), "LENSRADIUS") == 0) {
			camera.lensRadius = atof(tokens[1].c_str());
		} else if (strcmp(tokens[0].c_str(), "FOCALDISTANCE") == 0)
		{
			camera.focalDistance = atof(tokens[1].c_str());
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

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
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
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}



bool Scene::loadOBJ(const string& objPath)
{
	cout << objPath << endl;
	cout << objPath << endl;
	cout << objPath << endl;
	cout << objPath << endl;

	
	std::vector<tinyobj::shape_t> shapes;
	std::vector < tinyobj::material_t > mats;
	string error;
	tinyobj::attrib_t attr;
	bool result = tinyobj::LoadObj(&attr, &shapes, &mats, &error, objPath.c_str());
	if (!result) return false;
	if (!error.empty()) cout << "Error loading obj ::" << error << endl;
	cout << "loading obj success!" << endl;
	int num_tri = 0;
	for (int i = 0; i < shapes.size(); ++i)
	{
		cout << "shapes size" << endl;
		for (int j = 0; j < shapes[i].mesh.indices.size() / 3; ++j)
		{
			//cout << "shapes meshs size :   " <<  j  << endl;

			Triangle tri;
			int idxi0 = shapes[i].mesh.indices[3 * j + 0].vertex_index;
			int idxi1 = shapes[i].mesh.indices[3 * j + 1].vertex_index;
			int idxi2 = shapes[i].mesh.indices[3 * j + 2].vertex_index;
			int idxn = shapes[i].mesh.indices[3 * j + 0].normal_index;

			tri.v0 = glm::vec3(attr.vertices[3 * idxi0], attr.vertices[3 * idxi0 + 1], attr.vertices[3 * idxi0 + 2]);
			tri.v1 = glm::vec3(attr.vertices[3 * idxi1], attr.vertices[3 * idxi1 + 1], attr.vertices[3 * idxi1 + 2]);
			tri.v2 = glm::vec3(attr.vertices[3 * idxi2], attr.vertices[3 * idxi2 + 1], attr.vertices[3 * idxi2 + 2]);
			//tri.n = glm::vec3(attr.normals[3 * idxn], attr.normals[3 * idxn + 1], attr.normals[3 * idxn + 2]);
			float temp = 0.08f;
			tri.v0 *= temp;
			tri.v1 *= temp;
			tri.v2 *= temp;


			tri.n = glm::vec3(0.0f);
			//glm::vec3 outV = tri.v0;
			//cout << "triangle ID is(from 0) : " << num_tri << endl;
			//cout << "tri v0 : " << outV.x << " " << outV.y << " " << outV.z << endl;
			//outV = tri.v1;
			//cout << "tri v1 : " << outV.x << " " << outV.y << " " << outV.z << endl;
			//outV = tri.v2;
			//cout << "tri v2 : " << outV.x << " " << outV.y << " " << outV.z << endl;
			//outV = tri.n;
			//cout << "tri n  : " << outV.x << " " << outV.y << " " << outV.z << endl;
			//cout << endl;
			//cout << endl;
			//cout << endl;




			triangles.push_back(tri);
			num_tri++;
		}
	}

	
	return true;
}