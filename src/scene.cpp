#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_obj_loader.h"
#include "BVHKDTree.h"
#include <thrust/random.h>

Scene::Scene(const string& filename) {
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
                loadGeom(filename, tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }

	//My code here
	cout << "total tris: " << triangles.size() << endl;
	//debugNodeIntersectionTest(100, scene->triangles.data(), scene->geoms[6].root);

}

int Scene::loadMesh(const string& sceneName, const string& fileName, vector<Triangle>& mesh_triangles)
{
	//get the directory for the obj file
	int slash_forward = sceneName.find_last_of("/");
	int slash_backward = sceneName.find_last_of("\\");
	int slash = sceneName.size();
	if (slash_forward != slash && slash_backward != slash)
	{
		slash = max(slash_forward, slash_backward);
	}
	else if(slash_forward != slash)
	{
		slash = slash_forward;
	}
	else if (slash_backward != slash)
	{
		slash = slash_backward;
	}
	else
	{
		slash = -1;
	}

	string filePath = sceneName.substr(0, slash + 1);
	filePath.append(fileName);

	vector<tinyobj::shape_t> shapes; 
	vector<tinyobj::material_t> materials;
	string errors = tinyobj::LoadObj(shapes, materials, filePath.c_str());

	if (errors.size() == 0)
	{
		//Read the information from the vector of shape_ts
		for (unsigned int i = 0; i < shapes.size(); i++)
		{
			std::vector<float> &positions = shapes[i].mesh.positions;
			std::vector<float> &normals = shapes[i].mesh.normals;
			std::vector<float> &uvs = shapes[i].mesh.texcoords;
			std::vector<unsigned int> &indices = shapes[i].mesh.indices;
			for (unsigned int j = 0; j < indices.size(); j += 3)
			{
				glm::vec3 p1(positions[indices[j] * 3], positions[indices[j] * 3 + 1], positions[indices[j] * 3 + 2]);
				glm::vec3 p2(positions[indices[j + 1] * 3], positions[indices[j + 1] * 3 + 1], positions[indices[j + 1] * 3 + 2]);
				glm::vec3 p3(positions[indices[j + 2] * 3], positions[indices[j + 2] * 3 + 1], positions[indices[j + 2] * 3 + 2]);

				Triangle t(p1, p2, p3);

				t.max = glm::max(p1, glm::max(p2, p3));
				t.min = glm::min(p1, glm::min(p2, p3));
				t.center = (t.max + t.min) / 2.f;
				//assuming anti-clk-wise
				//      p2
				//     /  \
				//    p3---p1 
				t.planeNormal = glm::normalize(glm::cross(p3 - p2, p1 - p2));

				if (normals.size() > 0)
				{
					glm::vec3 n1(normals[indices[j] * 3], normals[indices[j] * 3 + 1], normals[indices[j] * 3 + 2]);
					glm::vec3 n2(normals[indices[j + 1] * 3], normals[indices[j + 1] * 3 + 1], normals[indices[j + 1] * 3 + 2]);
					glm::vec3 n3(normals[indices[j + 2] * 3], normals[indices[j + 2] * 3 + 1], normals[indices[j + 2] * 3 + 2]);
					t.n[0] = n1;
					t.n[1] = n2;
					t.n[2] = n3;
				}
				if (uvs.size() > 0)
				{
					glm::vec2 t1(uvs[indices[j] * 2], uvs[indices[j] * 2 + 1]);
					glm::vec2 t2(uvs[indices[j + 1] * 2], uvs[indices[j + 1] * 2 + 1]);
					glm::vec2 t3(uvs[indices[j + 2] * 2], uvs[indices[j + 2] * 2 + 1]);
					t.t[0] = t1;
					t.t[1] = t2;
					t.t[2] = t3;
				}
				mesh_triangles.push_back(t);
			}
		}
	}
	else
	{
		//An error loading the OBJ occurred!
		cout << filePath << ", error: " << errors << std::endl;
	}

	//assume mesh_triangles is empty before invoking this function
	return mesh_triangles.size();
}

int Scene::loadGeom(const string& sceneName, const string& objectid) {
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
			}
			else if (strcmp(tokens[0].c_str(), "square") == 0) {
				cout << "Creating new square..." << endl;
				newGeom.type = SQUARE;
			}
			else if (strcmp(tokens[0].c_str(), "mesh") == 0) {
				string fileName = tokens[1];
				cout << "Creating new mesh from " << fileName << "..." << endl;
				newGeom.type = MESH;
				vector<Triangle> mesh_triangles;
				if (loadMesh(sceneName, fileName, mesh_triangles) > 0)
				{
					int size = triangles.size();
					//build tree & flaten tree into array
					newGeom.root = BVHKDTree::buildTree(mesh_triangles, 0, size, 0, mesh_triangles.size() - 1);
					newGeom.triangleStartIndex = size;
					newGeom.triangleCount = mesh_triangles.size();
					//copy array to scene triangle vector
					triangles.insert(triangles.end(), mesh_triangles.begin(), mesh_triangles.end());
					//debug
					//printTriangles(triangles);
				}
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

        newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
		
		//My code here
		newGeom.id = id;

        geoms.push_back(newGeom);

		//My code here
		if (materials[newGeom.materialid].type == 1)//if emmissive
		{
			lights.push_back(newGeom);
		}
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

int Scene::loadMaterial(const string& materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {//My code here. Added a lot of stuff so changed to 10
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "RGB") == 0)
			{
				glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.color = color;
			}
			else if (strcmp(tokens[0].c_str(), "SPECRRGB") == 0) 
			{
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specularReflective.color = specColor;
            }
			else if (strcmp(tokens[0].c_str(), "SPECTRGB") == 0) 
			{
				glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.specularTransmissive.color = specColor;
			}
			else if (strcmp(tokens[0].c_str(), "IOR") == 0) 
			{
                newMaterial.specularReflective.indexOfRefraction = atof(tokens[1].c_str());
				newMaterial.specularTransmissive.indexOfRefraction = atof(tokens[1].c_str());
            }
			else if (strcmp(tokens[0].c_str(), "EMITRGB") == 0)
			{
				glm::vec3 emitColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.emissive.color = emitColor;
			}
			else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0)
			{
				newMaterial.emissive.emittance = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "TYPE") == 0) 
			{
                newMaterial.type = atoi(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

void Scene::printTriangles(const vector<Triangle> &triangles)
{
	for (int i = 0; i < triangles.size(); i++)
	{
		cout << i << 
			": left= " << triangles[i].leftIndex << 
			"; right= " << triangles[i].rightIndex << 
			"; min= " << triangles[i].min.x << "," << triangles[i].min.y << "," << triangles[i].min.z <<
			"; max= " << triangles[i].max.x << "," << triangles[i].max.y << "," << triangles[i].max.z << 
			endl;
		for (int j = 0; j < 3; j++)
		{
			cout << "p" << j << ": " << triangles[i].v[j].x << "," << triangles[i].v[j].y << "," << triangles[i].v[j].z << "; " << endl;
		}
		for (int j = 0; j < 3; j++)
		{
			cout << "n" << j << ": " << triangles[i].n[j].x << "," << triangles[i].n[j].y << "," << triangles[i].n[j].z << "; " << endl;
		}
		for (int j = 0; j < 3; j++)
		{
			cout << "t" << j << ": " << triangles[i].t[j].x << "," << triangles[i].t[j].y << "; " << endl;
		}

		cout << "plane normal: " << triangles[i].planeNormal.x << "," << triangles[i].planeNormal.y << "," << triangles[i].planeNormal.z << endl;
	}
}
