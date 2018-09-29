#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"



Scene::Scene(string filename) {
	meshcount = 0;
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
	buildKDtree();
}


bool Scene::buildKDtree()
{
	cout << "Start building KD tree .... .... " << endl;
	rootnode->Build(rootnode,triangles, 0,nodecount);
	cout << "KD tree build finised" << endl;
	int cc = 0;
	BuildTreeGPU(rootnode, 0);
	return 1;
}



void Scene::BuildTreeGPU(KDtreeNode* nn,int cc)
{
	
	if (nn!=NULL/*&&nn->triangles.size()!=0*/)
	{
		
		GPUKDtreeNode tmpnode;
		
		tmpnode.GPUtriangleidxinLst = triangleidxforGPU.size();
		tmpnode.trsize = nn->triangles.size();
		tmpnode.curidx = nn->nodeidx;
		for (int i = 0; i < nn->triangles.size(); ++i)
		{
			triangleidxforGPU.push_back(nn->triangles[i].triidx);
		}
		if (nn->left != NULL)
		tmpnode.leftidx = nn->left->nodeidx;
		if (nn->right != NULL)
		tmpnode.rightidx = nn->right->nodeidx;
		tmpnode.depth = nn->depth;
		tmpnode.maxB = nn->BoundingBox.maxB;
		tmpnode.minB = nn->BoundingBox.minB;
		if (nn->left == NULL) tmpnode.leftidx = -1;
		if (nn->right == NULL) tmpnode.rightidx = -1;
		KDtreeforGPU.push_back(tmpnode);
		if (nn->left != NULL)
		{
			BuildTreeGPU(nn->left, tmpnode.curidx + 1);
		}
		if (nn->right != NULL)
		{
			BuildTreeGPU(nn->right, tmpnode.curidx + 2);
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
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
			else if (strcmp(line.c_str(), "mesh") == 0)
			{
		//load obj file

				mesh newmesh;
				cout << "Building mesh..." << endl;
				newGeom.type = MESH;
				newGeom.meshid = meshcount;
				newmesh.TriStartIndex = triangles.size();
				string objfiledir;
				utilityCore::safeGetline(fp_in, line);
				objfiledir = line;
				cout << "Loading obj from" << objfiledir << endl;
				std::vector<tinyobj::shape_t> shapes;
				std::vector<tinyobj::material_t> mats;
				string error;
				tinyobj::attrib_t attr;
				bool success = tinyobj::LoadObj(&attr, &shapes, &mats, &error, objfiledir.c_str());
				if (!success) return -1;
				if (!error.empty()) cout << "Error loading obj" << error << endl;
				cout << "load obj success" << endl;
				int tricount = 0;
				float maxx = 1e+8; float maxy = 1e+8; float maxz = 1e+8;
				float minx = -(1e+8); float miny = -(1e+8); float minz = -(1e+8);
				for (int i = 0; i < shapes.size(); ++i)
				{
					for (int j = 0; j < shapes[i].mesh.indices.size() / 3; ++j)
					{
						glm::vec3 maxB(-(1e+8)),minB(1e+8);
						Triangle Trii;
						glm::vec3 avgn(0);
						for (int k = 0; k < 3; ++k)
						{
							int idxi = shapes[i].mesh.indices[3 * j + k].vertex_index;
							int idxn = shapes[i].mesh.indices[3 * j + k].normal_index;
							Trii.Triverts[k].pos = glm::vec3(attr.vertices[3 * idxi], attr.vertices[3 * idxi + 1], attr.vertices[3 * idxi + 2]);
							glm::vec3 curpos = Trii.Triverts[k].pos;

							maxB.x = maxB.x < curpos.x ? curpos.x : maxB.x;
							maxB.y = maxB.y < curpos.y ? curpos.y : maxB.y;
							maxB.z = maxB.z < curpos.z ? curpos.z : maxB.z;
							
							minB.x = minB.x > curpos.x ? curpos.x : minB.x;
							minB.y = minB.y > curpos.y ? curpos.y : minB.y;
							minB.z = minB.z > curpos.z ? curpos.z : minB.z;

							maxx = maxx < curpos.x ? curpos.x : maxx;
							maxy = maxy < curpos.y ? curpos.y : maxy;
							maxz = maxz < curpos.z ? curpos.z : maxz;

							minx = minx > curpos.x ? curpos.x : minx;
							miny = miny > curpos.y ? curpos.y : miny;
							minz = minz > curpos.z ? curpos.z : minz;

							Trii.Triverts[k].normal = glm::vec3(attr.normals[3 * idxn], attr.normals[3 * idxn + 1], attr.normals[3 * idxn + 2]);
							avgn += Trii.Triverts[k].normal;
						}
						Trii.BoundingBox.maxB = maxB;
						Trii.BoundingBox.minB = minB;
						Trii.Trinormal = avgn / 3.0f;
						Trii.triidx = globaltricount;
						tricount++;
						triangles.push_back(Trii);
						globaltricount++;
					}
				}
				newmesh.maxbound = glm::vec3(maxx, maxy, maxz);
				newmesh.minbound = glm::vec3(minx, miny, minz);
				newmesh.TriSize = tricount;
				meshs.push_back(newmesh);
				meshcount++;
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

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 8; i++) {
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
			else if (strcmp(tokens[0].c_str(), "DIFFUSE") == 0) {
				newMaterial.diffuse = atoi(tokens[1].c_str());
			}
        }
        materials.push_back(newMaterial);
        return 1;
    }
}
