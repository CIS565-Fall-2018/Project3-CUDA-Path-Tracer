#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tinyobjloader\tiny_obj_loader.h"

using namespace tinyobj;
using namespace std;
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

/**
 * Given an initialized triangle Geom, fill it with appropriate data using the parsed obj file
 */
void createTriangle(Geom& triangle, glm::vec3& boundMin, glm::vec3& boundMax, 
					int triIdx, const shape_t currPoly, const attrib_t tinyObjAttrib) {
	// triangle indices
	index_t i0 = currPoly.mesh.indices[3 * triIdx + 0];
	index_t i1 = currPoly.mesh.indices[3 * triIdx + 1];
	index_t i2 = currPoly.mesh.indices[3 * triIdx + 2];

	// triangle positions
	glm::vec3 pos0, pos1, pos2;
	for (int i = 0; i < 3; i++) {
		pos0[i] = tinyObjAttrib.vertices[3 * i0.vertex_index + i];
		pos1[i] = tinyObjAttrib.vertices[3 * i1.vertex_index + i];
		pos2[i] = tinyObjAttrib.vertices[3 * i2.vertex_index + i];

		boundMin[i] = min(pos2[i], min(pos1[i], min(pos0[i], boundMin[i])));
		boundMax[i] = max(pos2[i], max(pos1[i], max(pos0[i], boundMax[i])));
	}
	triangle.pos[0] = pos0;
	triangle.pos[1] = pos1;
	triangle.pos[2] = pos2;

	// triangle uvs
	glm::vec2 uv0(0.0f);
	glm::vec2 uv1(0.0f);
	glm::vec2 uv2(0.0f);
	if (tinyObjAttrib.texcoords.size()) {
		for (int i = 0; i < 2; i++) {
			uv0[i] = tinyObjAttrib.texcoords[2 * i0.texcoord_index + i];
			uv1[i] = tinyObjAttrib.texcoords[2 * i1.texcoord_index + i];
			uv2[i] = tinyObjAttrib.texcoords[2 * i2.texcoord_index + i];

			uv0[i] = i == 1 ? 1.0f - uv0[i] : uv0[i];
			uv1[i] = i == 1 ? 1.0f - uv1[i] : uv1[i];
			uv2[i] = i == 1 ? 1.0f - uv2[i] : uv2[i];
		}
	}
	triangle.uv[0] = uv0;
	triangle.uv[1] = uv1;
	triangle.uv[2] = uv2;

	// triangle normals
	glm::vec3 norm0, norm1, norm2;
	if (tinyObjAttrib.normals.size()) {
		for (int i = 0; i < 3; i++) {
			norm0[i] = tinyObjAttrib.normals[3 * i0.normal_index + i];
			norm1[i] = tinyObjAttrib.normals[3 * i1.normal_index + i];
			norm2[i] = tinyObjAttrib.normals[3 * i2.normal_index + i];
		}
	}
	else {
		glm::vec3 normal = calculate_geometric_normals(pos0, pos1, pos2);
		norm0 = normal;
		norm1 = normal;
		norm2 = normal;
	}
	triangle.norm[0] = norm0;
	triangle.norm[1] = norm1;
	triangle.norm[2] = norm2;
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
    } 

    cout << "Loading Geom " << id << "..." << endl;
    Geom newGeom;
	vector<Geom> parsedTriangles; // we might create many geoms if its a mesh
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
		else if (strcmp(tokens[0].c_str(), "mesh") == 0) {
			cout << "Loading new mesh..." << endl;
			newGeom.type = MESH;
			attrib_t tinyObjAttrib;
			vector<material_t> mats;
			vector<shape_t> shapes;
			string error;
			if (!LoadObj(&tinyObjAttrib, &shapes, &mats, &error, tokens[1].c_str())) {
				cout << "FAILURE: Loading " << tokens[1].c_str() << " did not succeed!" << endl;
			}

			if (!error.empty()) {
				cout << "FAILURE: Loading OBJ resulted in error - " << error << endl;
			}

			// init bounds of mesh
			glm::vec3 min(FLT_MAX);
			glm::vec3 max(FLT_MIN);

			for (int poly = 0; poly < shapes.size(); poly++) {
				// traverse each triangle in this polygon and create a triangle Geom accordingly
				shape_t currPoly = shapes[poly];
				for (int tri = 0; tri < currPoly.mesh.indices.size() / 3; tri++) {
					Geom triangle;
					triangle.type = TRI;
					createTriangle(triangle, min, max, tri, currPoly, tinyObjAttrib);
					parsedTriangles.push_back(triangle);
				}
			}
			int nbTriangles = parsedTriangles.size();
			newGeom.nbTriangles = nbTriangles;
			newGeom.min = min;
			newGeom.max = max;
		}
    }

    //link material
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
		int matId = atoi(tokens[1].c_str());
		for (Geom& g : parsedTriangles) {
			g.materialid = matId;
		}
		newGeom.materialid = matId;
        cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
    }

    //load transformations
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
			glm::vec3 t(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			for (Geom& g : parsedTriangles) {
				g.translation = t;
			}
			newGeom.translation = t;
        } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
			glm::vec3 r(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			for (Geom& g : parsedTriangles) {
				g.rotation = r;
			}
			newGeom.rotation = r;
        } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
			glm::vec3 s(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			for (Geom& g : parsedTriangles) {
				g.scale = s;
			}
            newGeom.scale = s;
        }

        utilityCore::safeGetline(fp_in, line);
    }

	newGeom.transform = utilityCore::buildTransformationMatrix(
		newGeom.translation, newGeom.rotation, newGeom.scale);
	newGeom.inverseTransform = glm::inverse(newGeom.transform);
	newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

	geoms.push_back(newGeom);
	for (Geom& g : parsedTriangles) {
		g.transform = utilityCore::buildTransformationMatrix(g.translation, g.rotation, g.scale);
		g.inverseTransform = glm::inverse(g.transform);
		g.invTranspose = glm::inverseTranspose(g.transform);
		geoms.push_back(g);
	}
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
		else if (strcmp(tokens[0].c_str(), "FOCAL") == 0) {
			camera.focalDistance = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "LENSR") == 0) {
			camera.lensRadius = atof(tokens[1].c_str());
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

glm::vec3 calculate_geometric_normals(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2) {
	glm::vec3 edge10 = p1 - p0;
	
	glm::vec3 edge20 = p2 - p0;

	return glm::normalize(glm::cross(edge20, edge10));
}
