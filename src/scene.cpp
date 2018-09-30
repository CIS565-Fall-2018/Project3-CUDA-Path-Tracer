#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_obj_loader.h"
#include <stb_image.h>
#include <stb_image_write.h>
#include "defines.h"

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

	#if KDTREE
	KDHelperNode *root = buildKDTreeCPU(geoms, 0, 200);
	int numNodes = countNodes(root);
	for (int i = 0; i < numNodes; ++i) {
		kdtree.push_back(KDNode());
	}
	int offset = 0;
	flattenKDTree(root, sortedGeoms, kdtree, &offset);
	#endif

	cout << endl << "Scene statistics: " << endl;
	cout << geoms.size() << " primitives" << endl;
	#if KDTREE
	cout << numNodes << " kd nodes" << endl;
	#endif
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    /*if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else */{
        cout << "Loading Geom " << id << "..." << endl;
		bool isMesh = false;
		string meshFile;
        Geom newGeom;
        string line;

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
			} else if (strcmp(tokens[0].c_str(), "mesh") == 0) {
				cout << "Creating new mesh..." << endl;
				// tokens[1] is filename of obj
				// read obj, and add all triangles to geoms
				meshFile = tokens[1];
				isMesh = true;
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

		if (!isMesh) {
			newGeom.transform = utilityCore::buildTransformationMatrix(
				newGeom.translation, newGeom.rotation, newGeom.scale);
			newGeom.inverseTransform = glm::inverse(newGeom.transform);
			newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

			geoms.push_back(newGeom);
		} else {
			// read meshFile and get triangles
			// create new Geometry for each triangle
			// set transform, inverse transform, and transpose for each triangle
			// add triangle to geoms
			std::vector<Material> mtls;
			std::vector<Geom> tris;
			loadMesh(meshFile, tris);
			for (auto &tri : tris) {
                tri.translation =  newGeom.translation;
                tri.rotation = newGeom.rotation;
                tri.scale = newGeom.scale;
				tri.transform = utilityCore::buildTransformationMatrix(
					tri.translation, tri.rotation, tri.scale);
				tri.inverseTransform = glm::inverse(tri.transform);
				tri.invTranspose = glm::inverseTranspose(tri.transform);
				tri.pos[0] = glm::vec3(tri.transform * glm::vec4(tri.pos[0], 1));
				tri.pos[1] = glm::vec3(tri.transform * glm::vec4(tri.pos[1], 1));
				tri.pos[2] = glm::vec3(tri.transform * glm::vec4(tri.pos[2], 1));
				tri.materialid = newGeom.materialid;
				geoms.push_back(tri);
			}
		}

        return 1;
    }
}

int Scene::loadMesh(string filename, std::vector<Geom> &tris) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;
	std::string basedir = "";
	if (filename.find_last_of("/\\") != std::string::npos) {
		basedir = filename.substr(0, filename.find_last_of("/\\"));
	}

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str(), basedir.c_str());
	if (!err.empty()) {
		std::cerr << err << std::endl;
	}
	if (!ret) {
		std::cerr << "Skipping " << filename << std::endl;
		return -1;
	}

	for (size_t s = 0; s < shapes.size(); s++) {
  // Loop over faces(polygon)
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			int fv = 3; // all objs should have only triangles

			Geom tri;
			tri.type = TRIANGLE;

			// Loop over vertices in the face.
			for (size_t v = 0; v < fv; v++) {
			  // access to vertex
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
				tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
				tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
				tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
				tinyobj::real_t nx = 0;
				tinyobj::real_t ny = 0;
				tinyobj::real_t nz = 0;
				if (attrib.normals.size() > 0) {
					nx = attrib.normals[3 * idx.normal_index + 0];
					ny = attrib.normals[3 * idx.normal_index + 1];
					nz = attrib.normals[3 * idx.normal_index + 2];
				}
				tinyobj::real_t tx = 0;
				tinyobj::real_t ty = 0;
				if (attrib.texcoords.size() > 0) {
					tx = attrib.texcoords[2 * idx.texcoord_index + 0];
					ty = attrib.texcoords[2 * idx.texcoord_index + 1];
				}

				tri.pos[v] = glm::vec3(vx, vy, vz);
				tri.nor[v] = glm::vec3(nx, ny, nz);
				tri.uv[v] = glm::vec2(tx, ty);
			}
			index_offset += fv;

			// per-face material
			tris.push_back(tri);
		}
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
		newMaterial.textureOffset = -1;
		newMaterial.normalOffset = -1;
        //load static properties
        for (int i = 0; i < 10; i++) {
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
			} else if (strcmp(tokens[0].c_str(), "DISP") == 0) {
				newMaterial.dispersion = atof(tokens[1].c_str());
			} else if (strcmp(tokens[0].c_str(), "TEX") == 0 && tokens.size() == 2 && strcmp(tokens[1].c_str(), "NONE") != 0) {
				// tokens[1] is the filename of the texture
				int channels = 0;
				float *rawTexData = stbi_loadf(tokens[1].c_str(), &newMaterial.texWidth, &newMaterial.texHeight, &channels, 3);
				newMaterial.textureOffset = textureData.size();
				if (channels == 3 || channels == 4) {
					for (int i = 0; i < newMaterial.texWidth * newMaterial.texHeight; ++i) {
						glm::vec3 color;
						color.r = rawTexData[i * channels];
						color.g = rawTexData[i * channels + 1];
						color.b = rawTexData[i * channels + 2];
						textureData.push_back(color);
					}
					std::cout << "Successfully loaded texture " << tokens[1] << std::endl;
				} else {
					std::cerr << "Error loading texture " << tokens[1] << std::endl;
				}
				stbi_image_free(rawTexData);
			} else if (strcmp(tokens[0].c_str(), "NOR") == 0 && tokens.size() == 2 && strcmp(tokens[1].c_str(), "NONE") != 0) {
				int channels = 0;
				float *rawTexData = stbi_loadf(tokens[1].c_str(), &newMaterial.norWidth, &newMaterial.norHeight, &channels, 3);
				newMaterial.normalOffset = textureData.size();
				if (channels == 3 || channels == 4) {
					for (int i = 0; i < newMaterial.norWidth * newMaterial.norHeight; ++i) {
						glm::vec3 color;
						color.r = rawTexData[i * channels];
						color.g = rawTexData[i * channels + 1];
						color.b = rawTexData[i * channels + 2];
						textureData.push_back(color);
					}
					std::cout << "Successfully loaded normal map " << tokens[1] << std::endl;
				} else {
					std::cerr << "Error loading normal map " << tokens[1] << std::endl;
				}
				stbi_image_free(rawTexData);
			}
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

KDHelperNode *Scene::buildKDTreeCPU(std::vector<Geom> &geoms, int depth, int maxDepth) {
	KDHelperNode *node = new KDHelperNode();
	node->left = nullptr;
	node->right = nullptr;
	node->bounds = GetBounds(geoms);

	if (geoms.size() <= 4 || depth > maxDepth) {
		node->geoms = geoms;
		return node;
	}

	int axis = longestAxis(node->bounds);
	if (glm::abs(node->bounds.max[axis] - node->bounds.min[axis]) < 0.001) {
		node->geoms = geoms;
		return node;
	}
	std::vector<Geom> rightGeos;
	std::vector<Geom> leftGeos;

#if SPLIT_SAH
	// Cost in cost() and split()
	float splitPoint = split(geoms, axis, node->bounds.min[axis], node->bounds.max[axis]);
	for (const auto &g : geoms) {
		Bounds b = GetBounds(g);
		float p = b.min[axis] + ((b.max[axis] - b.min[axis]) / 2.f);
		if (p > splitPoint) {
			rightGeos.push_back(g);
			float min = b.min[axis];
		} else {
			leftGeos.push_back(g);
			float max = b.max[axis];
		}
	}
#else
	std::sort(geoms.begin(), geoms.end(), [&](const Geom &g1, const Geom &g2) {
		return getMidpoint(GetBounds(g1))[axis] < getMidpoint(GetBounds(g2))[axis];
	});
	leftGeos = std::vector<Geom>(geoms.begin(), geoms.begin() + (geoms.size() / 2));
	rightGeos = std::vector<Geom>(geoms.begin() + (geoms.size() / 2), geoms.end());
#endif
	node->left = buildKDTreeCPU(leftGeos, depth + 1, maxDepth);
	node->right = buildKDTreeCPU(rightGeos, depth + 1, maxDepth);
	node->axis = axis;
	return node;
}

int Scene::countNodes(KDHelperNode *node) {
	if (node->left == nullptr && node->right == nullptr) {
		return 1;
	}
	int size = 0;
	if (node->left != nullptr) {
		size += countNodes(node->left);
	}
	if (node->right != nullptr) {
		size += countNodes(node->right);
	}
	return size + 1;
}

int Scene::flattenKDTree(KDHelperNode *helperNode, std::vector<Geom> &sortedGeoms, std::vector<KDNode> &kdtree, int *offset) {
	KDNode *node = &kdtree[*offset];
	node->bounds = helperNode->bounds;
	int myOffset = (*offset)++;
	if (helperNode->left == nullptr || helperNode->right == nullptr) {
		//leaf
		node->primOffset = sortedGeoms.size();
		for (Geom g : helperNode->geoms) {
			sortedGeoms.push_back(g);
		}
		node->numPrims = helperNode->geoms.size();
	} else {
		// interior node
		node->axis = helperNode->axis;
		node->numPrims = 0;
		flattenKDTree(helperNode->left, sortedGeoms, kdtree, offset);
		node->secondChildOffset = flattenKDTree(helperNode->right, sortedGeoms, kdtree, offset);
	}
	return myOffset;
}

float Scene::cost(float split, const std::vector<Geom> &geoms, int axis, float min, float max) {
	int leftGeos = 0;
	int rightGeos = 0;

	// determine where each geometry is relative to the split
	for (int i = 0; i < geoms.size(); ++i) {
		Geom g = geoms[i];
		Bounds b = GetBounds(g);
		// world space geometry midpoint
		float p = b.min[axis] + ((b.max[axis] - b.min[axis]) / 2.f);
		if (p > split) {
			rightGeos++;
			float min = b.min[axis];
			if (min <= split) {
				leftGeos++;
			}
		} else {
			leftGeos++;
			float max = b.max[axis];
			if (max >= split) {
				rightGeos++;
			}
		}
	}
	float leftSize = split - min;
	float rightSize = max - split;
	return (leftSize * leftGeos) + (rightSize * rightGeos);
}

float Scene::split(const std::vector<Geom> &geoms, int axis, float min, float max) {
	float center = (min + max) / 2.f;
	float geoMedian = 0;
	for (const auto &g : geoms) {
		Bounds b = GetBounds(g);
		geoMedian += b.min[axis] + ((b.max[axis] - b.min[axis]) / 2.f);
	}

	geoMedian /= geoms.size();

	float step = (center - geoMedian) / N_BUCKETS;
	float minCost = FLT_MAX;
	float bestSplit = geoMedian;
	if (glm::abs(step) > EPSILON) {
		for (float i = geoMedian; i < center; i += step) {
			float c = cost(i, geoms, axis, min, max);
			if (minCost > c) {
				minCost = c;
				bestSplit = i;
			}
		}
	}
	return bestSplit;
}