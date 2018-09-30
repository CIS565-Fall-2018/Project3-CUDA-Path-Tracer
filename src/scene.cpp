#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <stb_image_write.h>
#include <stb_image.h>
#include "tiny_obj_loader.h"

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
	this->root = buildKDTree(this->geoms, 0, 20);
	cout << "Built KDTree" << endl;
	int n = computeKDTreeSize(this->root);
	cout << "Number of nodes: " << n << endl;
	for (int i = 0; i < n; i++) {
		flatKDTree.push_back(LinearKDNode());
	}
	int offset = 0;
	flattenKDTree(root, sortedGeoms, flatKDTree, &offset);
	cout << "Number of primitives: " << sortedGeoms.size() << endl;
}

int Scene::loadObj(string filename, std::vector<Geom> &tris) {
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

			Geom g;
			g.type = TRIANGLE;

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

				g.t.pts[v] = glm::vec3(vx, vy, vz);
				g.t.normals[v] = glm::vec3(nx, ny, nz);
				g.t.uvs[v] = glm::vec2(tx, ty);
			}
			index_offset += fv;

			// per-face material
			tris.push_back(g);
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
		std::vector<Geom> tris;
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
            } else if (strcmp(line.c_str(), "diamond") == 0) {
				cout << "Creating new diamond..." << endl;
				newGeom.type = DIAMOND;
			} else if (strcmp(line.c_str(), "mandelbulb") == 0) {
				cout << "Creating new mandelbulb..." << endl;
				newGeom.type =  MANDELBULB;
			}
			else if (strcmp(tokens[0].c_str(), "mesh") == 0) {
				cout << "Loading in many triangles..." << endl;
				cout << tokens[1]<< endl;
				loadObj(tokens[1], tris);
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

		if (tris.size() > 0) {
			for (Geom g : tris) {
				g.materialid = newGeom.materialid;
				g.transform = utilityCore::buildTransformationMatrix(
					newGeom.translation, newGeom.rotation, newGeom.scale);
				g.inverseTransform = glm::inverse(newGeom.transform);
				g.invTranspose = glm::inverseTranspose(newGeom.transform);
				for (int i = 0; i < 3; i++) {
					g.t.pts[i] = glm::vec3(g.transform * glm::vec4(g.t.pts[i], 1.f));
					g.t.normals[i] = glm::vec3(g.invTranspose * glm::vec4(g.t.normals[i], 0.f));
				}
				geoms.push_back(g);
			}
		}
		else {
			newGeom.transform = utilityCore::buildTransformationMatrix(
				newGeom.translation, newGeom.rotation, newGeom.scale);
			newGeom.inverseTransform = glm::inverse(newGeom.transform);
			newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

			geoms.push_back(newGeom);
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

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 9; i++) {
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
			else if (strcmp(tokens[0].c_str(), "TEXT") == 0) {
				if (strcmp(tokens[1].c_str(), "NONE") == 0) continue;
				
				std::string path(tokens[1]);
				int width, height, channels;
				float *rawPixels = stbi_loadf(path.c_str(), &width, &height, &channels, 3);
				if (channels == 3 || channels == 4)
				{
					newMaterial.textureOffset = textureData.size();
					newMaterial.tex_height = height;
					newMaterial.tex_width = width;
					for (int i = 0; i < width * height; i++)
					{
						glm::vec3 color;
						color.x = rawPixels[i * channels];
						color.y = rawPixels[i * channels + 1];
						color.z = rawPixels[i * channels + 2];

						textureData.push_back(color);
					}
					std::cout << "Loaded texture \"" << path << "\" [" << width << "x" << height << "|" << channels << "]" << std::endl;
				}
				else
				{
					newMaterial.textureOffset = -2;
					std::cerr << "Error loading texture " << path << std::endl;
				}
				stbi_image_free(rawPixels);
			}
			else if (strcmp(tokens[0].c_str(), "NORM") == 0) {
				if (strcmp(tokens[1].c_str(), "NONE") == 0) continue;

				std::string path(tokens[1]);
				int width, height, channels;
				float *rawPixels = stbi_loadf(path.c_str(), &width, &height, &channels, 3);
				if (channels == 3 || channels == 4)
				{
					newMaterial.normMapOffset = textureData.size();
					newMaterial.n_m_height = height;
					newMaterial.n_m_width = width;
					for (int i = 0; i < width * height; i++)
					{
						glm::vec3 color;
						color.x = rawPixels[i * channels];
						color.y = rawPixels[i * channels + 1];
						color.z = rawPixels[i * channels + 2];

						textureData.push_back(color);
					}
					std::cout << "Loaded texture \"" << path << "\" [" << width << "x" << height << "|" << channels << "]" << std::endl;
				}
				else
				{
					newMaterial.normMapOffset = -2;
					std::cerr << "Error loading texture " << path << std::endl;
				}
				stbi_image_free(rawPixels);
			}
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

KDTreeNode* Scene::buildKDTree(std::vector<Geom> geoms, int currentDepth, int maxDepth) {
	KDTreeNode *node = new KDTreeNode();
	node->geoms = geoms;
	if (geoms.size() <= 4 || currentDepth > maxDepth) return node;
	
	// compute bounds
	Bounds b;
	b.min = glm::vec3(999999.f);
	b.max = glm::vec3(-999999.f);
	for (int i = 0; i < geoms.size(); i++) {
		b = boundsUnion(b, getGeoBounds(geoms[i]));
	}
	node->bounds = b;

	// find longest axis
	int axis = getLongestAxis(b);

	std::sort(geoms.begin(), geoms.end(), [&](Geom &g1, Geom &g2) {
		return getMedian(getGeoBounds(g1))[axis] < getMedian(getGeoBounds(g2))[axis];
	});

	glm::vec3 median = getMedian(b);
	// copy to left and right vectors
	std::vector<Geom> leftHalf = std::vector<Geom>(geoms.begin(), geoms.begin() + (geoms.size() / 2));
	std::vector<Geom> rightHalf = std::vector<Geom>(geoms.begin() + (geoms.size() / 2), geoms.end());
	/*for (int i = 0; i < geoms.size() / 2; i++) {
		leftHalf.push_back(geoms[i]);
	}

	for (int i = geoms.size() / 2; i < geoms.size(); i++) {
		rightHalf.push_back(geoms[i]);
	}
	*/
	
	node->axis = axis;
	node->left = buildKDTree(leftHalf, currentDepth + 1, maxDepth);
	node->right = buildKDTree(rightHalf, currentDepth + 1, maxDepth);
	return node;
}

int Scene::computeKDTreeSize(KDTreeNode *node) {
	if (!node->left && !node->right) return 1;
	int count = 0;
	if (node->left) count += computeKDTreeSize(node->left);
	if (node->right) count += computeKDTreeSize(node->right);
	return count + 1;
}

int Scene::flattenKDTree(KDTreeNode *treeNode, std::vector<Geom> &sortedGeoms, std::vector<LinearKDNode> &kdtree, int *offset) {
	LinearKDNode *node = &kdtree[*offset];
	node->bounds = treeNode->bounds;
	int myOffset = (*offset)++;
	if (treeNode->left == nullptr || treeNode->right == nullptr) {
		//leaf
		node->primitivesOffset = sortedGeoms.size();
		for (Geom g : treeNode->geoms) {
			sortedGeoms.push_back(g);
		}
		node->nPrimitives = treeNode->geoms.size();
	}
	else {
		// interior node
		node->axis = treeNode->axis;
		node->nPrimitives = 0;
		flattenKDTree(treeNode->left, sortedGeoms, kdtree, offset);
		node->secondChildOffset = flattenKDTree(treeNode->right, sortedGeoms, kdtree, offset);
	}
	return myOffset;
}