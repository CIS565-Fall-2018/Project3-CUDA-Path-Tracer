#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
	int loadObj(string path, std::vector<Geom> &tris);
	KDTreeNode* buildKDTree(std::vector<Geom> geoms, int currentDepth, int maxDepth);
	int computeKDTreeSize(KDTreeNode *root);
	int flattenKDTree(KDTreeNode *treeNode, std::vector<Geom> &sortedGeoms, std::vector<LinearKDNode> &kdtree, int *offset);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
	std::vector<glm::vec3> textureData;
	std::vector<LinearKDNode> flatKDTree;
	std::vector<Geom> sortedGeoms;

    RenderState state;
	KDTreeNode *root = nullptr;
	glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
		return glm::vec3(m * v);
	}

	Bounds boundsUnion(Bounds &b1, Bounds &b2) {
		glm::vec3 min(std::min(b1.min.x, b2.min.x),
			std::min(b1.min.y, b2.min.y),
			std::min(b1.min.z, b2.min.z));
		glm::vec3 max(std::max(b1.max.x, b2.max.x),
			std::max(b1.max.y, b2.max.y),
			std::max(b1.max.z, b2.max.z));
		Bounds b;
		b.min = min;
		b.max = max;
		return b;
	}

	Bounds boundsUnion(Bounds &b1, glm::vec3 &p) {
		glm::vec3 min(std::min(b1.min.x, p.x),
			std::min(b1.min.y, p.y),
			std::min(b1.min.z, p.z));
		glm::vec3 max(std::max(b1.max.x, p.x),
			std::max(b1.max.y, p.y),
			std::max(b1.max.z, p.z));
		Bounds b;
		b.min = min;
		b.max = max;
		return b;
	}

	int getLongestAxis(Bounds &b) {
		glm::vec3 d = b.max - b.min;
		if (d.x > d.y && d.x > d.z)
			return 0;
		else if (d.y > d.z)
			return 1;
		else
			return 2;
	}

	glm::vec3 getMedian(Bounds &b) {
		return (b.max - b.min) / 2.f;
	}

	Bounds applyTransformation(glm::mat4 &tr, Bounds &b) {
		glm::vec3 min = b.min;
		glm::vec3 max = b.max;
		Bounds ret;
		glm::vec3 p = multiplyMV(tr, glm::vec4(min, 1.0f));
		ret.min = p;
		ret.max = p;

		ret = boundsUnion(ret, multiplyMV(tr, glm::vec4(max.x, min.y, min.z, 1.0f)));
		ret = boundsUnion(ret, multiplyMV(tr, glm::vec4(min.x, max.y, min.z, 1.0f)));
		ret = boundsUnion(ret, multiplyMV(tr, glm::vec4(min.x, min.y, max.z, 1.0f)));
		ret = boundsUnion(ret, multiplyMV(tr, glm::vec4(min.x, max.y, max.z, 1.0f)));
		ret = boundsUnion(ret, multiplyMV(tr, glm::vec4(max.x, max.y, min.z, 1.0f)));
		ret = boundsUnion(ret, multiplyMV(tr, glm::vec4(max.x, min.y, max.z, 1.0f)));
		ret = boundsUnion(ret, multiplyMV(tr, glm::vec4(max.x, max.y, max.z, 1.0f)));
		return ret;
	}

	Bounds getGeoBounds(Geom &geom) {
		if (geom.type == CUBE)
		{
			glm::vec3 min = glm::vec3(-0.5f, -0.5f, -0.5f);
			glm::vec3 max = glm::vec3(0.5f, 0.5f, 0.5f);
			Bounds objectBound;
			objectBound.min = min;
			objectBound.max = max;
			return applyTransformation(geom.transform, objectBound);
		}
		else if (geom.type == SPHERE)
		{
			glm::vec3 min(-1.0f, -1.0f, -1.0f);
			glm::vec3 max(1.0f, 1.0f, 1.0f);
			Bounds objectBound;
			objectBound.min = min;
			objectBound.max = max;
			return applyTransformation(geom.transform, objectBound);
		}
		else if (geom.type == DIAMOND)
		{

		}
		else if (geom.type == MANDELBULB)
		{

		}
		else if (geom.type == TRIANGLE)
		{
			glm::vec3 min(FLT_MAX);
			glm::vec3 max(-FLT_MAX);
			glm::vec3 *points = geom.t.pts;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					if (points[i][j] < min[j]) {
						min[j] = points[i][j];
					}
				}
			}

			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					if (points[i][j] > max[j]) {
						max[j] = points[i][j];
					}
				}
			}
			Bounds ret;
			glm::vec3 epsilon(.001);
			ret.max = max + epsilon;
			ret.min = min - epsilon;
			return ret;
		}
	}

};
