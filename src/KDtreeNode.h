#ifndef KDTREE_NODE_H
#define KDTREE_NODE_H

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "../src/sceneStructs.h"

#include<iostream>
#include<vector>
using namespace std;

class KDtreeNode {
public:
	struct 
	{
		glm::vec3 maxB;
		glm::vec3 minB;
	}BoundingBox;
	int depth;
	KDtreeNode* left; 
	KDtreeNode* right;
	KDtreeNode* parent;
	int nodeidx;
	vector<Triangle> triangles;
	KDtreeNode() {BoundingBox.maxB = glm::vec3(0); BoundingBox.minB = glm::vec3(0); };
	KDtreeNode(glm::vec3& Meshmaxbound, glm::vec3 Meshminbound) { BoundingBox.maxB = Meshmaxbound;
	BoundingBox.minB = Meshminbound;
	};
	void Build(KDtreeNode* node, vector<Triangle> tris, int depth, int& nodecount, KDtreeNode* parent);
	int computeLongestAxis(glm::vec3 maxv,glm::vec3 minv);
};




#endif // !OCT_TREE_H
