#include"KDtreeNode.h"



int KDtreeNode::computeLongestAxis(glm::vec3 maxv, glm::vec3 minv)
{
	int axis;
	glm::vec3 diff = maxv - minv;
	if (diff.x >= diff.y && diff.x >= diff.z) axis = 0;
	else if (diff.y >= diff.x&& diff.y >= diff.z) axis = 1;
	else if (diff.z >= diff.x && diff.z >= diff.y) axis = 2;
	return axis;
}

void KDtreeNode::Build(KDtreeNode* node,vector<Triangle> tris, int depth, int& nodecount)
{
	nodecount++;
	node->triangles = tris;
	node->right = NULL;
	node->left = NULL;
	node->depth = depth;
	node->nodeidx = nodecount;

	if (tris.size() == 0)
	{
		node = NULL;
		return;
	}
		

	if (tris.size() == 1)
	{
		node->BoundingBox.maxB = tris[0].BoundingBox.maxB;
		node->BoundingBox.minB = tris[0].BoundingBox.minB;
		node->right = NULL;
		node->left = NULL;
		return;
	}

	glm::vec3 mid(0);
	glm::vec3 maxbb(-(1e+8));
	glm::vec3 minbb(1e+8);

	for (int i = 0; i < tris.size(); ++i)
	{
		maxbb.x = maxbb.x < tris[i].BoundingBox.maxB.x ? tris[i].BoundingBox.maxB.x : maxbb.x;
		maxbb.y = maxbb.y < tris[i].BoundingBox.maxB.y ? tris[i].BoundingBox.maxB.y : maxbb.y;
		maxbb.z = maxbb.z < tris[i].BoundingBox.maxB.z ? tris[i].BoundingBox.maxB.z : maxbb.z;

		minbb.x = minbb.x > tris[i].BoundingBox.minB.x ? tris[i].BoundingBox.minB.x : minbb.x;
		minbb.y = minbb.y > tris[i].BoundingBox.minB.y ? tris[i].BoundingBox.minB.y : minbb.y;
		minbb.z = minbb.z > tris[i].BoundingBox.minB.z ? tris[i].BoundingBox.minB.z : minbb.z;

		mid += tris[i].computeMidpt()*(1.0f / tris.size());
	}
	node->BoundingBox.maxB = maxbb;
	node->BoundingBox.minB = minbb;
	vector<Triangle> left_tris;
	vector<Triangle> right_tris;
	int axis = node->computeLongestAxis(node->BoundingBox.maxB, node->BoundingBox.minB);
	for (int i = 0; i < tris.size(); ++i)
	{
		switch (axis)
		{
		case 0:
			mid.x >= tris[i].computeMidpt().x ? right_tris.push_back(tris[i]) : left_tris.push_back(tris[i]);
			break;
		case 1:
			mid.y >= tris[i].computeMidpt().y ? right_tris.push_back(tris[i]) : left_tris.push_back(tris[i]);
			break;
		case 2:
			mid.z >= tris[i].computeMidpt().z ? right_tris.push_back(tris[i]) : left_tris.push_back(tris[i]);
			break;
		}
	}
	if (left_tris.size() == 0 && right_tris.size() > 0) left_tris = right_tris;
	if (right_tris.size() == 0 && left_tris.size() > 0) right_tris = left_tris;

	int match = 0;
	for (int i = 0; i < left_tris.size(); ++i)
	{
		for (int j = 0; j < right_tris.size(); ++j)
		{
			if (left_tris[i] == right_tris[j]) match++;
		}
	}
	if ((float)match / left_tris.size() < 0.5 && (float)match / right_tris.size() < 0.5)
	{
		node->right = new KDtreeNode();
		node->left = new KDtreeNode();
		Build(node->left ,left_tris, depth + 1,nodecount);
		Build(node->right ,right_tris, depth + 1,nodecount);
	}
	else
	{
		node->right = NULL;
		node->left = NULL;
	}
}