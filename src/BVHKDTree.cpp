#include "BVHKDTree.h"

BVHKDTree::BVHKDTree()
{
}

BVHKDTree::~BVHKDTree()
{
}

bool xSort(const Triangle& a, const Triangle& b) { return a.center.x < b.center.x; }
bool ySort(const Triangle& a, const Triangle& b) { return a.center.y < b.center.y; }
bool zSort(const Triangle& a, const Triangle& b) { return a.center.z < b.center.z; }

//need to add an offset so that indices are for the one large triangle array for all the meshes in the scene instead of only one mesh
int BVHKDTree::buildTree(vector<Triangle> &triangles, const int axis, const int indexOffset, const int start, const int end, glm::vec3 *min, glm::vec3 *max)//[start, end]
{
	if (start>end) return -1;
	//else if (start == end) return start + indexOffset; //because in this case we still want to set min and max

	int root = 0;

	switch (axis)
	{
	case 0:
		sort(triangles.begin() + start, triangles.begin() + end + 1, xSort);//[start, end + 1)
		break;
	case 1:
		sort(triangles.begin() + start, triangles.begin() + end + 1, ySort);//[start, end + 1)
		break;
	case 2:
		sort(triangles.begin() + start, triangles.begin() + end + 1, zSort);//[start, end + 1)
		break;
	}

	int median = (start + end) / 2;

	root = median;

	glm::vec3 minL = triangles[root].min;
	glm::vec3 minR = triangles[root].min; 
	glm::vec3 maxL = triangles[root].max;
	glm::vec3 maxR = triangles[root].max;
	//[start, start + median - 1]
	triangles[root].leftIndex = buildTree(triangles, (axis + 1) % 3, indexOffset, start, median - 1, &minL, &maxL);
	//[start + median + 1, start + end]
	triangles[root].rightIndex = buildTree(triangles, (axis + 1) % 3, indexOffset, median + 1, end, &minR, &maxR);
	
	triangles[root].min = glm::min(triangles[root].min, glm::min(minL, minR));
	triangles[root].max = glm::max(triangles[root].max, glm::max(maxL, maxR));

	if (min != nullptr) *min = triangles[root].min;
	if (max != nullptr) *max = triangles[root].max;

	return root + indexOffset;
}
