#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/numpy.h>
#include <iostream>
#include <thread>
#include<stdlib.h>
#include <ctime>
#include<vector>
#include<deque>
#include <queue>
#include <bitset>
#include<string>
#include<cmath>
#include <memory>
#define N 256
#define M 10000
#define BITS_LENGTH  20 
#define BITSET_MAX 1000000
using namespace std;
namespace py = pybind11;


struct Octant {
	vector<shared_ptr<Octant>> children;
	vector<float> center;
	float extent;
	int depth;
	bool is_leaf;
	int octant;
	Octant(vector<shared_ptr<Octant>> _children, vector<float> _center, float _extent, int _depth, bool _is_leaf) {
		children = _children;
		center = _center;
		extent = _extent;
		depth = _depth;
		is_leaf = _is_leaf;
		octant = 0;
	}
};


struct Node {
	int index;
	int fatherIndex;
	map<int, vector<int>> dis_n_neibours;

	//features
	vector<float> location;
	int depth;
	int curIndex;
	int parentOccupancy;

	// ... can not be used for prediction
	int occupancy;

	Node() {
		index = -1;
		fatherIndex = -1;
		dis_n_neibours.clear();

		location = { 0.1,0.1,0.1 };
		depth = -1;
		curIndex = 0;
		parentOccupancy = 0;

		occupancy = 0;
	}
};

struct process_info {
	map<int, Node> nodedict;
	map<int,int> layerIndexs;
	int maxIndex;
	int maxLayer;
	process_info() {
		nodedict.clear();
		layerIndexs.clear();
		maxIndex = -1;
	}
};


process_info bfs_process_octree(shared_ptr<Octant> root) {

	map<int, Node> nodedict; nodedict.clear();
	map<int,int> layerIndexs; layerIndexs.clear();
	queue<shared_ptr<Octant>> octantQueue; 
	octantQueue.push(root);
	int index = 0;
	int maxLayer = 0;

	while (!octantQueue.empty()) {
		shared_ptr<Octant> top = octantQueue.front();
		octantQueue.pop();
		int remains = octantQueue.size();
		if (nodedict.find(index) == nodedict.end()) {
			//not found
			Node newNode;
			nodedict[index] = newNode;
		}
		nodedict[index].index = index;
		nodedict[index].location = top->center;
		nodedict[index].depth = top->depth;

		if(layerIndexs.find(top->depth) == layerIndexs.end()){
			layerIndexs[top->depth] = index;
			maxLayer = max(maxLayer,top->depth);
		}


		//count octant
		int curOctant = 0;
		int i = 0;
		for (auto x : top->children) {
			if (x != nullptr) {
				curOctant += pow(2, 7 - i);
			}
			i = i + 1;
		}

		nodedict[index].occupancy = curOctant;


		int num = 1;
		int curIndex = 0;

		for (auto x : top->children) {
			if (x != nullptr) {
				octantQueue.push(x);
				if (nodedict.find(index + remains + num) == nodedict.end()) {
					//not found
					Node newNode;
					nodedict[index + remains + num] = newNode;
					nodedict[index + remains + num].fatherIndex = index;
					nodedict[index + remains + num].curIndex = curIndex;
					nodedict[index + remains + num].parentOccupancy = curOctant;
				}
				num = num + 1;
			}
			curIndex = curIndex + 1;
		}
		index = index + 1;
	}


	process_info result;
	result.nodedict = nodedict;
	result.layerIndexs = layerIndexs;
	result.maxIndex = index - 1;
	result.maxLayer = maxLayer;
	return result;
}


vector<int> findNeiboursForNode(map<int, Node*>& nodedict, int maxIndex, int tarIndex, int n) {
	vector<int> neibourIndexs; neibourIndexs.clear();

	if (nodedict[tarIndex]->dis_n_neibours.find(n) != nodedict[tarIndex]->dis_n_neibours.end()) {
		// found
		neibourIndexs = nodedict[tarIndex]->dis_n_neibours[n];
	}
	else {
		vector<int> dis1Indexs = nodedict[tarIndex]->dis_n_neibours[1];
		set<int> neibourSet; neibourSet.clear();
		for (auto idx : dis1Indexs) {
			neibourSet.insert(idx);
			vector<int> tmpIndexs = findNeiboursForNode(nodedict, maxIndex, idx, n - 1);
			for (auto idx2 : tmpIndexs) {
				neibourSet.insert(idx2);
			}
		}
		neibourIndexs.assign(neibourSet.begin(), neibourSet.end());
		nodedict[tarIndex]->dis_n_neibours[n] = neibourIndexs;
	}
	return neibourIndexs;
}

vector<Node*> getNodeListFromList(map<int, Node*>& nodedict, int tarIndex, vector<int>& neibourIndexs) {
	vector<Node*> nodelist; nodelist.clear();
	int myDepth = nodedict[tarIndex]->depth;
	for (auto idx : neibourIndexs) {
		if (nodedict[idx]->depth < myDepth) {
			nodelist.push_back(nodedict[idx]);
		}
	}
	return nodelist;
}

tuple<float, float, float, int, int, int> flattenFeatures(Node* node) {
	tuple<float, float, float, int, int, int> ans(
		node->location[0],
		node->location[1],
		node->location[2],
		node->depth,
		node->curIndex,
		node->parentOccupancy
	);
	return ans;
}

shared_ptr<Octant> octree_recursive_build(shared_ptr<Octant> root, vector<vector<float> >& db, vector<float>& center, float& extent, vector<int>& point_indices, int depth, int& max_depth)
{

	if (point_indices.size() == 0 || depth > max_depth)
	{
		shared_ptr<Octant>emptyptr;
		return emptyptr;
	}
	if (!root)
	{
		vector<shared_ptr<Octant>>childs;
		for (int i = 0; i < 8; i++)
		{
			shared_ptr<Octant>c;
			childs.push_back(c);
		}
		shared_ptr<Octant> new_root(new Octant(childs, center, extent, depth, true));
		root = new_root;

	}
	//determine whether to split this octant
	if (root->depth != max_depth)
	{
		root->is_leaf = false;
		vector<vector<int>>children_point_indices(8);
		for (auto point_idx : point_indices)
		{
			vector<float>point_db = db[point_idx];
			int morton_code = 0;
			if (point_db[0] > center[0])
				morton_code = morton_code | 1;
			if (point_db[1] > center[1])
				morton_code = morton_code | 2;
			if (point_db[2] > center[2])
				morton_code = morton_code | 4;
			children_point_indices[morton_code].push_back(point_idx);
		}
		//count octant
		for (int i = 0; i < 8; i++)
		{
			if (children_point_indices[i].size() > 0)
			{
				int octplus = pow(2, 7 - i);
				root->octant += octplus;
			}

		}
		// create children
		vector<float>factor = { -0.5, 0.5 };
		for (int i = 0; i < 8; i++)
		{
			float child_center_x = center[0] + factor[(i & 1) > 0] * extent;
			float child_center_y = center[1] + factor[(i & 2) > 0] * extent;
			float child_center_z = center[2] + factor[(i & 4) > 0] * extent;
			float child_extent = 0.5 * extent;
			vector<float>child_center = { child_center_x, child_center_y, child_center_z };
			root->children[i] = octree_recursive_build(root->children[i], db, child_center, child_extent, children_point_indices[i], depth + 1, max_depth);

		}

	}
	return root;

}

struct RCE {
	shared_ptr<Octant> root;
	vector<float>center;
	float extent;
};
RCE octree_construction(vector<vector<float>>& db_np, int& max_depth)
{
	int pointnums = db_np.size();
	if (pointnums == 0)
	{
		RCE rce;
		return rce;
	}
	else
	{
		float xmax = db_np[0][0];
		float xmin = db_np[0][0];
		float ymax = db_np[0][1];
		float ymin = db_np[0][1];
		float zmax = db_np[0][2];
		float zmin = db_np[0][2];
		float xsum = 0;
		float ysum = 0;
		float zsum = 0;
		vector<int>db_index;
		for (int i = 0; i < pointnums; i++)
		{
			xmax = max(db_np[i][0], xmax);
			xmin = min(db_np[i][0], xmin);
			ymax = max(db_np[i][1], ymax);
			ymin = min(db_np[i][1], ymin);
			zmax = max(db_np[i][2], zmax);
			zmin = min(db_np[i][2], zmin);
			db_index.push_back(i);
		}
		float db_extent_temp = max(xmax - xmin, ymax - ymin);
		float db_extent = max(db_extent_temp, zmax - zmin) * 0.5;
		vector<float>db_center = { (xmax + xmin) / 2,(ymax + ymin) / 2,(zmax + zmin) / 2 };
		int db_depth = 1;
		shared_ptr<Octant> root;
		root = octree_recursive_build(root, db_np, db_center, db_extent, db_index, db_depth, max_depth);
		RCE rce;
		rce.center = db_center;
		rce.extent = db_extent;
		rce.root = root;
		return rce;
	}



}


void DFS(shared_ptr<Octant> root, vector<int>& d_set, int& max_depth)
{
	if (!root)
		return;
	d_set.push_back(root->octant);
	if (root->depth == max_depth)
		return;
	else
	{
		for (int i = 0; i < 8; i++)
			DFS(root->children[i], d_set, max_depth);
	}
}
vector<int> pre_order(shared_ptr<Octant> root, int& max_depth)
{

	vector<int>d_set;
	if (root)
		DFS(root, d_set, max_depth);
	return d_set;

}


shared_ptr<Octant> deserialize(shared_ptr<Octant> root, deque<int>& arr, vector<float> center, float extent, int depth, int& max_depth)
{
	if (arr.size() == 0)
	{
		shared_ptr<Octant>t;
		return t;
	}
	int octant = arr.front();
	arr.pop_front();
	if (!root)
	{
		vector<shared_ptr<Octant>>childs;
		for (int i = 0; i < 8; i++)
		{
			shared_ptr<Octant>c;
			childs.push_back(c);
		}
		shared_ptr<Octant> new_root(new Octant(childs, center, extent, depth, true));
		root = new_root;
	}
	if (root->depth == max_depth)
	{

		vector<float>factor = { -0.5, 0.5 };
		for (int i = 0; i < 8; i++)
		{
			int octant_pos = pow(2, 7 - i);
			if (root->octant & octant_pos)
			{
				float child_center_x = root->center[0] + factor[(i & 1) > 0] * root->extent;
				float child_center_y = root->center[1] + factor[(i & 2) > 0] * root->extent;
				float child_center_z = root->center[2] + factor[(i & 4) > 0] * root->extent;
				float child_extent = 0.5 * root->extent;
				vector<float>child_center = { child_center_x, child_center_y, child_center_z };
				vector<shared_ptr<Octant>>childs;
				for (int j = 0; j < 8; j++)
				{
					shared_ptr<Octant>c;
					childs.push_back(c);
				}
				shared_ptr<Octant> new_root(new Octant(childs, child_center, child_extent, root->depth + 1, true));
				root->children[i] = new_root;
			}


		}
	}
	else
	{
		root->is_leaf = false;
		root->octant = octant;
		//create children
		vector<float>factor = { -0.5, 0.5 };
		for (int i = 0; i < 8; i++)
		{
			int octant_pos = pow(2, 7 - i);
			if (root->octant & octant_pos)
			{
				float child_center_x = center[0] + factor[(i & 1) > 0] * extent;
				float child_center_y = center[1] + factor[(i & 2) > 0] * extent;
				float child_center_z = center[2] + factor[(i & 4) > 0] * extent;
				float child_extent = 0.5 * extent;
				vector<float>child_center = { child_center_x, child_center_y, child_center_z };
				root->children[i] = deserialize(root->children[i], arr, child_center, child_extent, depth + 1, max_depth);
			}


		}
	}
	return root;
}

shared_ptr<Octant> preorder_deserialize(vector<int>& arr, vector<float> center, float extent, int& max_depth)
{
	shared_ptr<Octant>root;
	int depth = 1;
	deque<int> arr_deque;
	for (int i = 0; i < arr.size(); i++)
	{
		arr_deque.push_back(arr[i]);
	}

	root = deserialize(root, arr_deque, center, extent, depth, max_depth);
	return root;
}


void leaf_DFS(shared_ptr<Octant>root, vector<vector<float>>& points)
{
	if (!root)
	{
		return;
	}
	else
	{
		if (root->is_leaf)
		{
			points.push_back(root->center);
			return;
		}
		else
		{
			for (int i = 0; i < 8; i++)
				leaf_DFS(root->children[i], points);
			return;
		}
	}
}
vector<vector<float>> octree2pointcloud(shared_ptr<Octant>root)
{
	vector<vector<float>> points;
	if (!root)
	{

		return points;
	}
	else
	{
		leaf_DFS(root, points);
		return points;
	}
}





PYBIND11_MODULE(myutils, m) {
	m.doc() = "pybind11 myutils plugin"; // optional module docstring

	py::class_<process_info>(m, "process_info")
		.def_readwrite("nodedict", &process_info::nodedict)
		.def_readwrite("layerIndexs", &process_info::layerIndexs)
		.def_readwrite("maxIndex", &process_info::maxIndex)
		.def_readwrite("maxLayer", &process_info::maxLayer);

	py::class_<Octant, shared_ptr<Octant>>(m, "Octant")
		.def(py::init<vector<shared_ptr<Octant>>, vector<float>, float, int, bool>())
		.def_readwrite("children", &Octant::children, py::keep_alive<0, 1>())
		.def_readwrite("center", &Octant::center)
		.def_readwrite("extent", &Octant::extent)
		.def_readwrite("is_leaf", &Octant::is_leaf)
		.def_readwrite("depth", &Octant::depth)
		.def_readwrite("octant", &Octant::octant);

	py::class_<Node>(m, "Node")
		.def(py::init<>())
		.def_readwrite("index", &Node::index)
		.def_readwrite("fatherIndex", &Node::fatherIndex)
		.def_readwrite("dis_n_neibours", &Node::dis_n_neibours)
		.def_readwrite("location", &Node::location)
		.def_readwrite("depth", &Node::depth)
		.def_readwrite("curIndex", &Node::curIndex)
		.def_readwrite("parentOccupancy", &Node::parentOccupancy)
		.def_readwrite("occupancy", &Node::occupancy);

	m.def("bfs_process_octree", &bfs_process_octree, "A function which adds aba tmp expose");
	m.def("findNeiboursForNode", &findNeiboursForNode, "tmp expose");
	//m.def("generateDataFromOctree", &generateDataFromOctree, "to be exposed");

	py::class_<RCE>(m, "RCE")
		.def(py::init<shared_ptr<Octant>, vector<float>, float>())
		.def_readwrite("root", &RCE::root)
		.def_readwrite("center", &RCE::center)
		.def_readwrite("extent", &RCE::extent);


	m.def("octree_recursive_build", &octree_recursive_build, "to be exposed");
	m.def("octree_construction", &octree_construction, "to be exposed");
	m.def("DFS", &DFS, "to be exposed");
	m.def("pre_order", &pre_order, "to be exposed");
	m.def("deserialize", &deserialize, "to be exposed");
	m.def("preorder_deserialize", &preorder_deserialize, "to be exposed");
	m.def("leaf_DFS", &leaf_DFS, "to be exposed");
	m.def("octree2pointcloud", &octree2pointcloud, "to be exposed");

}
