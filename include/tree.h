#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <map>
#include <algorithm>
#include <set>
#include <string>

#include "datapoint.h"
#include "treenode.h"
#include "dataloader.h"

using namespace std;

class Tree {

	TreeNode* m_root;
    vector<TreeNode> m_nodes;
	int m_treeSize;
    TreeParams m_params;
    vector<vector<int>> m_leafs;
    vector<int> rootLabelHistogram;
	vector<Varray<float>> m_meanDataLabel;

public:

	~Tree() { }

	Tree() {
		m_root = NULL;
		m_treeSize = 0;
	}
	
	void setParams(TreeParams params) { m_params = params; };

	void buildTree(const DataLoader &trData, const DataLoader &trLabel);
	
	void normalizeLableHist();

    void testBatch(const DataLoader &teData);

	Varray<float> testData(int idx, const DataLoader &teData, 
		const vector<Varray<float>> &meanDataLabel) const;

	vector<int> getRootLabelHistogram() const { return rootLabelHistogram; }

	vector<Varray<float>> getMeanDataLabel() { return m_meanDataLabel; }
	
	int getTreeDepth();
	
	int getTreeSize() { return m_treeSize; }

};
