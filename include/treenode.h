#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <forward_list>
#include <string>
#include <functional>

#include "datapoint.h"
#include "dataloader.h"

using namespace std;

struct TreeParams {
    float alpha;
    float l1;
    float l2;
    int m;
    int k;
    int d;
    int nMax;
    int epochs;
    float gamma;
    float beta;
    bool sparse;
    bool entropyLoss;
    bool muFlag;
    float coefL1;
    float coefL2;
    bool exampleLearn;
};

class TreeNode {

private:

    TreeNode* m_root;
	TreeParams *m_params;
    int m_depth;
    vector<int> m_dataIndex; // data index of train data reaching to a node.
    vector<Varray<float>> m_weightSparse;
    vector<float> m_mean;
    vector<float> m_std;

	static vector<vector<float>> m_weight;
	static vector<int> m_lastUpdate;
	static vector<int> m_dataClassCounter;
	static vector<float> m_probAllVector;
	static vector<vector<float>> m_probSingleVector;
	static vector<vector<float>> m_sNol;
	static vector<vector<float>> m_gNol;
	static vector<float> m_nNol;

	static vector<vector<int>> m_children_labelHistogram;

public:

	int m_nodeId;
    vector<int> m_children;
    Varray<int> m_labelHistogramSparse;
	vector<pair<int, float>> m_NormalLabelHistogramSparse;

    TreeNode(TreeParams *params) { m_params = params; m_depth = 0; };

	void initialize();

    void meanStdCalc(const DataLoader &trData, const DataLoader &trLabel);

	void weightUpdate(const DataLoader &trData, const DataLoader &trLabel,
		vector<int>& rooLabelHist, int maxLabelRoot);

	int makeChildren(const DataLoader &trData, const DataLoader &trLabel,
		const int& N, vector<TreeNode>& nodes);
		
    void destroyChildren(vector<TreeNode>& nodes); 

    void testBatch(const DataLoader &teData, vector<TreeNode>& nodes,
                   vector<vector<int>>& leafs);

    void addHistogram(labelEst& labelHistogramSum, int leafCount) const;

    bool isLeaf() const {
        if (m_children.size() == 0)
            return true;
        else
            return false;
    }  
	
	~TreeNode() { }

	void normalizeLabelHist();

    void setRoot(TreeNode* x) { m_root = x; }

    void setDepth(int x) { m_depth = x; }

    int getDepth() { return m_depth; }

    int getChild(int m) { return m_children[m]; }

    void setDataIndex(vector<int> x) { m_dataIndex = x; }

    void setLabelHistogramSparse(vector<int> x) { m_labelHistogramSparse.set(x); }

};
