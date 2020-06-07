#include "tree.h"

using namespace std;

void savemu(vector<Varray<float>> mu, string strFile);

void Tree::buildTree(const DataLoader &trData, const DataLoader &trLabel)
{

    m_nodes.resize(m_params.nMax, TreeNode(&m_params));
	m_root = &(m_nodes[0]);

    m_root->m_nodeId = 0; 
    m_root->setRoot(m_root);
    m_root->setDepth(0);
    m_root->initialize();
    multimap<int, int> nodesArray;
    nodesArray.insert(pair<int, int>(0, m_root->m_nodeId));

    vector<int> dataIndexRoot(trData.size());
    for (int i = 0; i < trData.size(); i++)
        dataIndexRoot[i] = i;
    random_shuffle(dataIndexRoot.begin(), dataIndexRoot.end());
    m_root->setDataIndex(dataIndexRoot); 
    rootLabelHistogram.resize(m_params.k, 0); 
    m_meanDataLabel.resize(m_params.k);
    vector<forward_list<int>> idxListForLabel(m_params.k);

    for (size_t i = 0; i < dataIndexRoot.size(); i++) {
        for (int j = 0; j < trLabel.getDataPoint(dataIndexRoot[i]).size(); j++) {
            int k = trLabel.getDataPoint(dataIndexRoot[i]).getLabelVector()[j];
            rootLabelHistogram[k] += 1;
            idxListForLabel[k].push_front(i);
        }
    }
    
    if (m_params.muFlag) {
        for (int k = 0; k < m_params.k; k++) {
            vector<float> dataMean(m_params.d, 0.f);
            while (!idxListForLabel[k].empty()) {
                int index = idxListForLabel[k].front();
                idxListForLabel[k].pop_front();
                const DataPoint& dataNormal = trData.getDataPoint(dataIndexRoot[index]);
                const vector<int>& dataPointIndeces = dataNormal.getDataIndeces();
                const vector<float>& dataPointValues = dataNormal.getDataValues();
                for (size_t i = 0; i < dataPointIndeces.size(); i++) {
                    int idx = dataPointIndeces[i];
                    float val = dataPointValues[i];
                    dataMean[idx] += val;
                }
            }
            for (int i = 0; i < m_params.d; i++) {
                dataMean[i] /= rootLabelHistogram[k];
            }
            m_meanDataLabel[k] = Varray<float>(dataMean);
        }
        savemu(m_meanDataLabel, "../results/mu_"+trData.getDataSetName());
    }

    int maxLabelRoot = 0;
    for (int i = 0; i < m_params.k; i++)
        if(rootLabelHistogram[i] > maxLabelRoot)
            maxLabelRoot = rootLabelHistogram[i];

    m_root->setLabelHistogramSparse(rootLabelHistogram);

	int N = 1;
    while (!nodesArray.empty()) {
        int n = (*--nodesArray.end()).second; // the last element in the multimap has the maximum priority
        nodesArray.erase(--nodesArray.end());

        if (N < m_params.nMax - m_params.m + 1) {

            if (!m_params.sparse) {
                m_nodes[n].meanStdCalc(trData, trLabel);
            }
            m_nodes[n].weightUpdate(trData, trLabel, rootLabelHistogram, maxLabelRoot);
   
            int numOfChildren = m_nodes[n].makeChildren(trData, trLabel, N, m_nodes);

            N += numOfChildren;
            if (N == 1 || N == 11 || N == 101 || N == 1001 || N == 10001 || N == 100001 || N == 300001)
                cerr << "N = " << N << endl;

            if (numOfChildren == m_params.m)
                for (int m = 0; m < m_params.m; m++) {
                    int ch = m_nodes[n].getChild(m);
                    m_nodes[ch].setRoot(m_root);
                }

            // update the nodesArray with the created children
            if (numOfChildren == m_params.m) {
                for (int m = 0; m < m_params.m; m++) {
                    int priority = 0;
                    int maxValue = 0;
                    int ch = m_nodes[n].getChild(m);
					for (size_t i = 0; i < m_nodes[ch].m_labelHistogramSparse.size(); i++) {
						int labelCount = m_nodes[ch].m_labelHistogramSparse.myMap.value[i];;
						priority += labelCount;
                        if (labelCount > maxValue)
                            maxValue = labelCount;
                    }
                    priority -= maxValue;
                    nodesArray.insert(pair<int, int>(priority, ch));
                }
            }
        }
    }
    m_treeSize = N;
}

void Tree::normalizeLableHist() {
	for (size_t n = 0; n < m_nodes.size(); n++) {
		if (m_nodes[n].isLeaf())
			m_nodes[n].normalizeLabelHist();
	}
}

void Tree::testBatch(const DataLoader &teData) {
	m_leafs.resize(teData.size());
	vector<int> dataIndexRoot(teData.size());
	for (int i = 0; i < teData.size(); i++)
		dataIndexRoot[i] = i;
	m_root->setDataIndex(dataIndexRoot);

	for (size_t n = 0; n < m_nodes.size(); n++) {
		m_nodes[n].testBatch(teData, m_nodes, m_leafs);
	}
}


Varray<float> Tree::testData(int idx, const DataLoader &teData,
	const vector<Varray<float>> &meanDataLabel) const {
	labelEst labelHistogramSum;
	labelHistogramSum.regular.resize(m_params.k, 0.f);
	int leafCount = m_leafs[idx].size();
	for (int n = 0; n < leafCount; n++) {
		m_nodes[m_leafs[idx][n]].addHistogram(labelHistogramSum, leafCount);
	}
	
	if (m_params.beta != 1) {
		for (int k = 0; k < m_params.k; k++) {
			if (labelHistogramSum.regular[k] != 0.f) {
				labelHistogramSum.regular[k] = m_params.beta*labelHistogramSum.regular[k]
					+ (1 - m_params.beta) / (1 + exp(m_params.gamma*teData.getDataPoint(idx).dist(meanDataLabel[k]) / 2));
			}
		}
	}

	return (Varray<float>(labelHistogramSum.regular));
}


void savemu(vector<Varray<float>> mu, string strFile) {
	ofstream fs;
	fs.open(strFile, ios::out | ios::binary);
	if (!fs.is_open()) {
		cout << "Cannot open file " << strFile << endl;
	}
	else {
		int muSize = mu.size();
		fs.write(reinterpret_cast<const char*>(&muSize), sizeof(muSize));
		for (int k = 0; k < muSize; k++) {
			int muKSize = mu[k].myMap.index.size();
			fs.write(reinterpret_cast<const char*>(&muKSize), sizeof(muKSize));
			for (int i = 0; i < muKSize; i++) {
				fs.write(reinterpret_cast<const char*>(&mu[k].myMap.index[i]),
					sizeof(mu[k].myMap.index[i]));
				fs.write(reinterpret_cast<const char*>(&mu[k].myMap.value[i]),
					sizeof(mu[k].myMap.value[i]));
			}
		}
	}
}

vector<Varray<float>> loadmu(string strFile) {
	vector<Varray<float>> mu;
	ifstream fs;
	fs.open(strFile, ios::in | ios::binary);
	int muSize = 0;
	fs.read(reinterpret_cast<char*>(&muSize), sizeof(muSize));
	mu.resize(muSize);
	for (int k = 0; k < muSize; k++) {
		int muKSize = 0;
		fs.read(reinterpret_cast<char*>(&muKSize), sizeof(muKSize));
		mu[k].myMap.index.resize(muKSize);
		mu[k].myMap.value.resize(muKSize);
		for (int i = 0; i < muKSize; i++) {
			fs.read(reinterpret_cast<char*>(&mu[k].myMap.index[i]),
				sizeof(mu[k].myMap.index[i]));
			fs.read(reinterpret_cast<char*>(&mu[k].myMap.value[i]),
				sizeof(mu[k].myMap.value[i]));
		}
	}
	return mu;
}

int Tree::getTreeDepth() {
    vector<int> depthArray;
    int depth = 0;
    for (size_t i = 0; i < m_nodes.size(); i++)
        depth = max(depth, m_nodes[i].getDepth());
    return depth;
}