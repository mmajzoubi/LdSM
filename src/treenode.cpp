#include "treenode.h"
#include <float.h>

using namespace std;

vector<vector<float>> TreeNode::m_weight;
vector<int> TreeNode::m_lastUpdate;
vector<int> TreeNode::m_dataClassCounter;
vector<float> TreeNode::m_probAllVector;
vector<vector<float>> TreeNode::m_probSingleVector;
vector<vector<float>> TreeNode::m_sNol;
vector<vector<float>> TreeNode::m_gNol;
vector<float> TreeNode::m_nNol;
vector<vector<int>> TreeNode::m_children_labelHistogram;

void TreeNode::initialize() {
	m_weight.clear();
	m_lastUpdate.clear();
	m_dataClassCounter.clear();
	m_probAllVector.clear();
	m_probSingleVector.clear();
	m_sNol.clear();
	m_gNol.clear();
	m_nNol.clear();
	m_children_labelHistogram.clear();

	m_weight.resize(m_params->m, vector<float>(m_params->d, 0.f));
	m_lastUpdate.resize(m_params->d, 0);
	m_dataClassCounter.resize(m_params->k, 0); // number of data points in each class at each node
	m_probAllVector.resize(m_params->m, 0.f);
	m_probSingleVector.resize(m_params->m, vector<float>(m_params->k, 0.f));
	m_sNol.resize(m_params->m, vector<float>(m_params->d, 0.f));
	m_gNol.resize(m_params->m, vector<float>(m_params->d, 0.f));
	m_nNol.resize(m_params->m, 0.f);
	m_children_labelHistogram.resize(m_params->m, vector<int>(m_params->k, 0));

}

void TreeNode::meanStdCalc(const DataLoader &trData, const DataLoader &trLabel) { // only for non-sparse data

	int dim = trData.getDim();
	int dataIndexSize = m_dataIndex.size();
	m_mean.resize(dim, 0.f);
	m_std.resize(dim, 0.f);

	for (int index = 0; index < dataIndexSize; index++) {
		const DataPoint& data = trData.getDataPoint(m_dataIndex[index]);
		const vector<int>& dataPointIndeces = data.getDataIndeces();
		const vector<float>& dataPointValues = data.getDataValues();
		for (size_t i = 0; i < dataPointIndeces.size(); i++) {
			int idx = dataPointIndeces[i];
			float val = dataPointValues[i];
			m_mean[idx] += val;
		}
	}

	for (int i = 0; i < dim; i++) {
		m_mean[i] /= dataIndexSize;
	}

	for (int index = 0; index < dataIndexSize; index++) {
		const DataPoint& data = trData.getDataPoint(m_dataIndex[index]);
		const vector<int>& dataPointIndeces = data.getDataIndeces();
		const vector<float>& dataPointValues = data.getDataValues();
		for (size_t i = 0; i < dataPointIndeces.size(); i++) {
			int idx = dataPointIndeces[i];
			float val = dataPointValues[i];
			m_std[idx] += (val - m_mean[idx])*(val - m_mean[idx]);
		}
	}

	for (int i = 0; i < dim; i++) {
		m_std[i] = sqrt(m_std[i] / (dataIndexSize - 1));
	}
	m_std[dim - 1] = 1.f;
	
}

float gradAbsoluteLoss(int label, float prediction) {
	float expNegPred = exp(-prediction);
	float gradient = 0.5f * expNegPred / (1.f + expNegPred) / (1.f + expNegPred);
	if (label == 1)
		return -gradient;

	return gradient;
}

float gradEntropyLoss(int label, float prediction) {
	return -(label - 1 / (1 + exp(-prediction)));
}

void decayReg(float &weight, float coefL1, float coefL2, int stepsCount) {
	float decayL1 = coefL1 * stepsCount;
	if (decayL1 > fabs(weight))
		weight = 0.f;
	else if (decayL1 < fabs(weight)) {
		if (weight > 0)
			weight -= decayL1;
		else
			weight += decayL1;
	}
	float decayL2 = 1.f;
	if (coefL2 != 0)
		decayL2 = pow(1.f - coefL2, stepsCount);
	weight *= decayL2;
}

void TreeNode::weightUpdate(const DataLoader &trData, const DataLoader &trLabel,
	vector<int>& rooLabelHist, int maxLabelRoot) {
	
	std::function<float(int, float)> calcGradient;
	if (m_params->entropyLoss)
		calcGradient = gradEntropyLoss;
	else
		calcGradient = gradAbsoluteLoss;

	int d = m_params->d; 
	float clip = 1; // for clipping data.dot(weight)

	if (m_params->coefL1 != 0 || m_params->coefL2 != 0)
		fill(m_lastUpdate.begin(), m_lastUpdate.end(), 0);
	fill(m_dataClassCounter.begin(), m_dataClassCounter.end(), 0);
	fill(m_probAllVector.begin(), m_probAllVector.end(), 0.f);
	fill(m_nNol.begin(), m_nNol.end(), 0.f);
	for (int m = 0; m < m_params->m; m++) {
		fill(m_sNol[m].begin(), m_sNol[m].end(), 0.f);
		fill(m_gNol[m].begin(), m_gNol[m].end(), 0.f);
		fill(m_probSingleVector[m].begin(), m_probSingleVector[m].end(), 0.f);
	}

	const float weightMag = 1.f / m_params->d;
	for (int m = 0; m < m_params->m; m++) {
		for (int j = 0; j < d; j++)
			m_weight[m][j] = ((float)(rand() - RAND_MAX / 2) / RAND_MAX) * weightMag;
	}

	int dataCounter = 0; 

	float J; // objective function initialization
	vector<int> yhat; // regressors' label

	int tStep = 0;
	for (int e = 0; e < m_params->epochs; e++) {
		for (size_t index = 0; index < m_dataIndex.size(); index++) {
			tStep++;

			float lrWeight = 0.0;
			const vector<int>& labelVector = trLabel.getDataPoint(m_dataIndex[index]).getLabelVector();
			int labelSize = labelVector.size();
			for (int l = 0; l < labelSize; l++) {
				int k = labelVector[l];
				lrWeight += (float)maxLabelRoot / (float)rooLabelHist[k];
				m_dataClassCounter[k]++;
			}
			dataCounter += labelSize;
			float alpha = 0.0;
			if (m_params->exampleLearn)
				alpha = m_params->alpha * lrWeight;
			else
				alpha = m_params->alpha;

			// optimum parameters
			float optJ = FLT_MAX;
			float optJPurity = FLT_MAX;
			vector<int> optYhat(m_params->m);

			for (int s = 1; s < (1 << m_params->m); s++) {
				vector<float> tmpProbAllVector(m_params->m, 0);
				vector<vector<float>> tmpProbSingleVector(m_params->m, vector<float>(labelSize, 0.f));
				vector<int> tmpYhat(m_params->m);

				for (int m = 0; m < m_params->m; m++) {
					int a = (1 << m);
					tmpYhat[m] = ((s & a) == 0) ? 0 : 1;
					tmpProbAllVector[m] = (m_probAllVector[m] * (dataCounter - labelSize) +
						labelSize * tmpYhat[m]) /
						(float)dataCounter;

					for (int l = 0; l < labelSize; l++) {
						int k = labelVector[l];
						tmpProbSingleVector[m][l] = (m_probSingleVector[m][k] *
							(m_dataClassCounter[k] - 1) + tmpYhat[m]) /
							(float)m_dataClassCounter[k];
					}
				} 

				float tmpBalance = 0.0;
				for (int m1 = 0; m1 < m_params->m; m1++) {
					for (int m2 = m1 + 1; m2 < m_params->m; m2++) {
						tmpBalance += abs(tmpProbAllVector[m1] - tmpProbAllVector[m2]);
					}
				}

				float tmpBoth = 0.0;
				for (int m = 0; m < m_params->m; m++)
					tmpBoth += tmpProbAllVector[m];
				tmpBoth -= 1.0;
				if (tmpBoth < 0.0)
					tmpBoth = -tmpBoth;

				float tmpJPurity = 0.0;
				for (int l = 0; l < labelSize; l++) {
					int k = labelVector[l];
					float purity = 0.0;
					for (int m1 = 0; m1 < m_params->m; m1++) {
						for (int m2 = m1 + 1; m2 < m_params->m; m2++) {
							float tmp = tmpProbSingleVector[m1][l] - tmpProbSingleVector[m2][l];
							if (tmp < 0.0)
								tmp = -tmp;
							purity += tmp;
						}
					}
					tmpJPurity += purity * ((float)m_dataClassCounter[k] / (float)dataCounter);
				}
				float tmpJ = tmpBalance + m_params->l1 * tmpBoth - m_params->l2 * tmpJPurity;
				if (tmpJ == optJ && (s != (1 << m_params->m) - 1)) {
					int r = rand() % 2;
					if (r >= 1) {
						optJ = tmpJ;
						optJPurity = tmpJPurity;
						optYhat = tmpYhat;
					}
				}

				if (tmpJ < optJ) {
					optJ = tmpJ;
					optJPurity = tmpJPurity;
					optYhat = tmpYhat;
				}
			}

			J = optJ;
			yhat = optYhat;

			vector<float> newDotProduct(m_params->m);

			if (m_params->sparse) {
				const DataPoint& dataNormal = trData.getDataPoint(m_dataIndex[index]);
				const vector<int>& dataPointIndeces = dataNormal.getDataIndeces();
				const vector<float>& dataPointValues = dataNormal.getDataValues();

				for (int m = 0; m < m_params->m; m++) {
					for (size_t i = 0; i < dataPointIndeces.size(); i++) {
						int idx = dataPointIndeces[i];
						float val = dataPointValues[i];
						if (m_params->coefL1 != 0 || m_params->coefL2 != 0) {
							decayReg(m_weight[m][idx], m_params->coefL1, m_params->coefL2, tStep - m_lastUpdate[idx]);
						}
						if (abs(val) > m_sNol[m][idx]) {
							m_weight[m][idx] = m_weight[m][idx] * m_sNol[m][idx] / abs(val);
							m_sNol[m][idx] = abs(val);
						}
					}

					float dotProduct = dataNormal.dot(m_weight[m]);

					for (size_t i = 0; i < dataPointIndeces.size(); i++) {
						int idx = dataPointIndeces[i];
						float val = dataPointValues[i];
						m_nNol[m] += (val * val) / (m_sNol[m][idx] * m_sNol[m][idx]);
					}

					float gradient_const = calcGradient(yhat[m], dotProduct);
					for (size_t i = 0; i < dataPointIndeces.size(); i++) {
						int idx = dataPointIndeces[i];
						float val = dataPointValues[i];
						float gradient = gradient_const * val;
						m_gNol[m][idx] += (gradient * gradient);
						if (m_gNol[m][idx] != 0)
							m_weight[m][idx] -= alpha * sqrt(tStep / m_nNol[m]) * gradient /
							(m_sNol[m][idx] * sqrt(m_gNol[m][idx]));
					}
				}
				if (m_params->coefL1 != 0 || m_params->coefL2 != 0) {
				for (size_t i = 0; i < dataPointIndeces.size(); i++)
					m_lastUpdate[i] = tStep;
				}
				
				for (int m = 0; m < m_params->m; m++) {
					newDotProduct[m] = dataNormal.dot(m_weight[m]);
				}
			}
			else {
				DataPoint data = trData.getDataPoint(m_dataIndex[index]);
				DataPoint dataNormal = data.normal(m_mean, m_std);
				const vector<int>& dataPointIndeces = dataNormal.getDataIndeces();
				const vector<float>& dataPointValues = dataNormal.getDataValues();

				for (int m = 0; m < m_params->m; m++) {
					
					float dotProduct = dataNormal.dot(m_weight[m]);
					float gradient_const = calcGradient(yhat[m], dotProduct);
					for (size_t i = 0; i < dataPointIndeces.size(); i++) {
						int idx = dataPointIndeces[i];
						float val = dataPointValues[i];
						float gradient = gradient_const * val;
						m_weight[m][idx] -= alpha * gradient;
					}
				}
								
				for (int m = 0; m < m_params->m; m++) {
					newDotProduct[m] = dataNormal.dot(m_weight[m]);
				}				
			}

			for (int m = 0; m < m_params->m; m++) {
				newDotProduct[m] = min(max(newDotProduct[m], 0.0f), clip);
				m_probAllVector[m] = (m_probAllVector[m] * (dataCounter - labelSize) +
					labelSize * newDotProduct[m]) / (float)dataCounter;

				for (int l = 0; l < labelSize; l++) {
					int k = labelVector[l];
					m_probSingleVector[m][k] = (m_probSingleVector[m][k] *
						(m_dataClassCounter[k] - 1) +
						newDotProduct[m]) / (float)m_dataClassCounter[k];
				}
			}
			

		}
	}

	const float weightTreshold = weightMag;

	for (int m = 0; m < m_params->m; m++) {
		for (int j = 0; j < d; j++) {
			if ((m_params->coefL1 != 0 || m_params->coefL2 != 0) && m_lastUpdate[j] > 0) {
				decayReg(m_weight[m][j], m_params->coefL1, m_params->coefL2, tStep - m_lastUpdate[j]);
			}
			if (fabs(m_weight[m][j]) < weightTreshold) {
				m_weight[m][j] = 0.f;
			}
		}
	}
}


void TreeNode::destroyChildren(vector<TreeNode>& nodes) {
    for (int m = 0; m < m_params->m; m++) {
        int ch = m_children[m];
        nodes[ch].m_dataIndex.clear();
    }
    m_children.clear();
}

int TreeNode::makeChildren(const DataLoader &trData, const DataLoader &trLabel, const int& N, vector<TreeNode>& nodes) {

    m_children.resize(m_params->m); // m_params->m children

	for (int m = 0; m < m_params->m; m++) {
        fill(m_children_labelHistogram[m].begin(), m_children_labelHistogram[m].end(), 0);
    }

    for (int m = 0; m < m_params->m; m++) {
        m_children[m] = N + m;
        int ch = m_children[m];
    }

    for (size_t index = 0; index < m_dataIndex.size(); index++) {
		vector<float> dotProduct(m_params->m);
		if (m_params->sparse) {
			DataPoint dataNormal = trData.getDataPoint(m_dataIndex[index]);
			for (int m = 0; m < m_params->m; m++) {
				dotProduct[m] = dataNormal.dot(m_weight[m]);
			}
		}
        else {
			DataPoint data = trData.getDataPoint(m_dataIndex[index]);
			DataPoint dataNormal = data.normal(m_mean, m_std);
			for (int m = 0; m < m_params->m; m++) {
				dotProduct[m] = dataNormal.dot(m_weight[m]);
			}
		}
        vector<int> childIndicator(m_params->m, 0);
        for (int m = 0; m < m_params->m; m++) {
            
			int ch = m_children[m];
            if (dotProduct[m] >= 0.5) {
                childIndicator[m] = 1;
                nodes[ch].m_dataIndex.push_back(m_dataIndex[index]);
                const vector<int>& labelVector = trLabel.getDataPoint(m_dataIndex[index]).getLabelVector();
                for (size_t l = 0; l < labelVector.size(); l++) {
                    int k = labelVector[l];
					m_children_labelHistogram[m][k]++;
                }
            }
        }
        int cr = 0;
        for (int m = 0; m < m_params->m; m++)
            cr += childIndicator[m] * (1 << m);

        if (cr == 0) { // none direction
            float maxDot = *max_element(dotProduct.begin(), dotProduct.end());
            for (int m = 0; m < m_params->m; m++) {
                int ch = m_children[m];
                if (dotProduct[m] == maxDot) {
                    nodes[ch].m_dataIndex.push_back(m_dataIndex[index]);
                    const vector<int>& labelVector = trLabel.getDataPoint(
                                                m_dataIndex[index]).getLabelVector();
                    for (size_t l = 0; l < labelVector.size(); l++) {
                        int k = labelVector[l];
						m_children_labelHistogram[m][k]++;
                    }
                }
            }
        }
    }

    bool indic = true;
    for (int m = 0; m < m_params->m; m++) {
        int ch = m_children[m];
        if (m_dataIndex.size() == nodes[ch].m_dataIndex.size() || nodes[ch].m_dataIndex.empty())
            indic = false;
    }

    m_dataIndex.clear();
    m_dataIndex.resize(1);
    m_dataIndex.shrink_to_fit();

    if (indic == false) {
        destroyChildren(nodes);
        return 0;
    } else {
        for (int m = 0; m < m_params->m; m++) {
            int ch = m_children[m];
            nodes[ch].m_nodeId = ch;
            nodes[ch].m_depth = m_depth + 1;
			nodes[ch].m_labelHistogramSparse.set(m_children_labelHistogram[m]);
			m_weightSparse.push_back(Varray<float>(m_weight[m]));
        }
    }

    return m_params->m;
}

void TreeNode::testBatch(const DataLoader &teData, vector<TreeNode>& nodes,
                         vector<vector<int>>& leafs) {

    if (isLeaf()) {
        for (size_t index = 0; index < m_dataIndex.size(); index++) {
            leafs[m_dataIndex[index]].push_back(m_nodeId);
        }
    } else {
        for (int m = 0; m < m_params->m; m++) {
            fill(m_weight[m].begin(), m_weight[m].end(), 0);
        }
        for (int m = 0; m < m_params->m; m++) {
            nodes[m_children[m]].m_dataIndex.clear();
            nodes[m_children[m]].m_dataIndex.reserve(m_dataIndex.size());
            for (size_t i = 0; i < m_weightSparse[m].myMap.index.size(); i++) {
                m_weight[m][m_weightSparse[m].myMap.index[i]] = m_weightSparse[m].myMap.value[i];
            }
        }

        for (size_t index = 0; index < m_dataIndex.size(); index++) {
            bool sent = false;
            int mMaxIdx = 0;
            float mMax = -1.0;
            for (int m = 0; m < m_params->m; m++) {
				float dotProduct;
				if (m_params->sparse) {
					const DataPoint& data = teData.getDataPoint(m_dataIndex[index]);
					dotProduct = data.dot(m_weight[m]);
				}
				else {
					DataPoint data = teData.getDataPoint(m_dataIndex[index]);
					DataPoint dataNormal = data.normal(m_mean, m_std);
					dotProduct = dataNormal.dot(m_weight[m]);
				}

                if (dotProduct > mMax) {
                    mMaxIdx = m;
                    mMax = dotProduct;
                }

                if (dotProduct >= 0.5) {
                    nodes[m_children[m]].m_dataIndex.push_back(m_dataIndex[index]);
                    sent = true;
                }
            }

            if (!sent) {
                nodes[m_children[mMaxIdx]].m_dataIndex.push_back(m_dataIndex[index]);
            }
        }
    }
	
    m_dataIndex.clear();
    m_dataIndex.resize(1);
    m_dataIndex.shrink_to_fit();
}

void TreeNode::addHistogram(labelEst& labelHistogramSum, int leafCount) const {
    for (size_t i = 0; i < m_NormalLabelHistogramSparse.size(); i++) {
        int idx = m_NormalLabelHistogramSparse[i].first;
        float val = m_NormalLabelHistogramSparse[i].second;
        labelHistogramSum.regular[idx] += (val / leafCount);
    }
}

void TreeNode::normalizeLabelHist() {
	int sumLabel = 0;
	size_t labelSize = m_labelHistogramSparse.myMap.value.size();
	for (size_t i = 0; i < labelSize; i++)
		sumLabel += m_labelHistogramSparse.myMap.value[i];

	for (size_t i = 0; i < m_labelHistogramSparse.myMap.index.size(); i++) {
		m_NormalLabelHistogramSparse.push_back(make_pair(m_labelHistogramSparse.myMap.index[i], 
			(float)m_labelHistogramSparse.myMap.value[i] / sumLabel));
	}
}