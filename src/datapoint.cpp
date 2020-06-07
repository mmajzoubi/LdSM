#include "datapoint.h"

using namespace std;

int DataPoint::size() const {
    if (m_labelFlag)
        return m_labelPoint.size();
    else
        return m_dataPointIndeces.size();
}

float DataPoint::dot(const vector<float> &weightVector) const {
    float dotProduct = 0.0;
    for (size_t i = 0; i < m_dataPointIndeces.size(); i++) {
        dotProduct += weightVector[m_dataPointIndeces[i]] * m_dataPointValues[i];
    }
    return dotProduct;
}

float DataPoint::dot(const Varray<float> &weightVector) const {
	float dotProduct = 0.f;
	size_t idx1 = 0;
	size_t idx2 = 0;
	while (idx1 < m_dataPointIndeces.size() && idx2 < weightVector.size()) {
		if (m_dataPointIndeces[idx1] < weightVector.myMap.index[idx2])
			idx1++;
		else if (m_dataPointIndeces[idx1] > weightVector.myMap.index[idx2])
			idx2++;
		else {
			dotProduct += m_dataPointValues[idx1] * weightVector.myMap.value[idx2];
			idx1++; idx2++;
		}			
	}	
	return dotProduct;
}

DataPoint DataPoint::normal(const vector<float> &mean, const vector<float> &std) { // only for non-sparse data
    int dim = m_dataPointIndeces.size();
    vector<int> indeces(dim);
    vector<float> values(dim);
    for (int i = 0; i < dim; i++) {
        indeces[i] = m_dataPointIndeces[i];
        values[i] = (m_dataPointValues[i] - mean[i]) / std[i];
    }
    DataPoint dataNormal(indeces, values);
    return dataNormal;
}

float DataPoint::dist(const Varray<float> &mu) const {
    float distance = 0.f;
	size_t i = 0;
	size_t j = 0;
    while (i < m_dataPointIndeces.size() || j < mu.myMap.index.size()) {
        int id = m_dataPointIndeces[i];
        float vald = m_dataPointValues[i];
        int jm = mu.myMap.index[j];
        float valm = mu.myMap.value[j];
        if (id == jm) {
            distance += (vald - valm)*(vald - valm);
            i++;
            j++;
        } else if (jm < id) {
            distance += valm*valm;
            j++;
        } else {
            distance += vald*vald;
            i++;
        }
    }
    return distance;
}

vector<float> DataPoint::sum(const vector<float> &x) const {
    vector<float> summation = x;
    for (size_t i = 0; i < m_dataPointIndeces.size(); i++) {
        summation[m_dataPointIndeces[i]] += m_dataPointValues[i];
    }
    return summation;
}
