#pragma once

#include <iostream>
#include <vector>
#include "varray.h"

using namespace std;

class DataPoint {

private:

    vector<int> m_dataPointIndeces{};
    vector<float> m_dataPointValues{};
    vector<int> m_labelPoint{};
    bool m_labelFlag = false;

public:

    DataPoint() {}

    DataPoint(const vector<int>& indeces, const vector<float>& values) {
        m_dataPointIndeces = indeces;
        m_dataPointValues = values;
        m_labelFlag = false;
    }

    DataPoint(const vector<int>& x) {
        m_labelPoint = x;
        m_labelFlag = true;
    }

    DataPoint(const DataPoint &dpIn)
    {
        m_dataPointIndeces = dpIn.m_dataPointIndeces;
        m_dataPointValues  = dpIn.m_dataPointValues;
        m_labelPoint       = dpIn.m_labelPoint;
        m_labelFlag        = dpIn.m_labelFlag;
    }

    float dot(const vector<float> &) const;

    float dot(const Varray<float> &) const;

    float dist(const Varray<float> &) const;

    vector<float> sum(const vector<float> &) const;

    DataPoint normal(const vector<float> &mean, const vector<float> &std);

    int size() const;

    inline const vector<int>& getDataIndeces() const { return m_dataPointIndeces; }

    inline const vector<float>& getDataValues() const { return m_dataPointValues; }
    
    inline const vector<int>& getLabelVector() const { return m_labelPoint; }
};
