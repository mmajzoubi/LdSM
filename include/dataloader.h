#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <set>
#include <random>

#include "datapoint.h"
#include "utils.h"

using namespace std;

class DataLoader {

private:

    string m_dataSetPath;
    string m_dataSetName; 
    bool m_trainFlag; 
    bool m_labelFlag; 
    vector<DataPoint> m_dataPoints;
    int m_dim; // feature dimension: only for data
    int m_nbOfClasses; // number of classes: only for label

public:

    DataLoader(string dataSetPath, bool labelFlag, bool trainFlag);

    inline const DataPoint& getDataPoint(int j) const { return m_dataPoints[j]; }

    int size() const { return m_dataPoints.size(); }

    int getDim() const { return m_dim; }

    int getNbOfClasses() const { return m_nbOfClasses; }
    
    string getDataSetName() const { return m_dataSetName; }
};
