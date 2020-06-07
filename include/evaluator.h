#pragma once
#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <map>
#include <set>
#include <math.h>

#include "treenode.h"
#include "datapoint.h"
#include "utils.h"
#include "dataloader.h"
#include "tree.h"

using namespace std;

struct ScoreValue {
    int Ri;           
    float hamming;   
    float precision; 
    float recall;    
    float fscore;   
    float nDCG;      

    float psPrecision; // propensity score precision
    float psNDCG; 
    float psTruePrecision; 
    float psTrueNDCG; 
};

struct ScoreVector {
    vector<float> precisionVector; 
    vector<float> recallVector;    
    vector<float> nDCGVector;

    vector<float> psPrecisionVector;
	vector<float> psNDCGVector;
	vector<float> psTruePrecisionVector;
	vector<float> psTrueNDCGVector;
};

class Evaluator {

    vector<ScoreValue> m_scoreValue;
    vector<ScoreVector> m_scoreVector;

public:

    vector<ScoreValue> evaluate(const DataLoader &trData, const DataLoader &trLabel, const labelEstPairAll &labelEsimatePairAll,
        const vector<int> &rootLabelHist, const vector<int> &R);

};
