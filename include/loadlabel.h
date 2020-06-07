#pragma once
#include <vector>
#include <thread>
#include <mutex>

#include "varray.h"
#include "utils.h"
#include "dataloader.h"

using namespace std;

class LoadLabel {

private:

    vector<Varray<float>> labelVarrayVect;
    labelEstPairAll labelEstPairVect;
    vector<int> rootLabelHist;
    int nbOfClasses;

public:

    LoadLabel() {}
    
    LoadLabel(vector<Varray<float>> labels, vector<int> rootLabelHistogram, int k) {
        labelVarrayVect = labels;
        rootLabelHist = rootLabelHistogram;
        nbOfClasses = k;
    }

    static void threadTest(ifstream& fs, LoadLabel* loadLabel, labelEst& labelHistogramSum, std::mutex& labelMutex);

    void saveLabelEst(string strFile);

    labelEstPairAll loadLabelEst(string strFile, const DataLoader &teData, const
         vector<Varray<float>> &meanDataLabel, float beta, float gamma,
         vector<int>& rootLabelHistogram, int treeId, int nbThreads);

    Varray<float> getLabel(ifstream& fs);
    
};
