#include "loadlabel.h"

using namespace std;

void LoadLabel::threadTest(ifstream& fs, LoadLabel* loadLabel, labelEst& labelHistSum, std::mutex& labelMutex) {
    Varray<float> singleEst = loadLabel->getLabel(fs);
    for (size_t i = 0; i < singleEst.myMap.index.size(); i++) {
        std::lock_guard<std::mutex> lockGuard(labelMutex);
        labelHistSum.regular[singleEst.myMap.index[i]] += singleEst.myMap.value[i];
    }
}

void LoadLabel::saveLabelEst(string strFile) {
    ofstream fs;
    fs.open(strFile, ios::out | ios::binary);
    if (!fs.is_open()){
        cout << "Cannot open file " << strFile << endl;
    } else {
        int testSize = labelVarrayVect.size();
        fs.write(reinterpret_cast<const char*>(&testSize), sizeof(testSize));
        fs.write(reinterpret_cast<const char*>(&nbOfClasses), sizeof(nbOfClasses));
        for (int k = 0; k < nbOfClasses; k++) {
            fs.write(reinterpret_cast<const char*>(&rootLabelHist[k]), sizeof(rootLabelHist[k]));
        }
        for (int i = 0; i < testSize; i++) {
            int labelSize = labelVarrayVect[i].myMap.index.size();
            fs.write(reinterpret_cast<const char*>(&labelSize), sizeof(labelSize));
            for (int j = 0; j < labelSize; j++) {
                fs.write(reinterpret_cast<const char*>(&labelVarrayVect[i].myMap.index[j]),
                         sizeof(labelVarrayVect[i].myMap.index[j]));
                fs.write(reinterpret_cast<const char*>(&labelVarrayVect[i].myMap.value[j]),
                         sizeof(labelVarrayVect[i].myMap.value[j]));
            }
         }
    }
    fs.close();
}

labelEstPairAll LoadLabel::loadLabelEst(string strFile, const DataLoader &teData,
    const vector<Varray<float>> &meanDataLabel, float beta, float gamma,
    vector<int>& rootLabelHistogram, int treeId, int nbTrees) {
    vector<thread> testThrd(nbTrees);
    vector<ifstream> fs(nbTrees);
    int testSize = 0;
    for (int t = 0; t < nbTrees; t++) {
        fs[t].open(strFile+to_string(t+treeId), ios::in | ios::binary);
        fs[t].read(reinterpret_cast<char*>(&testSize), sizeof(testSize));
        fs[t].read(reinterpret_cast<char*>(&nbOfClasses), sizeof(nbOfClasses));
        rootLabelHistogram.resize(nbOfClasses);
        for (int k = 0; k < nbOfClasses; k++) {
            fs[t].read(reinterpret_cast<char*>(&rootLabelHistogram[k]), sizeof(rootLabelHistogram[k]));
        }
    }
    for (int i = 0; i < testSize; i++) {
        labelEst labelHistSum;
        labelHistSum.regular.resize(nbOfClasses, 0.f);
        std::mutex labelMutex;
        for (int t = 0; t < nbTrees; t++) {
            //testThrd[t] = thread(threadTest, ref(fs[t]), this, ref(labelHistSum), ref(labelMutex));
            Varray<float> singleEst = getLabel(fs[t]);
            for (size_t ii = 0; ii < singleEst.myMap.index.size(); ii++) {
                labelHistSum.regular[singleEst.myMap.index[ii]] += singleEst.myMap.value[ii];
            }
        }
        for (int k = 0; k < nbOfClasses; k++) {
            if (labelHistSum.regular[k] != 0.f) {
                labelHistSum.regular[k] /= nbTrees;
                if (beta != 1) {
                    labelHistSum.regular[k] = beta*labelHistSum.regular[k]
                    +(1-beta)/(1 + exp(gamma*teData.getDataPoint(i).dist(meanDataLabel[k])/2));
                }
            }
        }
        /*for (int t = 0; t < nbTrees; t++) {
            testThrd[t].join();
        }*/
        labelEstPair labelHistPair;
        for (int j = 0; j < nbOfClasses; j++) {
            labelHistPair.regular.push_back(make_pair(labelHistSum.regular[j], j));
        }
        sort(labelHistPair.regular.begin(), labelHistPair.regular.end(), std::greater<std::pair<float, int>>());
        int t = min(5, (int)labelHistPair.regular.size());
        vector<pair<float, int>> labelHistPairCopy(labelHistPair.regular.begin(), labelHistPair.regular.begin()+t);
        labelEstPairVect.regular.push_back(labelHistPairCopy);
    }
    for (int t = 0; t < nbTrees; t++) {
        fs[t].close();
    }
    return labelEstPairVect;
}

Varray<float> LoadLabel::getLabel(ifstream& fs) {
    Varray<float> labelVarray;
    int labelSize = 0;
    fs.read(reinterpret_cast<char*>(&labelSize), sizeof(labelSize));
    labelVarray.myMap.index.resize(labelSize);
    labelVarray.myMap.value.resize(labelSize);
    for (int j = 0; j < labelSize; j++) {
        fs.read(reinterpret_cast<char*>(&labelVarray.myMap.index[j]),
                sizeof(labelVarray.myMap.index[j]));
        fs.read(reinterpret_cast<char*>(&labelVarray.myMap.value[j]),
                sizeof(labelVarray.myMap.value[j]));
    }
    return labelVarray;
}
