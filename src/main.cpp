#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <string>
#include <cassert>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <future>
#include <chrono>

#include "loadlabel.h" 
#include "utils.h"
#include "datapoint.h" 
#include "treenode.h" 
#include "tree.h" 
#include "evaluator.h" 
#include "dataloader.h" 

#include "cxxopts.hpp" 

using namespace std;
using namespace std::chrono;

vector<Varray<float>> loadmu(string strFile);

int avgLabelPerData(const DataLoader& label, int nb) {
	int Size = max(label.size(), nb);
	int count = 0;
	for (int i = 0; i < Size; i++) {
		count += label.getDataPoint(i).size();
	}
	return (count / Size);
}

int main(int argc, char* argv[])
{

    int R1; // precision@R1; chooses R1 number of labels for each data point.
    int R2; // precision@R2
    int R3; // precision@R3
    string dataSetName;
    string dataSetPath;
    string saveLabel;
    string loadLabel;
    TreeParams params;
    int seed;
    int treeId;
    int nbTrees;
    bool loadOnly;

    try 
    {
        cxxopts::Options options(argv[0], "Multilabel Options");
        options.add_options()
            ("n,nmax", "maximum number of nodes", cxxopts::value<int>()->default_value("1000"))
            ("r1", "selecting r1 number of labels", cxxopts::value<int>()->default_value("1"))
            ("r2", "selecting r2 number of labels", cxxopts::value<int>()->default_value("3"))
            ("r3", "selecting r3 number of labels", cxxopts::value<int>()->default_value("5"))
            ("e,epochs", "number of epochs", cxxopts::value<int>()->default_value("1"))
            ("lr", "learning rate", cxxopts::value<float>()->default_value("0.1"))
            ("path", "path of data set", cxxopts::value<string>()->default_value("../../data/"))
            ("name", "name of data set", cxxopts::value<string>()->default_value("delicious"))
            ("savelabel", "file name for saving labels", cxxopts::value<string>()->default_value("./mylabel.dat"))
            ("loadlabel", "file name for loading labels", cxxopts::value<string>()->default_value("./mylabel.dat"))
            ("loadonly", "flag for loading the labels", cxxopts::value<bool>()->default_value("false"))
            ("mary", "arity of the tree", cxxopts::value<int>()->default_value("2"))
            ("l1", "lambda1: both term in the objective", cxxopts::value<float>()->default_value("1"))
            ("l2", "lambda2: purity term in the objective", cxxopts::value<float>()->default_value("2"))
            ("muFlag", "flag for saving mu (meanDataLabel) for tail label", cxxopts::value<bool>()->default_value("false"))
            ("gamma", "tail label parameter", cxxopts::value<float>()->default_value("0"))
            ("beta", "tail label portion - 1 = no contribution", cxxopts::value<float>()->default_value("1"))
            ("coefl1", "L1 regularizer coef- weight sparsity", cxxopts::value<float>()->default_value("0"))
            ("coefl2", "L2 regularizer coef - weight sparsity", cxxopts::value<float>()->default_value("0"))
            ("xx", "example learning flag", cxxopts::value<bool>()->default_value("false"))
            ("entropyLoss", "use entropy loss", cxxopts::value<bool>()->default_value("true"))
            ("sparse", "sparsity of data", cxxopts::value<bool>()->default_value("true"))
            ("seed", "random number generator seed", cxxopts::value<int>()->default_value("0"))
            ("treeid", "tree ID", cxxopts::value<int>()->default_value("0"))
            ("ens", "ensemble size", cxxopts::value<int>()->default_value("1"))
            ;
        auto result = options.parse(argc, argv);
        params.nMax = result["nmax"].as<int>();
        R1 = result["r1"].as<int>();
        R2 = result["r2"].as<int>();
        R3 = result["r3"].as<int>();
        params.epochs = result["epochs"].as<int>();
        params.alpha = result["lr"].as<float>();
        dataSetName = result["name"].as<string>();
        dataSetPath = result["path"].as<string>();
        saveLabel = result["savelabel"].as<string>();
        loadLabel = result["loadlabel"].as<string>();
        loadOnly = result["loadonly"].as<bool>();
        params.m = result["mary"].as<int>();
        params.l1 = result["l1"].as<float>();
        params.l2 = result["l2"].as<float>();
        params.gamma = result["gamma"].as<float>();
        params.beta = result["beta"].as<float>();
        params.coefL1 = result["coefl1"].as<float>();
        params.coefL2 = result["coefl2"].as<float>();
        params.exampleLearn = result["xx"].as<bool>();
        params.entropyLoss = result["entropyLoss"].as<bool>();
        params.sparse = result["sparse"].as<bool>();
        seed = result["seed"].as<int>();
        treeId = result["treeid"].as<int>();
        nbTrees = result["ens"].as<int>();
    }
    catch (const cxxopts::OptionException& e)
    {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }
    
	srand(seed);

    int treeSize = 0;
    int treeDepth = 0;
    float trTime = 0.f;
    float teTime = 0.f;
	float raTime = 0.f;
    float loadTime = 0.f;

	DataLoader teData(dataSetPath + dataSetName, false, false);
    DataLoader teLabel(dataSetPath + dataSetName, true, false);
	cerr << "Loaded test data." << endl;

    labelEstPairAll teLabelEstPair;
    vector<int> rootLabelHist;

    if (!loadOnly) {
        DataLoader trData(dataSetPath + dataSetName, false, true);
        DataLoader trLabel(dataSetPath + dataSetName, true, true);
		cerr << "Loaded training data." << endl;

        params.d = max(teData.getDim(), trData.getDim());
        params.k = trLabel.getNbOfClasses();

        Tree tree;
        tree.setParams(params);

        high_resolution_clock::time_point start_train = high_resolution_clock::now();
        tree.buildTree(trData, trLabel);
		tree.normalizeLableHist();
        high_resolution_clock::time_point end_train = high_resolution_clock::now();
        trTime = duration_cast<microseconds>(end_train - start_train).count() / 1000000.f;
        cerr << "Finished the training." << endl;

        vector<Varray<float>> teLabelEst;
		vector<Varray<float>> meanDataLabel;
		if (params.beta != 1)
			meanDataLabel = loadmu("../results/mu_" + dataSetName); // meanDataLabel = tree.getMeanDataLabel(); 
		
        high_resolution_clock::time_point start_test = high_resolution_clock::now();
        tree.testBatch(teData);
        for (int i = 0; i < teData.size(); i++) {
			teLabelEst.push_back(tree.testData(i, teData, meanDataLabel));
        }
        high_resolution_clock::time_point end_test = high_resolution_clock::now();
        teTime = duration_cast<microseconds>(end_test - start_test).count() / 1000000.f;
        cerr << "Finished testing test data." << endl;
        
        rootLabelHist = tree.getRootLabelHistogram();
        if (saveLabel != "") {
            LoadLabel labels = LoadLabel(teLabelEst, rootLabelHist, params.k);
            labels.saveLabelEst(saveLabel+to_string(treeId));
            cerr << "Finished saving the labels." << endl;
        }
		
        high_resolution_clock::time_point start_rank = high_resolution_clock::now();
        for (int i = 0; i < teData.size(); i++) {
            Varray<float> singleEst = teLabelEst[i];
            labelEstPair signleEstPair;
            for (size_t i = 0; i < singleEst.myMap.index.size(); i++) {
                signleEstPair.regular.push_back(make_pair(singleEst.myMap.value[i], singleEst.myMap.index[i]));
            }
            sort(signleEstPair.regular.begin(), signleEstPair.regular.end(), std::greater<std::pair<float, int>>());
            int t = min(5, (int)signleEstPair.regular.size());
            vector<pair<float, int>> singleEstPairCopy(signleEstPair.regular.begin(), signleEstPair.regular.begin()+t);
            teLabelEstPair.regular.push_back(singleEstPairCopy);
        }
		high_resolution_clock::time_point end_rank = high_resolution_clock::now();
		raTime = duration_cast<microseconds>(end_rank - start_rank).count() / 1000000.0;
        
        treeSize = tree.getTreeSize();
        treeDepth = tree.getTreeDepth();
        loadLabel = saveLabel;
    }


    LoadLabel labels = LoadLabel();
    vector<Varray<float>> meanDataLabel = loadmu("../results/mu_" + dataSetName);
    high_resolution_clock::time_point start_load = high_resolution_clock::now();
    teLabelEstPair.regular.clear();
    teLabelEstPair = labels.loadLabelEst(loadLabel, teData, meanDataLabel,
    params.beta, params.gamma, rootLabelHist, treeId, nbTrees);
    high_resolution_clock::time_point end_load = high_resolution_clock::now();
    loadTime = duration_cast<microseconds>(end_load - start_load).count() / 1000000.f;
    cerr << "Loaded labels." << endl;

    // evaluation
    Evaluator teEvaluateReg;
    vector<ScoreValue> teScoreValueReg;
    vector<int> R = { R1, R2, R3 };
    teScoreValueReg = teEvaluateReg.evaluate(teData, teLabel, teLabelEstPair, rootLabelHist, R);

    cout << dataSetName << ", nmax = "
         << params.nMax << ", lr = " << params.alpha << ", m = " << params.m << ", e = " << params.epochs 
         << ", l1 = " << params.l1 << ", l2 = " << params.l2 
         << ", beta = " << params.beta << ", gamma = " << params.gamma 
         << ", tree id = " << treeId << ", #trees = " << nbTrees << "\n"
         << "-----------------------------------------------------------\n"
         << "precision: " << teScoreValueReg[0].precision * 100 << ", "
         << teScoreValueReg[1].precision * 100 << ", "
         << teScoreValueReg[2].precision * 100 << "\n"
         << "nDCG: " << teScoreValueReg[0].nDCG * 100 << ", "
         << teScoreValueReg[1].nDCG * 100 << ", "
         << teScoreValueReg[2].nDCG * 100 << "\n"
	     << "PSprecision: " << teScoreValueReg[0].psPrecision * 100 << ", "
         << teScoreValueReg[1].psPrecision * 100 << ", "
         << teScoreValueReg[2].psPrecision * 100 << "\n"
         << "PSnDCG: " << teScoreValueReg[0].psNDCG * 100 << ", "
         << teScoreValueReg[1].psNDCG * 100 << ", "
         << teScoreValueReg[2].psNDCG * 100 << "\n"
		 << "tree size = " << treeSize << ", tree depth = " << treeDepth << "\n";
        
        if (!loadOnly) {
            cout << "training time = " << trTime << " s\n"
		    << "testing time = " << (teTime + raTime)*1000/teData.size() << " ms/point\n";
        }
    cout << endl;
    
    return 0;
}

