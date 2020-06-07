#include "dataloader.h"

using namespace std;

DataLoader::DataLoader(string dataSetPath, bool labelFlag, bool trainFlag) {
    
    m_dataSetPath = dataSetPath;
    m_labelFlag = labelFlag;
    m_trainFlag = trainFlag;

    if (!m_labelFlag) {
	    string dataPointName;
	    string dataDimName;
	    string dataValueName;

	    if (m_trainFlag) {
		    dataPointName = m_dataSetPath + "_trDataPoint.csv";
		    dataDimName   = m_dataSetPath + "_trDataDim.csv";
		    dataValueName = m_dataSetPath + "_trDataValue.csv";
	    } else {
		    dataPointName = m_dataSetPath + "_teDataPoint.csv";
		    dataDimName   = m_dataSetPath + "_teDataDim.csv";
		    dataValueName = m_dataSetPath + "_teDataValue.csv";
	    }

	    ifstream dataPointFile(dataPointName);
	    ifstream dataDimFile(dataDimName);
	    ifstream dataValueFile(dataValueName);

	    vector<int> dataPoint;
	    vector<int> dataDim;
	    vector<float> dataValue;
	    for (int x; dataPointFile >> x;)
		dataPoint.push_back(x);

	    for (int x; dataDimFile >> x;)
		dataDim.push_back(x);

	    for (float x; dataValueFile >> x;)
		dataValue.push_back(x);

	    int nbOfPoint = dataPoint[dataPoint.size() - 1];

	    m_dim = *max_element(dataDim.begin(), dataDim.end()) + 1;
	    size_t iter = 0;
	    for (int i = 1; i <= nbOfPoint; i++)  {
		    vector<int> tmpIndeces; 
		    vector<float> tmpValues; 

		    bool missingLabel = true;
		    while (iter < dataPoint.size() && dataPoint[iter] == i) {
		        missingLabel = false;
                tmpIndeces.push_back(dataDim[iter] - 1);
                tmpValues.push_back(dataValue[iter]);
                iter++;
		    }

		    if (!missingLabel) {
                tmpIndeces.push_back(m_dim - 1);
                tmpValues.push_back(1);
	            DataPoint rowData(tmpIndeces, tmpValues);
	            m_dataPoints.push_back(rowData);
		    }
	    }
    } else { // label
        string labelDimName;
        string labelPointName;
        if (m_trainFlag == true) { // train
            labelPointName = m_dataSetPath + "_trLabelPoint.csv";;
            labelDimName = m_dataSetPath + "_trLabelDim.csv";
        } else { // test
            labelPointName = m_dataSetPath + "_teLabelPoint.csv";;
            labelDimName = m_dataSetPath + "_teLabelDim.csv";
        }
        
        ifstream labelPointFile(labelPointName);
        ifstream labelDimFile(labelDimName);

        vector<int> labelPoint;
        vector<int> labelDim;
        for (int x; labelPointFile >> x;)
            labelPoint.push_back(x);

        for (int x; labelDimFile >> x;)
            labelDim.push_back(x);

        m_nbOfClasses = *max_element(labelDim.begin(), labelDim.end());

        int nbOfPoint = labelPoint[labelPoint.size() - 1];

		size_t iter = 0;
        for (int i = 1; i <= nbOfPoint; i++)  {
            vector<int> tmp;

            bool missingLabel = true;
            while (iter < labelPoint.size() && labelPoint[iter] == i) {
                missingLabel = false;
                tmp.push_back(labelDim[iter] - 1);
                iter++;
            }

            if (!missingLabel) {
                DataPoint rowData(tmp);
                m_dataPoints.push_back(rowData);
            }
        } 
    }
}
