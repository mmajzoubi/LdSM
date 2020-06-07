#include "evaluator.h"

using namespace std;

vector<ScoreValue> Evaluator::evaluate(const DataLoader &trData, const DataLoader &trLabel, 
                                       const labelEstPairAll &labelEsimatePairAll,
                                       const vector<int> &rootLabelHist, const vector<int> &R) {
    
    // for propensity scores: TODO: names should be changed accordingly
    string dataSetName = trData.getDataSetName();
    float A, B, C;
    if (dataSetName == "amazon" || dataSetName == "amazon3m") {
        A = 0.6f;
        B = 2.6f;
    } else if (dataSetName == "wikiLSHTC") {
        A = 0.5f;
        B = 0.4f;
    } else {
        A = 0.55f;
        B = 1.5f;
    }
    C = (float)(log(trData.size()) - 1.f)*pow((B + 1.f), A);
    vector<float> pScore(rootLabelHist.size());
    for (size_t k = 0; k < rootLabelHist.size(); k++) {
        pScore[k] = 1.f / (1.f + C*pow((rootLabelHist[k]+B), -A));
    }
    
    // for calculating the scores for 3 values of R1, R2, R3
    m_scoreVector.resize(R.size()); 
    m_scoreValue.resize(R.size());
    for (size_t r = 0; r < R.size(); r++) {
        m_scoreValue[r].hamming = 0;
        m_scoreVector[r].precisionVector.reserve(trData.size());
        m_scoreVector[r].recallVector.reserve(trData.size());
        m_scoreVector[r].nDCGVector.reserve(trData.size());
        
        m_scoreVector[r].psPrecisionVector.reserve(trData.size());
        m_scoreVector[r].psNDCGVector.reserve(trData.size());
        m_scoreVector[r].psTruePrecisionVector.reserve(trData.size());
        m_scoreVector[r].psTrueNDCGVector.reserve(trData.size());
    }

    for (int i = 0; i < trData.size(); i++) {
        
        vector<pair<float, int> > labelEsimatePairRegular = labelEsimatePairAll.regular[i];
        const vector<int>& labelTrue = trLabel.getDataPoint(i).getLabelVector();
        
        vector<pair<float, int> > labelTruePair;
        for (size_t j = 0; j < labelTrue.size(); j++)
        {
            int k = labelTrue[j];
            labelTruePair.push_back(make_pair(1.f/pScore[k], k));
        }
        sort(labelTruePair.begin(), labelTruePair.end(), std::greater<std::pair<float, int>>());

        for (size_t r = 0; r < R.size(); r++) {
            m_scoreValue[r].Ri = R[r];
            float tmp = 0;
            float dcg = 0;
            float psTmp = 0;
            float psDCG = 0;
            float psTrueTmp = 0;
            float psTrueDCG = 0;
            int ri = min((int)(labelEsimatePairRegular.size()), R[r]);
            for (int j = 0; j < ri; j++) {
                int l = j + 1;
                if (find(labelTrue.begin(), labelTrue.end(), 
                         get<1>(labelEsimatePairRegular[j])) != labelTrue.end()) {
                    int k = get<1>(labelEsimatePairRegular[j]);
                    tmp++;
                    dcg += 1.f / log2(l + 1.f);
                    psTmp += 1.f / pScore[k];
                    psDCG += 1.f / (pScore[k]*log2(l + 1.f));
                }
            }
            
            ri = min((int)(labelTrue.size()), R[r]);
            for (int j = 0; j < ri; j++) {
                int l = j + 1;
                int k = get<1>(labelTruePair[j]);
                psTrueTmp += 1.f / pScore[k];
                psTrueDCG += 1.f / (pScore[k] * log2(l + 1.f));
            }
            
            float idcg = 0;
            int lmax = min(R[r], (int)labelTrue.size());
            for (int l = 1; l <= lmax; l++)
                idcg += 1.f / log2(l + 1.f);

            m_scoreVector[r].precisionVector.push_back(tmp / R[r]); 
            m_scoreVector[r].recallVector.push_back(tmp / labelTrue.size()); 
            m_scoreVector[r].nDCGVector.push_back(dcg / idcg); 
            m_scoreValue[r].hamming += (labelTrue.size() - tmp ); 
            
            m_scoreVector[r].psPrecisionVector.push_back(psTmp / R[r]); 
            m_scoreVector[r].psNDCGVector.push_back(psDCG / idcg); 

            m_scoreVector[r].psTruePrecisionVector.push_back(psTrueTmp / R[r]); 
            m_scoreVector[r].psTrueNDCGVector.push_back(psTrueDCG / idcg); 
        }
    }

    for (size_t r = 0; r < R.size(); r++) {
        m_scoreValue[r].hamming /= trData.size();
        m_scoreValue[r].precision = 0;
        m_scoreValue[r].recall = 0;
        m_scoreValue[r].nDCG = 0;
        
        m_scoreValue[r].psPrecision = 0;
        m_scoreValue[r].psNDCG = 0;
        m_scoreValue[r].psTruePrecision = 0;
        m_scoreValue[r].psTrueNDCG = 0;


        for (size_t i = 0; i < m_scoreVector[r].precisionVector.size(); i++) {
            m_scoreValue[r].precision += m_scoreVector[r].precisionVector[i];
            m_scoreValue[r].recall += m_scoreVector[r].recallVector[i];
            m_scoreValue[r].nDCG += m_scoreVector[r].nDCGVector[i];
            
            m_scoreValue[r].psPrecision += m_scoreVector[r].psPrecisionVector[i];
            m_scoreValue[r].psNDCG += m_scoreVector[r].psNDCGVector[i];
            m_scoreValue[r].psTruePrecision += m_scoreVector[r].psTruePrecisionVector[i];
            m_scoreValue[r].psTrueNDCG += m_scoreVector[r].psTrueNDCGVector[i];
        }

        m_scoreValue[r].precision /= m_scoreVector[r].precisionVector.size();
        m_scoreValue[r].recall /= m_scoreVector[r].precisionVector.size();
        m_scoreValue[r].nDCG /= m_scoreVector[r].precisionVector.size();
        m_scoreValue[r].fscore = 2 * m_scoreValue[r].precision*m_scoreValue[r].recall / 
                                 (m_scoreValue[r].precision + m_scoreValue[r].recall);
        
        m_scoreValue[r].psPrecision /= m_scoreValue[r].psTruePrecision;
        m_scoreValue[r].psNDCG /= m_scoreValue[r].psTrueNDCG;
    }

    return m_scoreValue;
}

