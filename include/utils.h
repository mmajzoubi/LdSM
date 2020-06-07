#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <cassert>
#include <map>

#include "dataloader.h"

using namespace std;

struct labelEst {
    vector<float> regular;
};

struct labelEstPair {
    vector<pair<float, int>> regular;
};

struct labelEstPairAll {
    vector<vector<pair<float, int>>> regular;
};
