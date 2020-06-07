#pragma once
#include <vector>

using namespace std;

template <class T> 
class MyMap {

public:

    vector<int> index;
    vector<T> value;

};


template <class T> 
class Varray {

public:
    
    MyMap<T> myMap;
    
    Varray() {}
    
    Varray(const vector<T>& vect) {
        int len = vect.size();
        for (int i = 0; i < len; i++) {
            if (vect[i] != static_cast<T>(0)) {
                myMap.index.push_back(i);
                myMap.value.push_back(vect[i]);
            }
        }    
    }

	Varray(const vector<int>& index, const vector<T>& value) {
		myMap.index = index;
		myMap.value = value;
	}
  
    void set(const vector<T>& vect) {
        int len = vect.size();
        for (int i = 0; i < len; i++) {
            if (vect[i] != static_cast<T>(0)) {
                myMap.index.push_back(i);
                myMap.value.push_back(vect[i]);
            }
        }
    }

	size_t size() const {
		return myMap.index.size();
	}

    T getValue(int idx, int& start) const { 
        int begin = start;
        int end = myMap.index.size();
        int i = (begin + end) / 2;
        
        while ((end - begin) > 1) {
            if (myMap.index[i] <= idx)
                begin = i;
            else 
                end = i;
                   
            i = (begin + end) / 2;        
        }
                
        if (myMap.index[begin] != idx)
            return 0;
        
        start = begin;
        return myMap.value[begin];
    }
};
