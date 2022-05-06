#include "assert.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <fstream>
#include <thrust/set_operations.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

#define T unsigned long long int


///////////////////////INPUT FILES//////////////////////////////////////
//BIO-SC-GT
std::string bioscgt_path = "../../../datasets/biological/bio-SC-GT/bio-SC-GT.edges";
int N_BIOSCGT = 1716; 
int E_BIOSCGT = 33987; 

//BIO-HUMAN-GENE2
std::string biohumangene2_path = "../../../datasets/biological/bio-human-gene2/bio-human-gene2.edges";
int N_BIOHUMANGENE2 = 14340; 
int E_BIOHUMANGENE2 = 9041364; 

//C500-9
std::string c500_path = "../../../datasets/dimac/C500-9/C500-9.mtx";
int N_C500 = 500;
int E_C500 = 112332;

//SC-PWTK
std::string scpwtk_path = "../../../datasets/scientific/sc-pwtk/sc-pwtk.mtx";
int N_SCPWTK = 217891;
int E_SCPWTK = 5653221;
/////////////////////////////////////////////////////////////////////////




T N = N_BIOSCGT;
T E = E_BIOSCGT;
std::ifstream INPUT(bioscgt_path);
int main(){
    std::vector<std::vector<T>> g;
    for(int i = 0; i < N; i++){
        std::vector<T> v;
        g.push_back(v);
    }
    thrust::device_vector<T> edges1(E,0);
    thrust::device_vector<T> edges2(E,0);
    thrust::device_vector<T> res(E);
    thrust::device_vector<T> nodes(N,0);
    for(T i = 0; i < E; i++){
        T u,v;
        std::string w;
        INPUT >> u >> v >> w;
        g.at(u).push_back(v);
        g.at(v).push_back(u);
    }
    int index = 0;
    for(int i = 0; i < N; i++){
        int size = g.at(i).size();
        nodes[i] = size;
        for(int j = 0; j < size; j++){
            if(i < j){
                edges1[index] = i;
                edges2[index] = g.at(i).at(j);
                index++;
            }
            
        }
    }
    thrust::set_intersection(edges1.begin(), edges1.end(), edges2.begin(), edges2.end(), res.begin());
    std::cout << res.size() << std::endl;

}

