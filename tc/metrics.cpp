#include "graphreader.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>

int main(){
    GraphReader graph = BNMOUSE;
    std::string filename = graph.name + ".csv";
    std::ofstream out(filename);
    std::streambuf *coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(out.rdbuf());
    std::ifstream INPUT(graph.filepath);    
    std::vector<unsigned int> degs(graph.N, 0);
    unsigned int N = graph.N;
    unsigned int E = graph.E;
    unsigned int maxdeg = 0;
    unsigned long long n2 = 1ULL * N * N;
    double density = n2/(1.0*E);
    double neRatio = N/(1.0*E);
    double enRatio = E/(1.0*N);
    unsigned long long degsum = 0;
    double avgdeg;
    
    unsigned int u,v;
    std::string w;
    while(INPUT >> u >> v){
        if(u < v){
            degs.at(u)++;
            degs.at(v)++;
        }
        else if(v < u){
            degs.at(u)++;
            degs.at(v)++;
        }
    }
    for(unsigned int i = 0; i < N; i++){
        degsum += degs.at(i);
        if(maxdeg < degs.at(i)){
            maxdeg = degs.at(i);
        }
    }

    avgdeg = degsum*1.0/N;

    std::cout << N << "," << E << "," << maxdeg << "," << avgdeg << "," << density << "," << neRatio << "," << enRatio << "," << std::endl;
    for(unsigned int j = 0; j < N; j++){
        std::cout << j << "," << degs.at(j) << std::endl;
    }
}
