#include "graphreader.h"
#include <iostream>
#include <chrono>
#include <assert.h>
#include <inttypes.h>
#define T u_int64_t


void tccpu(unsigned int n, T *sum, std::vector<std::vector<unsigned int>> g){
    #pragma omp parallel num_threads(32)
    {   
        #pragma omp for
        for(T i = 0; i < n; i++){
            std::vector<unsigned int> u = g.at(i);
            for(int j : u){
                if(i < j){
                    std::vector<unsigned int> v = g.at(j);
                    std::vector<unsigned int> w;
                    std::set_intersection(u.begin(), u.end(), v.begin(), v.end(), back_inserter(w));
                    sum[i] += w.size();
                }
                
            }
        } 
    }

}



int main(){
    /*for(int i = 0; i < testGraphs.size()-1; i++){
        GraphReader graph = testGraphs.at(i);
        graph.read();
        
        double cputime = 0;
        for(int i = 0; i < 5; i++){
            T *sum;
            cudaMallocManaged(&sum, graph.N*sizeof(T));
            for(T j = 0; j < graph.N; j++){
                sum[j] = 0;
            }
            auto start = std::chrono::steady_clock::now();
            tccpu(graph.N, sum, graph.g);
            auto end = std::chrono::steady_clock::now();
            T cpures = 0;
            for(T j = 0; j < graph.N; j++){
                cpures += sum[j];
            }
            assert(cpures==graph.res);
            std::chrono::duration<double> time = end-start;
            cputime += time.count();
            cudaFree(sum);
        }
        std::cout << "Graph Name: " << graph.name << ", CPU time: " << cputime/5.0 << std::endl;
    }*/
    GraphReader graph = SINAWEIBO;
    graph.read();
    std::cout << "OK" << std::endl;
    double cputime = 0;
    T *sum;
    cudaMallocManaged(&sum, graph.N*sizeof(T));
    for(T i = 0; i < graph.N; i++){
        sum[i] = 0; 
    }
    auto start = std::chrono::steady_clock::now();
    tccpu(graph.N, sum, graph.g);
    auto end = std::chrono::steady_clock::now();
    T cpures = 0;
    for(T i = 0; i < graph.N; i++){
        cpures += sum[i];
    }
    assert(cpures==graph.res);
    std::chrono::duration<double> time = end-start;
    cputime = time.count();
    std::cout << "CPU TIME: " << cputime << std::endl;
}