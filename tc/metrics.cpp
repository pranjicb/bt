#include "graphreader.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>

int main(){

#pragma omp parallel num_threads(16)
{

    std::ofstream out("metrics.txt");
    std::streambuf *coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(out.rdbuf());
    std::cout << "Name, nodes, edges, density, max degree, avg degree, mode, median, sigma, mode sigma, sigma2, mode sigma2" << std::endl;
    #pragma omp for
    for(unsigned int it = 0; it < testGraphs.size(); it++){
        GraphReader graph = testGraphs.at(it);
        
        std::ifstream INPUT(graph.filepath);    
        std::vector<unsigned int> degs(graph.N, 0);
        unsigned int N = graph.N;
        unsigned int E = graph.E;
        unsigned int maxdeg = 0;
        unsigned long long n2 = 1ULL * N * N;
        double density = 1.0*E/n2;
        unsigned long long degsum = 0;
        double avgdeg = 0.0;
        unsigned int median = 0;
        unsigned int modalwert = 0;
    
        unsigned int u,v;
        std::string w;

        if(graph.type==0){
            while(INPUT >> u >> v >> w){
                if(u < v){
                    degs.at(u)++;
                    degs.at(v)++;
                }
                if(v < u){
                    degs.at(u)++;
                    degs.at(v)++;
                }
            }
        }
        else if(graph.type==1){
            while(INPUT >> u >> v >> w){
                if(u < v){
                    --u;
                    --v;
                    degs.at(u)++;
                    degs.at(v)++;
                }
                if(v < u){
                    --u;
                    --v;
                    degs.at(u)++;
                    degs.at(v)++;
                }
            }
        }
        else if(graph.type==2){
            while(INPUT >> u >> v){
                if(u < v){
                    degs.at(u)++;
                    degs.at(v)++;
                }
                if(v < u){
                    degs.at(u)++;
                    degs.at(v)++;
                }
            }
        }
        else{
            while(INPUT >> u >> v){
                if(u < v){
                    --u;
                    --v;
                    degs.at(u)++;
                    degs.at(v)++;
                }
                if(v < u){
                    --u;
                    --v;
                    degs.at(u)++;
                    degs.at(v)++;
                }
            }
        }
        
        for(unsigned int i = 0; i < N; i++){
            degsum += degs.at(i);
            if(maxdeg < degs.at(i)){
                maxdeg = degs.at(i);
            }
        }
        avgdeg = degsum*1.0/N;
        std::vector<unsigned int> histogram(maxdeg+1, 0);
        for(unsigned int i = 0; i < graph.N; i++){
            unsigned int tmp = degs.at(i);
            histogram.at(tmp)++;
        }
        
        modalwert = std::distance(histogram.begin(), std::max_element(histogram.begin(), histogram.end()));

        auto m = degs.begin() + degs.size()/2;
        std::nth_element(degs.begin(), m, degs.end());
        median = degs.at(degs.size()/2);

        double sigma = 0.0;
        double mode_sigma = 0.0;
        double sigma2 = 0.0;
        double mode_sigma2 = 0.0;
        for(unsigned int i = 0; i < N; i++){
            sigma += abs(degs.at(i)-avgdeg);
            mode_sigma += abs(degs.at(i)-modalwert);
            sigma2 += (degs.at(i) - avgdeg)*(degs.at(i) - avgdeg);
            mode_sigma2 += (degs.at(i) - modalwert) * (degs.at(i) - modalwert);
        }

        sigma /= N;
        mode_sigma /= N;
        sigma2 /= N;
        mode_sigma2 /= N;
        

        std::cout << graph.name << "," << N << "," << E << "," << density << "," << maxdeg << "," << avgdeg << "," 
        << modalwert << "," << median << "," << sigma << "," << mode_sigma << ", " << sigma2 << "," << mode_sigma2 << std::endl;
    }

}
}
