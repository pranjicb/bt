#ifndef GRAPHREADER_H
#define GRAPHREADER_H

#include <vector>
#include <set>
#include <algorithm>
#include <fstream>

//GRAPH FORMATS
//u v c, starts with 0 type = 0
//u v c, start with 1  type = 1
//u v, starts with 0   type = 2
//u v, starts with 1   type = 3


struct GraphReader{
    std::string name;
    unsigned int N;
    unsigned int E;
    unsigned int type;
    std::string filepath;
    std::vector<std::vector<unsigned int>> g;
    unsigned long long res;
    double time;
    GraphReader(std::string name, unsigned int N, unsigned int E, unsigned int type, std::string filepath, unsigned long long res, double time);
    void read();
    void readSparse();

    //DENSE IMPLEMENTATION
    void readDense();
    unsigned int *g_dense;
    unsigned int n1 = N;
    unsigned int n2 = 0;
    

    //METRICS
    unsigned int maxdeg = 0;
    unsigned long long n_sq = 1ULL * N * N;
    double density = 1.0*E/n_sq;
    unsigned long long degsum = 0;
    double avgdeg = 0.0;
    unsigned int median = 0;
    unsigned int mode = 0; 
    double mean_sigma = 0.0;
    double mode_sigma = 0.0;
    std::vector<unsigned int> degs = std::vector<unsigned int>(N,0);

    //SELECTED ALGORITHM
    //1 thread per edge = 0 ==> default case
    //1 warp node, merge = 1 ==>
    //1 warp per node, binary search = 2
    //1 warp per edge = 3
    //2 kernels hybrid approach = 4
    //Dense implementation = 5
    unsigned int alg = 0; //1 thread per edge by default
};

GraphReader::GraphReader(std::string name, unsigned int nodes, unsigned int edges, unsigned int t, std::string path, unsigned long long trueres, double cputime) : 
name(name), N(nodes), E(edges), type(t), filepath(path), res(trueres), time(cputime){}

void::GraphReader::read(){
    if(density >= 0.03){
        alg = 5;
        readDense();
    }
    else{
        readSparse();
    }
}

void GraphReader::readDense(){
    if(n1 % 32 == 0){
        n2 = n1 / 32;
    }
    else{
        n2 = (n1 + 31) / 32;
    }
    cudaMallocManaged(&g_dense, n1*n2*sizeof(unsigned int));
    for(unsigned int i = 0; i < n1*n2; i++){
        g_dense[i] = 0ULL;
    }
    std::ifstream INPUT(filepath);
    unsigned int u,v;
    std::string c;
    if(type == 0){
        while(INPUT >> u >> v >> c){
            if(u < v){
                unsigned int idx = v / 32;
                unsigned int shamt = v % 32;
                g_dense[n2*u+idx] |= (1ULL << shamt); 
            }
            else if(v < u){
                unsigned int idx = u / 32;
                unsigned int shamt = u % 32;
                g_dense[n2*v+idx] |= (1ULL << shamt);  
            }
        }
    }
    else if(type == 1){
        while(INPUT >> u >> v >> c){
            if(u < v){
                --u;
                --v;
                unsigned int idx = v / 32;
                unsigned int shamt = v % 32;
                g_dense[n2*u+idx] |= (1ULL << shamt); 
            }
            else if(v < u){
                --u;
                --v;
                unsigned int idx = u / 32;
                unsigned int shamt = u % 32;
                g_dense[n2*v+idx] |= (1ULL << shamt);   
            }
        }
    }
    else if(type == 2){
        while(INPUT >> u >> v){
            if(u < v){
                unsigned int idx = v / 32;
                unsigned int shamt = v % 32;
                g_dense[n2*u+idx] |= (1ULL << shamt); 
            }
            else if(v < u){
                unsigned int idx = u / 32;
                unsigned int shamt = u % 32;
                g_dense[n2*v+idx] |= (1ULL << shamt);  
            }
        }
    }
    else if(type == 3){
        while(INPUT >> u >> v){
            if(u < v){
                --u;
                --v;
                unsigned int idx = v / 32;
                unsigned int shamt = v % 32;
                g_dense[n2*u+idx] |= (1ULL << shamt); 
            }
            else if(v < u){
                --u;
                --v;
                unsigned int idx = u / 32;
                unsigned int shamt = u % 32;
                g_dense[n2*v+idx] |= (1ULL << shamt);   
            }
        }
    }
}

void GraphReader::readSparse(){
    for(unsigned int i = 0; i < N; i++){
        std::vector<unsigned int> u;
        g.push_back(u);
    }
    std::ifstream INPUT(filepath);
    unsigned int u,v;
    std::string c;
    unsigned int count = 0;
    if(type == 0){
        while(INPUT >> u >> v >> c){
            if(u < v){
                g.at(u).push_back(v);
                degs.at(u)++;
                degs.at(v)++;
            }
            else if(v < u){
                g.at(v).push_back(u);
                degs.at(u)++;
                degs.at(v)++;
            }
        }
        for(unsigned int i = 0; i < N; i++){
            std::set<unsigned int> s(g.at(i).begin(), g.at(i).end());
            g.at(i).assign(s.begin(), s.end());
            count += g.at(i).size();
            degsum += degs.at(i);
            if(maxdeg < degs.at(i)){
                maxdeg = degs.at(i);
            }
        }
        E = count;
    } 
    else if(type == 1){
        while(INPUT >> u >> v >> c){
            if(u < v){
                --u;
                --v;
                g.at(u).push_back(v);
                degs.at(u)++;
                degs.at(v)++;
            }
            else if(v < u){
                --u;
                --v;
                g.at(v).push_back(u);
                degs.at(u)++;
                degs.at(v)++;
            }
        }
        for(unsigned int i = 0; i < N; i++){
            std::set<unsigned int> s(g.at(i).begin(), g.at(i).end());
            g.at(i).assign(s.begin(), s.end());
            count += g.at(i).size();
            degsum += degs.at(i);
            if(maxdeg < degs.at(i)){
                maxdeg = degs.at(i);
            }
        }
        E = count;
    }
    else if(type == 2){
        while(INPUT >> u >> v){
            if(u < v){
                g.at(u).push_back(v);
                degs.at(u)++;
                degs.at(v)++;
            }
            else if(v < u){
                g.at(v).push_back(u);
                degs.at(u)++;
                degs.at(v)++;
            }
        }
        for(unsigned int i = 0; i < N; i++){
            std::set<unsigned int> s(g.at(i).begin(), g.at(i).end());
            g.at(i).assign(s.begin(), s.end());
            count += g.at(i).size();
            degsum += degs.at(i);
            if(maxdeg < degs.at(i)){
                maxdeg = degs.at(i);
            }
        }
        E = count;
    }
    else{
        while(INPUT >> u >> v){
            if(u < v){
                --u;
                --v;
                g.at(u).push_back(v);
                degs.at(u)++;
                degs.at(v)++;
            }
            else if(v < u){
                --u;
                --v;
                g.at(v).push_back(u);
                degs.at(u)++;
                degs.at(v)++;
            }
        }
        for(unsigned int i = 0; i < N; i++){
            std::set<unsigned int> s(g.at(i).begin(), g.at(i).end());
            g.at(i).assign(s.begin(), s.end());
            count += g.at(i).size();
            degsum += degs.at(i);
            if(maxdeg < degs.at(i)){
                maxdeg = degs.at(i);
            }
        }
        E = count;
    }

    //metrics
    //average degree
    avgdeg = 1.0*degsum/N;

    //mode
    std::vector<unsigned int> histogram(maxdeg+1,0);
    for(unsigned int i = 0; i < N; i++){
        histogram.at(degs.at(i))++;
    }
    mode = std::distance(histogram.begin(), std::max_element(histogram.begin(), histogram.end()));

    //median
    auto m = degs.begin() + degs.size()/2;
    std::nth_element(degs.begin(), m, degs.end());
    median = degs.at(degs.size()/2);

    //mean and mode variance
    for(unsigned int i = 0; i < N; i++){
        mean_sigma += abs(degs.at(i)-avgdeg);
        mode_sigma += abs((degs.at(i)-mode)*1.0);
    }
    mean_sigma /= N;
    mode_sigma /= N;


    //SELECTED ALGORITHM
    //1 thread per edge = 0 
    //1 warp node, merge = 1 
    //1 warp per node, binary search = 2
    //1 warp per edge = 3
    //2 kernels hybrid approach = 4
    if(mode < 32){
        alg = 0;
    }
    else{
        if(avgdeg >= median-5 && avgdeg <= median+5 && avgdeg >= mode-5 &&
           avgdeg <= mode+5 && median >= mode-5 && median <= mode+5)
        {
            alg = 1;
        }
    }
    
}

GraphReader BIOCEPG = GraphReader("BIO-CE-PG", 1871, 47754, 0, "../../../datasets/biological/bio-CE-PG/bio-CE-PG.edges", 784919, 0.00625852);
GraphReader BIODMCX = GraphReader("BIO-DM-CX", 4040, 76717, 0, "../../../datasets/biological/bio-DM-CX/bio-DM-CX.edges", 733893, 0.00477213);
GraphReader BIOHSLC = GraphReader("BIO-HS-LC", 4227, 39484, 0, "../../../datasets/biological/bio-HS-LC/bio-HS-LC.edges", 231634, 0.0020165);
GraphReader BIOSCGT = GraphReader("BIO-SC-GT", 1716, 33987, 0, "../../../datasets/biological/bio-SC-GT/bio-SC-GT.edges", 369047, 0.00204809);
GraphReader BIOSCHT = GraphReader("BIO-SC-HT", 2084, 63027, 0, "../../../datasets/biological/bio-SC-HT/bio-SC-HT.edges", 1397660, 0.00697327);
GraphReader BIOHUMANGENE2 = GraphReader("BIO-human-gene2", 14340, 9041364, 1, "../../../datasets/biological/bio-human-gene2/bio-human-gene2.edges", 4905433564, 4.50084);
GraphReader BIOMOUSEGENE = GraphReader("BIO-MOUSE-GENE", 45101, 14506196, 1, "../../../datasets/biological/bio-mouse-gene/bio-mouse-gene.edges", 3619097862, 7.54229);
GraphReader BIOWORMNETV3 = GraphReader("BIO-WormNet-v3", 2445, 78736, 2, "../../../datasets/biological/bio-WormNet-v3-benchmark/bio-WormNet-v3-benchmark.edges", 2015875, 0.00253452);
GraphReader BNFLY = GraphReader("BN-fly", 1781, 33641, 2, "../../../datasets/brain/bn-fly-drosophila_medulla_1/bn-fly-drosophila_medulla_1.edges", 16255, 0.000326714);
GraphReader BNMOUSE = GraphReader("BN-mouse", 213, 21807, 2, "../../../datasets/brain/bn-mouse_brain_1/bn-mouse_brain_1.edges", 622414, 0.000998777);
GraphReader C500 = GraphReader("C500", 500, 112332, 3, "../../../datasets/dimac/C500-9/C500-9.mtx", 15119852, 0.0121547);
GraphReader BEACXC = GraphReader("ECON-BEACXC", 497, 50409, 1, "../../../datasets/economic/econ-beacxc/econ-beacxc.mtx", 1831667, 0.00261246);
GraphReader BEAFLW = GraphReader("ECON-BEAFLW", 507, 53403, 1, "../../../datasets/economic/econ-beaflw/econ-beaflw.mtx", 2011146, 0.00271303);
GraphReader SCPWTK = GraphReader("SC-PWTK", 217891, 5653221, 3, "../../../datasets/scientific/sc-pwtk/sc-pwtk.mtx", 55981589, 0.083145);
GraphReader SINAWEIBO = GraphReader("SOC-SINAWEIBO", 58655849, 261321071, 3, "../../../datasets/social/soc-sinaweibo/soc-sinaweibo.mtx", 30518752, 160.583);

std::vector<GraphReader> testGraphs = {BIOCEPG, BIODMCX, BIOHSLC, BIOSCGT, BIOSCHT, BIOHUMANGENE2, BIOMOUSEGENE, BIOWORMNETV3, BNFLY, BNMOUSE, C500, BEACXC, BEAFLW, SCPWTK, SINAWEIBO};


#endif