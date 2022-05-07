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
#include "tcgpusa.cuh"
#include "tccpu.h"

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
    bool *g;
    cudaMallocManaged(&g, N*N);
    
    //read input
    for(T i = 0; i < E; i++){
        T u,v;
        std::string w;
        INPUT >> u >> v >> w;
        g[N*u+v] = true;
    }
    //make graph undirected
    for(T i = 0; i < N; i++){
        for(T j = 0; j < N; j++){
            g[N*i+j] = g[N*i+j] | g[N*j+i];
        }
    }
    //remove self cycles
    for(T i = 0; i < N; i++){
        g[N*i+i] = 0;
    }
    

    //make sparse array for CPU
    std::vector<std::vector<int>> gsa;
    for(T i = 0; i < N; i++){
        std::vector<int> u;
        for(T j = 0; j < N; j++){
            if(g[N*i+j]) u.push_back(j);
        }
        gsa.push_back(u);
    }

    //PARATELIZED CPU IMPLEMENTATION
    T *sumcpu;
    cudaMallocManaged(&sumcpu, N*sizeof(T));
    auto cpustart = std::chrono::steady_clock::now();
    tccpu(N,sumcpu, gsa);
    T rescpu = 0;
    for(T i = 0; i < N; i++){
        rescpu += sumcpu[i];
    }
    rescpu /= 3;
    auto cpuend = std::chrono::steady_clock::now();
    std::chrono::duration<double> cputime = cpuend-cpustart;


    //GPU SA IMPLEMENTATION

    T *edges1;
    cudaMallocManaged(&edges1, E*sizeof(T));
    T *edges2;
    cudaMallocManaged(&edges2, E*sizeof(T));
    T *nodes;
    cudaMallocManaged(&nodes, N*sizeof(T));

    for(int i = 0; i < E; i++){
        edges1[i] = 0;
        edges2[i] = 0;
    }

    for(int i = 0; i < N; i++){
        nodes[i] = 0;
    }

    T idx = 0;
    for(T i = 0; i < N; i++){
        nodes[i] = idx;
        for(T j = 0; j < N; j++){
            if(g[N*i+j] && i!=j){
                edges1[idx] = i;
                edges2[idx] = j;
                idx++;
            }
        }
    }

    T *sumgpusa;
    cudaMallocManaged(&sumgpusa, E*sizeof(T));
    cudaEvent_t sastart, sastop;
    cudaEventCreate(&sastart);
    cudaEventCreate(&sastop);
    
    int threadsPerBlock = 32;
    int blocksPerGrid = (E + threadsPerBlock - 1)/ threadsPerBlock;

    //launch and time kernel
    cudaEventRecord(sastart);
    tcgpusa<<<blocksPerGrid, threadsPerBlock>>>(N, E, nodes, edges1, edges2, sumgpusa); 
    cudaEventRecord(sastop);

    cudaDeviceSynchronize();
    cudaEventSynchronize(sastop);
    T resgpusa = 0;
    for(T i = 0; i < E; i++){
        resgpusa += sumgpusa[i];
    }
    resgpusa /= 3;
    float sakerneltime = 0;
    cudaEventElapsedTime(&sakerneltime, sastart, sastop);


    /////////////////////////////////////////////////////////
    
    T *sumgpusa2;
    cudaMallocManaged(&sumgpusa2, E*sizeof(T));
    for(int i = 0; i < E; i++){
        sumgpusa2[i] = 0;
    }
    cudaEvent_t sastart2, sastop2;
    cudaEventCreate(&sastart2);
    cudaEventCreate(&sastop2);
    
    int threadsPerBlock2 = 32;
    int blocksPerGrid2 = (32*N + threadsPerBlock2 - 1)/ threadsPerBlock2;

    //launch and time kernel
    cudaEventRecord(sastart2);
    tcgpusa2<<<blocksPerGrid2, threadsPerBlock2>>>(N, E, nodes, edges1, edges2, sumgpusa2); 
    cudaEventRecord(sastop2);

    cudaDeviceSynchronize();
    cudaEventSynchronize(sastop2);
    T resgpusa2 = 0;
    for(T i = 0; i < E; i++){
        resgpusa2 += sumgpusa2[i];
    }
    resgpusa2 /= 3;
    float sakerneltime2 = 0;
    cudaEventElapsedTime(&sakerneltime2, sastart2, sastop2);
    /////////////////////////////////////////////////////////

    std::cout << "TRUE RES: " << rescpu << std::endl;
    std::cout << "GPU SA RES: " << resgpusa << std::endl;
    std::cout << "CPU TIME: " << cputime.count() << "s" << std::endl;
    std::cout << "GPU SA TIME: " << sakerneltime/1000.0 << "s" << std::endl;
    std::cout << "GPU speedup:  " << cputime.count()/(sakerneltime/1000.0) << "x" <<  std::endl; 
    std::cout << "GPU SA2 RES: " << resgpusa2 << std::endl;
    std::cout << "GPU SA2 TIME: " << sakerneltime2/1000.0 << "s" << std::endl;
    std::cout << "GPU2 speedup:  " << cputime.count()/(sakerneltime2/1000.0) << "x" <<  std::endl;


}

