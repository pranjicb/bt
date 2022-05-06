#include "assert.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <fstream>

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

void tccpu(int n, T *sum, std::vector<std::vector<int>> g){
    #pragma omp parallel num_threads(32)
    {   
        #pragma omp for
        for(int i = 0; i < n; i++){
            std::vector<int> u = g.at(i);
            for(int j : u){
                if(i < j){
                    std::vector<int> v = g.at(j);
                    std::vector<int> w;
                    std::set_intersection(u.begin(), u.end(), v.begin(), v.end(), back_inserter(w));
                    sum[i] += w.size();
                }
                
            }
        } 
    }

}

//CUDA kernel for triangle counting using dense bitvectors
__global__ void tcdbgpu(T n, T *sum, bool *g){
    T u = blockIdx.x * blockDim.x + threadIdx.x; //each thread calculates neighbor intersection for one node
    if(u < n){
        bool *u1 = &g[n*u];
        for(int v = 0; v < u; v++){ //for all vertices in g
            if(g[n*u+v]){           //if v is a neighbor of u
                bool *v1 = &g[n*v];
                for(int i = 0; i < n-7; i+=8){
                    T tmp1,tmp2;
                    memcpy(&tmp1, &u1[i], 8);
                    memcpy(&tmp2, &v1[i], 8);
                    sum[u] += __popcll(tmp1&tmp2);
                }
            }
        }
    }
}


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

    //PARALLELIZED CPU IMPLEMENTATION
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


    //GPU IMPLEMENTATION BASED ON DENSE BITVECTORS
    T *sumgpudb;
    cudaMallocManaged(&sumgpudb, N*sizeof(T));
    cudaEvent_t dbstart, dbstop;
    cudaEventCreate(&dbstart);
    cudaEventCreate(&dbstop);
    
    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1)/ threadsPerBlock;

    //launch and time kernel
    cudaEventRecord(dbstart);
    tcdbgpu<<<blocksPerGrid, threadsPerBlock>>>(N, sumgpudb, g); 
    cudaEventRecord(dbstop);

    cudaDeviceSynchronize();
    cudaEventSynchronize(dbstop);
    T resgpudb = 0;
    for(T i = 0; i < N; i++){
        resgpudb += sumgpudb[i];
    }
    resgpudb /= 3;
    float dbkerneltime = 0;
    cudaEventElapsedTime(&dbkerneltime, dbstart, dbstop);

    std::cout << "TRUE RES: " << rescpu << std::endl;
    std::cout << "GPU RES: " << resgpudb << std::endl;
    std::cout << "CPU TIME: " << cputime.count() << "s" << std::endl;
    std::cout << "GPU TIME: " << dbkerneltime/1000.0 << "s" << std::endl;
    cudaFree(g);
    cudaFree(sumcpu);
    cudaFree(sumgpudb);
}