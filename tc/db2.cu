#include "assert.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>

#define T unsigned int

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

__device__ T intersectCount(int n, T *u, T *v){
    T count = 0U;
    for(int i = 0; i < n; i++){
        count += __popc(u[i] & v[i]);
    }
    return count;
}

//CUDA kernel for triangle counting using dense bitvectors
__global__ void tcdbgpu2(int n1, int n2, T *sum, T *g){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < n1){
        T count = 0U;
        T rowU = n2*tid;
        for(int i = 0; i < n2; i++){
            T u = g[rowU+i];
            if(u > 0U){
                for(int j = 0; j < 32; j++){
                    T tmp = u & (1<<j);
                    if(tmp > 0U){
                        T rowV = 32*i+j;
                        count += intersectCount(n2, &g[rowU], &g[rowV]);
                        
                    }

                }
            }
        }
        sum[tid] = count;
    }
}

int numbits = sizeof(T) * 8;
unsigned int N = N_BIOSCGT;
unsigned int E = E_BIOSCGT;
std::ifstream INPUT(bioscgt_path);

int main(){
    bool *g;
    cudaMallocManaged(&g, N*N);
    //read input
    for(int i = 0; i < E; i++){
        unsigned int u,v;
        std::string w;
        INPUT >> u >> v >> w;
        g[N*u+v] = true;
    }
    //make graph undirected
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            g[N*i+j] = g[N*i+j] | g[N*j+i];
        }
    }
    //remove self cycles
    for(int i = 0; i < N; i++){
        g[N*i+i] = 0;
    }


    /////////////////////////////////////////////////////////////////////////
    
    T *g2;
    T g2N;
    if(N % numbits != 0){
        g2N = N/numbits + numbits;
    }
    else{
        g2N = N/numbits;
    }
    cudaMallocManaged(&g2, N*g2N);
    
    int g2index = 0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < g2N; j++){
            std::string s = "";
            for(int k = 31; k >= 0; k--){
                std::string s1 = "0";
                if(g2index < N*N){
                    if(g[g2index]){
                        s1 = "1";
                    }
                    g2index++;
                }
                s.append(s1);
            }
            g2[g2N*i + j] = std::stoull(s, nullptr, 2);
        }
    }
    
    T *sum2;
    cudaMallocManaged(&sum2, N);
    for(int i = 0; i < N; i++){
        sum2[i] = 0;
    }
    cudaEvent_t db2start, db2end;
    cudaEventCreate(&db2start);
    cudaEventCreate(&db2end);

    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1)/ threadsPerBlock;

    cudaEventRecord(db2start);
    tcdbgpu2<<<blocksPerGrid, threadsPerBlock>>>(N,g2N,sum2,g2);
    cudaEventRecord(db2end);

    cudaDeviceSynchronize();
    cudaEventSynchronize(db2end);
    T res2 = 0;
    for(int i = 0; i < N; i++){
        res2 += sum2[i];
    }
    res2 /= 3;
    float time2 = 0;
    cudaEventElapsedTime(&time2, db2start, db2end);
    
    ///////////////////////////////////////////////////////////////////


    
    std::cout << "GPU RES: " << res2 << std::endl;
    std::cout << "GPU TIME: " << time2/1000.0 << "s" << std::endl;
    


    cudaFree(g);

}