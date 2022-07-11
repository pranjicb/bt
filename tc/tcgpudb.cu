#include "assert.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>

#define T u_int32_t

///////////////////////INPUT FILES//////////////////////////////////////
//BIO-CE-PG
std::string biocepg_path = "../../../datasets/biological/bio-SC-GT/bio-SC-GT.edges";
int N_BIOCEPG = 1716; 
int E_BIOCEPG = 33987; 

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

//BIO-MOUSE-GENE
std::string biomousegene_path = "../../../datasets/biological/bio-mouse-gene/bio-mouse-gene.edges";
int N_BIOMOUSEGENE = 45101;
int E_BIOMOUSEGENE = 14506196;
/////////////////////////////////////////////////////////////////////////


__device__ T intersectCount(int n, T *u, T *v){
    T count = 0ULL;
    for(int i = 0; i < n; i++){
        count += __popcll(u[i] & v[i]);
    }
    return count;
}

/*__global__ void tcgpudb(int n1, int n2, T *sum, T *g, int numbits){
    int u = blockIdx.x;
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;
    T u_neigh = u*n2;
    for(int i = 0; i < n2; i++){
        T tmp1 = g[u_neigh+i];
        T tmp2 = (1ULL << tid);
        T tmp3 = tmp1 & tmp2;
        T v = i*numbits+tid;
        T v_neigh = v*n2;
        if(tmp2 == tmp3){
            T count = intersectCount(n2, &g[u_neigh], &g[v_neigh]);
            sum[u*n1+v] += count;
        }
        
    }
}*/

__global__ void tcgpudb(int n1, int n2, T *sum, T *g){
    T tid = blockDim.x * blockIdx.x + threadIdx.x;
    T u = tid / 32;
    T lid = tid % 32;
    T u_neigh = n2*u;
    T count = 0;
    for(unsigned int i = 0; i < n2; i++){
        T tmp1 = g[u_neigh+i];
        T tmp2 = (1ULL << lid);
        T tmp3 = tmp1 & tmp2;
        T v = 32*i+lid;
        T v_neigh = n2*v;
        if(tmp2 == tmp3){
            for(unsigned int k = 0; k < n2; k++){
                count += __popcll(g[u_neigh+k] & g[v_neigh+k]);
            }
        }
    }
    for(int offset = 16; offset > 0; offset /= 2){
        count += __shfl_down_sync(0xffffffff, count, offset);
    }
    sum[u] = count;
}

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

/*void readGraph(int n, int e, std::string path, bool *g){

}*/


int numbits = sizeof(T) * 8;
unsigned int N = N_C500;
unsigned int E = E_C500;
std::ifstream INPUT(c500_path);

int main(){
    bool *g;
    cudaMallocManaged(&g, N*N);
    for(int i = 0; i < N*N; i++){
        g[i] = false;
    }
    
    //read input
    for(int i = 0; i < E; i++){
        int u,v;
        std::string w;
        INPUT >> u >> v;
        if(u < v){
            --u;
            --v;
            g[N*u+v] = true;
        }
        else if(v < u){
            --u;
            --v;
            g[N*v+u] = true;
        }
    }
    //remove self cycles
    for(T i = 0; i < N; i++){
        g[N*i+i] = 0;
    }
    T count = 0;
    for(T i = 0; i < N*N; i++){
        if(g[i]) count++;
    }
    E = count;

    int N2;
    if(N % numbits == 0){
        N2 = N / numbits;
    }
    else{
        N2 = (N + numbits - 1) / numbits;
    }

    T *g2;
    cudaMallocManaged(&g2, N*N2*sizeof(T));
    for(int i = 0; i < N*N2; i++){
        g2[i] = 0;
    }

    for(int i = 0; i < N; i++){
        int index = 0;
        for(int j = 0; j < N2; j++){
            std::string s = "";
            for(int k = 0; k < 32; k++){
                std::string s1 = "0";
                if(index < N){
                    if(g[N*i+index]){
                        s1 = "1";
                    }
                    index++;
                }
                s.append(s1);
            }
            std::reverse(s.begin(), s.end());
            g2[N2*i+j] = std::stoull(s,nullptr,2);
        }
    }

    T *sumdb;
    cudaMallocManaged(&sumdb, N*sizeof(T));
    for(int i = 0; i < N; i++){
        sumdb[i] = 0;
    }
    cudaEvent_t dbstart, dbend;
    cudaEventCreate(&dbstart);
    cudaEventCreate(&dbend);

    int threadsPerBlockdb = 128;
    int blocksPerGriddb = (32*N + threadsPerBlockdb - 1) / threadsPerBlockdb;

    cudaEventRecord(dbstart);
    tcgpudb<<<blocksPerGriddb, threadsPerBlockdb>>>(N,N2,sumdb,g2);
    cudaEventRecord(dbend);

    cudaDeviceSynchronize();
    cudaEventSynchronize(dbend);
    T resdb = 0;
    for(int i = 0; i < N; i++){
        resdb += sumdb[i];
    }
    float timedb = 0;
    cudaEventElapsedTime(&timedb, dbstart, dbend);


    


  
    std::cout << "GPU DB RES:   " << resdb << std::endl;
    std::cout << "GPU DB TIME:  " << timedb/1000.0 << "s" << std::endl;
    for(int i = 0; i < 500; i++){
        std::cout << g2[i] << std::endl;
    }
}