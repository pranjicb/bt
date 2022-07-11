#include "assert.h"
#include "graphreader.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <set>
#include <algorithm>
#include <fstream>

#define T u_int32_t
//GRAPH FORMATS
int type0 = 0; //u v c, starts with 0
int type1 = 1; //u v c, start with 1
int type2 = 2; //u v, starts with 0
int type3 = 3; //u v, starts with 1

__global__ void tcgpudb(int n1, int n2, T *sum, T *g){
    u_int64_t tid = 1ULL * blockDim.x * blockIdx.x + threadIdx.x;
    u_int64_t u = tid / 32;
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

int main(){
    GraphReader graph = SCPWTK;
    graph.read();

    T *sum;
    cudaMallocManaged(&sum, graph.n1*sizeof(T));
    for(int i = 0; i < graph.n1; i++){
        sum[i] = 0;
    }
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    unsigned int threadsPerBlock = 128;
    unsigned int blocksPerGrid = (32*graph.n1 + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    tcgpudb<<<blocksPerGrid, threadsPerBlock>>>(graph.n1,graph.n2,sum,graph.g_dense);
    cudaEventRecord(end);

    cudaDeviceSynchronize();
    cudaEventSynchronize(end);
    u_int64_t res = 0;
    for(int i = 0; i < graph.n1; i++){
        res += sum[i];
    }
    float time = 0;
    cudaEventElapsedTime(&time, start, end);

    std::cout << "GPU DB RES:   " << res << std::endl;
    std::cout << "GPU DB TIME:  " << time/1000.0 << "s" << std::endl;
}