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

int main(){
    GraphReader graph = C500;
    graph.read();
}