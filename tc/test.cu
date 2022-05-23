#include <iostream>
#define T u_int64_t

int main(){
    T n = 1ULL << 34;
    bool *g;
    cudaMallocManaged(&g, n*sizeof(bool));
    for(T i = 0; i < n; i++){
        g[i] = 0;
    }
    
}