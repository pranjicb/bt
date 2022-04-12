#include <iostream>
#include <stdlib.h>
#include <assert.h>

//set intersection on GPU
__global__ void setint(int n, bool *x, bool *y){
    for(int i = 0; i < n; i++){
        x[i] = x[i] & y[i];
    }
}

//triangle counting algorithm on GPU
int tc(int n, bool *g){
    int count = 0;
    for(int u = 0; u < n; u++){
        bool *u1; 
        cudaMallocManaged(&u1, n);
        memcpy(u1, &g[n*u], n);
        for(int j = 0; j < u; j++){
            if(g[n*u+j]){
                bool *v;
                cudaMallocManaged(&v, n);
                memcpy(v, &g[n*j], n);
                setint<<<1,1>>>(n,v,u1);
                cudaDeviceSynchronize();
                int tmp = 0;
                for(int k = 0; k < n; k++){
                    if(v[k]) tmp++;
                }
                count += tmp;
                cudaFree(v);
            }
        }
        cudaFree(u1);
    }
    return count;
}

//3 nested loops that check possible triangles
int count_triangles(int n, bool *g){
    int count = 0; 
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            for(int k = 0; k < n; k++){
                if(g[n*i+j] && g[n*j+k] && g[n*k+i]) count++;
            }
        }
    }
    return count;
}

int main(){
    int N = 1<<10;
    bool *g;
    cudaMallocManaged(&g, N*N);
    //initialize graph with random edges
    for(int i = 0; i < N*N; i++){
        g[i] = rand() % 2;
    }
    //make graph undirected
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            g[N*i+j] = g[N*i+j] | g[N*j+i];
        }
    }
    for(int i = 0; i < N; i++){
        g[N*i+i] = 0;
    }
    int res = tc(N, g);
    assert(res % 3 == 0);
    int trueres = count_triangles(N, g);
    assert(trueres % 6 == 0);
    res /= 3;
    trueres /= 6;
    assert(res == trueres);
    std::cout << "Number of triangles in graph g: " << res << std::endl;
}