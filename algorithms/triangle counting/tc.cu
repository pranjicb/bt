#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <chrono>

//CUDA kernel for triangle counting
__global__ void tcgpu(int *sum, int n, bool *g){
    int u = threadIdx.x; //each thread calculates neighbor intersection for one node
    bool *u1 = &g[n*u];
    for(int v = 0; v < u; v++){ //for all vertices in g
        if(g[n*u+v]){           //if v is a neighbor of u
            bool *v1 = &g[n*v];
            for(int i = 0; i < n; i++){
                if(v1[i] & u1[i]) sum[u]++; //store the count of the intersections in array for global reduction
            }
        }
    }
}

//CPU implementation for triangle counting
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
    //remove self cycles
    for(int i = 0; i < N; i++){
        g[N*i+i] = 0;
    }
    //array to store result for each thread 
    int *sum;
    cudaMallocManaged(&sum, N*sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //launch and time kernel
    cudaEventRecord(start);
    tcgpu<<<1,1024>>>(sum, N, g); 
    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    //sum up results from all threads
    int res = 0;
    for(int i = 0; i < N; i++){
        res += sum[i];
    }
    
    //time and launch CPU implementation
    auto cpustart = std::chrono::steady_clock::now();
    int trueres = count_triangles(N, g);
    auto cpuend = std::chrono::steady_clock::now();

    //makes sure the result is correct
    assert(res % 3 == 0);
    assert(trueres % 6 == 0);
    assert(res/3 == trueres/6);
    res /= 3;
    trueres /= 6;
    
    //print results and time
    float kerneltime = 0;
    cudaEventElapsedTime(&kerneltime, start, stop);
    std::chrono::duration<double> cputime = cpuend-cpustart;
    std::cout << "GPU result: " << res << std::endl;
    std::cout << "CPU result: " << trueres << std::endl;
    std::cout << "GPU kernel time: " << kerneltime/1000.0 << "s" << std::endl;
    std::cout << "CPU time: " << cputime.count() << "s" << std::endl;
    std::cout << "Speedup: " << cputime.count()/(kerneltime/1000.0) << "x" << std::endl;
}
