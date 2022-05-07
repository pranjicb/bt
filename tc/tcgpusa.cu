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

__device__ int binarySearch(T x, T l, T r, T *arr){
    while(l <= r){
        T m = l + (r-l) / 2;
        if(arr[m] == x){
            return x;
        }
        if(arr[m] < x){
            l = m + 1;
        }
        else{
            r = m - 1;
        }
    }
    return -1;
}

__device__ T intersectCount(T lu, T ru, T lv, T rv, T *edges2){
    T count = 0;
    for(int i = lu; i < ru; i++){
        T u = edges2[i];
        if(binarySearch(u, lv, rv, edges2) == u) count++;
    }
    return count;
}


__global__ void tcgpusa(T N, T E, T *nodes, T *edges1, T*edges2, T *sum){
    T t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t < E){
        T u = edges1[t];
        T v = edges2[t];
        T counter = 0;
        if(u < v){
            T lu = nodes[u];
            T lv = nodes[v];
            T ru, rv;
            if(u != N-1){
                ru = nodes[u+1];
            }
            else{
                ru = E-1;
            }
            if(v != N-1){
                rv = nodes[v+1];
            }
            else{
                rv = E-1;
            }
            T sizeU = ru-lu;
            T sizeV = rv-lu;
            if(sizeU < sizeV){
                counter += intersectCount(lu,ru,lv,rv,edges2);
            }
            else{
                counter += intersectCount(lv,rv,lu,ru,edges2);
            }
            
        }
        sum[t] = counter;

    }
}

__global__ void tcgpusa2(T N, T E, T *nodes, T *edges1, T*edges2, T *sum){
    int u = blockIdx.x; //launch 1 warp per node
    T lu = nodes[u]; 
    T ru;
    if(u != N-1){
        ru = nodes[u+1];
    }
    else{
        ru = E-1;
    }
    T sizeU = ru-lu;
    for(int i = threadIdx.x; i < sizeU; i+=32){
        T v = edges2[lu+i];
        T lv = nodes[v];
        T rv;
        if(v != N-1){
            rv = nodes[v+1];
        }
        else{
            rv = E-1;
        }
        T sizeV = rv-lv;
        T counter = 0;
        if(u < v){
            if(sizeU < sizeV){
                counter = intersectCount(lu,ru,lv,rv,edges2);
            }
            else{
                counter = intersectCount(lv,rv,lu,ru,edges2);
            }
        }
        sum[lu+i] = counter;
    }
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

T N = N_BIOSCGT;
T E = E_BIOSCGT*2;
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


    cudaFree(sumgpusa);
    cudaFree(edges1);
    cudaFree(edges2);
    cudaFree(nodes);
    cudaFree(g);
    cudaFree(sumcpu);
}