#include "assert.h"
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

///////////////////////INPUT FILES//////////////////////////////////////
//BIO-CE-PG
std::string biocepg_path = "../../../datasets/biological/bio-CE-PG/bio-CE-PG.edges";
int N_BIOCEPG = 1871; 
int E_BIOCEPG = 47754; 
int T_BIOCEPG = type0;

//BIO-DM-CX
std::string biodmcx_path = "../../../datasets/biological/bio-DM-CX/bio-DM-CX.edges";
int N_BIODMCX = 1716; 
int E_BIODMCX = 33987; 
int T_BIODMCX = type0;

//BIO-HS-LC
std::string biohslc_path = "../../../datasets/biological/bio-HS-LS/bio-HS-LC.edges";
int N_BIOHSLC = 1716; 
int E_BIOHSLC = 33987; 
int T_BIOHSLC = type0;

//BIO-HUMAN-GENE2
std::string biohumangene2_path = "../../../datasets/biological/bio-human-gene2/bio-human-gene2.edges";
int N_BIOHUMANGENE2 = 14340; 
int E_BIOHUMANGENE2 = 9041364; 
int T_BIOHUMANGENE2 = type1;

//BIO-MOUSE-GENE
std::string biomousegene_path = "../../../datasets/biological/bio-mouse-gene/bio-mouse-gene.edges";
int N_BIOMOUSEGENE = 45101;
int E_BIOMOUSEGENE = 14506196;
int T_BIOMOUSEGENE = type1;

//BIO-SC-GT
std::string bioscgt_path = "../../../datasets/biological/bio-SC-GT/bio-SC-GT.edges";
int N_BIOSCGT = 1716; 
int E_BIOSCGT = 33987; 
int T_BIOSCGT = type0;

//BIO-SC-HT
std::string bioscht_path = "../../../datasets/biological/bio-SC-HT/bio-SC-HT.edges";
int N_BIOSCHT = 1716; 
int E_BIOSCHT = 33987; 
int T_BIOSCHT = type0;

//BIO-WORMNET-V3
std::string biowormnetv3_path = "../../../datasets/biological/bio-WormNet-v3-benchmark/bio-WormNet-v3-benchmark.edges";
int N_WORMNETV3 = 0;
int E_WORMNETV3 = 0;
int T_WORMNETV3 = type2;

//BN-FLY
std::string bnfly_path = "../../../datasets/brain/bn-fly-drosophila_medulla_1/bn-fly-drosophila_medulla_1.edges";
int N_BNFLY = 0;
int E_BNFLY = 0;
int T_BNFLY = type2;

//BN-MOUSE
std::string bnmouse_path = "../../../datasets/brain/bn-mouse_brain_1/bn-mouse_brain_1.edges";
int N_BNMOUSE = 0;
int E_BNMOUSE = 0;
int T_BNMOUSE = type2;

//C500-9
std::string c500_path = "../../../datasets/dimac/C500-9/C500-9.mtx";
int N_C500 = 500;
int E_C500 = 112332;
int T_C500 = type3;

//ECON-BEACXC
std::string econbeacxc_path = "../../../datasets/economic/econ-beacxc/econ-beacxc.mtx";
int N_ECONBEACXC = 497;
int E_ECONBEACXC = 50409;
int T_ECONBEACXC = type1;

//ECON-BEAflw
std::string econbeaflw_path = "../../../datasets/economic/econ-beaflw/econ-beaflw.mtx";
int N_ECONBEAFLW = 507;
int E_ECONBEAFLW = 53403;
int T_ECONBEAFLW = type1;

//SC-PWTK
std::string scpwtk_path = "../../../datasets/scientific/sc-pwtk/sc-pwtk.mtx";
int N_SCPWTK = 217891;
int E_SCPWTK = 5653221;
int T_SCPWTK = type3;

//SOC-SINAWEIBO
std::string socweibo_path = "../../../datasets/social/soc-sinaweibo/soc-sinaweibo.mtx";
int N_SOCWEIBO = 58655849;
int E_SOCWEIBO = 261321071;
int T_SOCWEIBO = type3;
/////////////////////////////////////////////////////////////////////////

void tccpu(unsigned int n, T *sum, std::vector<std::vector<unsigned int>> g){
    #pragma omp parallel num_threads(32)
    {   
        #pragma omp for
        for(T i = 0; i < n; i++){
            std::vector<unsigned int> u = g.at(i);
            for(int j : u){
                if(i < j){
                    std::vector<unsigned int> v = g.at(j);
                    std::vector<unsigned int> w;
                    std::set_intersection(u.begin(), u.end(), v.begin(), v.end(), back_inserter(w));
                    sum[i] += w.size();
                }
                
            }
        } 
    }

}


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

//launch 1 thread per edge
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

int main(){
    unsigned int N = N_BIOMOUSEGENE;
    unsigned int E = E_BIOMOUSEGENE;
    std::vector<std::vector<unsigned int>> g;
    for(unsigned int i = 0; i < N; i++){
        std::vector<unsigned int> u;
        g.push_back(u);
    }
    std::ifstream INPUT(biomousegene_path);
    unsigned int u,v;
    std::string w;
    unsigned int count = 0;
    auto readstart = std::chrono::steady_clock::now();
    while(INPUT >> u >> v >> w){
        if(u != v){
            u--;
            v--;
            g.at(u).push_back(v);
            g.at(v).push_back(u);
        }
    }
    for(unsigned int i = 0; i < N; i++){
        std::set<unsigned int> s(g.at(i).begin(), g.at(i).end());
        g.at(i).assign(s.begin(), s.end());
        count += g.at(i).size();
    }
    auto readend = std::chrono::steady_clock::now();
    std::chrono::duration<double> readtime = readend-readstart;

    E = count;

    T *sumcpu;
    cudaMallocManaged(&sumcpu, N*sizeof(T));
    for(unsigned int i = 0; i < N; i++){
        sumcpu[i] = 0;
    }
    auto cpustart = std::chrono::steady_clock::now();
    tccpu(N, sumcpu, g);
    T cpures = 0;
    for(unsigned int i = 0; i < N; i++){
        cpures += sumcpu[i];
    }
    auto cpuend = std::chrono::steady_clock::now();
    std::chrono::duration<double> cputime = cpuend-cpustart;
    cpures /= 3;


    T *edges1;
    cudaMallocManaged(&edges1, E*sizeof(T));
    T *edges2;
    cudaMallocManaged(&edges2, E*sizeof(T));
    T *nodes;
    cudaMallocManaged(&nodes, N*sizeof(T));
    for(unsigned int i = 0; i < E; i++){
        edges1[i] = 0;
        edges2[i] = 0;
    }
    for(unsigned int i = 0; i < N; i++){
        nodes[i] = 0;
    }

    T idx = 0;
    for(T i = 0; i < N; i++){
        nodes[i] = idx;
        for(T j = 0; j < g.at(i).size(); j++){
            edges1[idx] = i;
            edges2[idx] = g.at(i).at(j);
            idx++;
        }
    }
    g = std::vector<std::vector<unsigned int>>();

    T *sumgpusa;
    cudaMallocManaged(&sumgpusa, E*sizeof(T));
    for(unsigned int i = 0; i < E; i++){
        sumgpusa[i] = 0;
    }
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


    std::cout << "Read Time:    " << readtime.count() << "s" << std::endl;
    std::cout << "CPU RES:      " << cpures << std::endl; 
    std::cout << "GPU SA RES:   " << resgpusa << std::endl;
    std::cout << "CPU TIME:     " << cputime.count() << "s" << std::endl;
    std::cout << "GPU SA TIME:  " << sakerneltime/1000.0 << "s" << std::endl;
    std::cout << "GPU speedup:  " << cputime.count()/(sakerneltime/1000.0) << "x" <<  std::endl;
    
}
