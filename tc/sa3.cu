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

///////////////////////INPUT FILES//////////////////////////////////////
//BIO-CE-PG
std::string biocepg_path = "../../../datasets/biological/bio-CE-PG/bio-CE-PG.edges";
int N_BIOCEPG = 1871; 
int E_BIOCEPG = 47754; 
int T_BIOCEPG = type0;
u_int64_t BIOCEPG_RES = 784919;
double BIOCEPG_TIME = 0.0101918;

//BIO-DM-CX
std::string biodmcx_path = "../../../datasets/biological/bio-DM-CX/bio-DM-CX.edges";
int N_BIODMCX = 4040; 
int E_BIODMCX = 76717; 
int T_BIODMCX = type0;
u_int64_t BIODMCX_RES = 733893;
double BIODMCX_TIME = 0.00839039;

//BIO-HS-LC
std::string biohslc_path = "../../../datasets/biological/bio-HS-LC/bio-HS-LC.edges";
int N_BIOHSLC = 4227; 
int E_BIOHSLC = 39484; 
int T_BIOHSLC = type0;
u_int64_t BIOHSLC_RES = 231634;
double BIOHSLC_TIME = 0.00427897;

//BIO-HUMAN-GENE2
std::string biohumangene2_path = "../../../datasets/biological/bio-human-gene2/bio-human-gene2.edges";
int N_BIOHUMANGENE2 = 14340; 
int E_BIOHUMANGENE2 = 9041364; 
int T_BIOHUMANGENE2 = type1;
u_int64_t BIOHUMANGENE2_RES = 4905433564;
double BIOHUMANGENE2_TIME = 10.222;

//BIO-MOUSE-GENE
std::string biomousegene_path = "../../../datasets/biological/bio-mouse-gene/bio-mouse-gene.edges";
int N_BIOMOUSEGENE = 45101;
int E_BIOMOUSEGENE = 14506196;
int T_BIOMOUSEGENE = type1;
u_int64_t BIOMOUSEGENE_RES = 3619097862;
double BIOMOUSEGENE_TIME = 15.4146;

//BIO-SC-GT
std::string bioscgt_path = "../../../datasets/biological/bio-SC-GT/bio-SC-GT.edges";
int N_BIOSCGT = 1716; 
int E_BIOSCGT = 33987; 
int T_BIOSCGT = type0;
u_int64_t BIOSCGT_RES = 369047;
double BIOSCGT_TIME = 0.00381811;

//BIO-SC-HT
std::string bioscht_path = "../../../datasets/biological/bio-SC-HT/bio-SC-HT.edges";
int N_BIOSCHT = 2084; 
int E_BIOSCHT = 63027; 
int T_BIOSCHT = type0;
u_int64_t BIOSCHT_RES = 1397660;
double BIOSCHT_TIME = 0.0117545;

//BIO-WORMNET-V3
std::string biowormnetv3_path = "../../../datasets/biological/bio-WormNet-v3-benchmark/bio-WormNet-v3-benchmark.edges";
int N_WORMNETV3 = 2445;
int E_WORMNETV3 = 78736;
int T_WORMNETV3 = type2;
u_int64_t WORMNETV3_RES = 2015875;
double WORMNETV3_TIME = 0.00372158;

//BN-FLY
std::string bnfly_path = "../../../datasets/brain/bn-fly-drosophila_medulla_1/bn-fly-drosophila_medulla_1.edges";
int N_BNFLY = 1781;
int E_BNFLY = 33641;
int T_BNFLY = type2;
u_int64_t BNFLY_RES = 16255;
double BNFLY_TIME = 0.000915822;

//BN-MOUSE
std::string bnmouse_path = "../../../datasets/brain/bn-mouse_brain_1/bn-mouse_brain_1.edges";
int N_BNMOUSE = 213;
int E_BNMOUSE = 21807;
int T_BNMOUSE = type2;
u_int64_t BNMOUSE_RES = 622414;
double BNMOUSE_TIME = 0.00194085;

//C500-9
std::string c500_path = "../../../datasets/dimac/C500-9/C500-9.mtx";
int N_C500 = 500;
int E_C500 = 112332;
int T_C500 = type3;
u_int64_t C500_RES = 15119852;
double C500_TIME = 0.0169249;

//ECON-BEACXC
std::string econbeacxc_path = "../../../datasets/economic/econ-beacxc/econ-beacxc.mtx";
int N_ECONBEACXC = 497;
int E_ECONBEACXC = 50409;
int T_ECONBEACXC = type1;
u_int64_t BEACXC_RES = 1831667;
double BEACXC_TIME = 0.00518382;

//ECON-BEAFLW
std::string econbeaflw_path = "../../../datasets/economic/econ-beaflw/econ-beaflw.mtx";
int N_ECONBEAFLW = 507;
int E_ECONBEAFLW = 53403;
int T_ECONBEAFLW = type1;
u_int64_t BEAFLW_RES = 2011146;
double BEAFLW_TIME = 0.00553629;

//SC-PWTK
std::string scpwtk_path = "../../../datasets/scientific/sc-pwtk/sc-pwtk.mtx";
int N_SCPWTK = 217891;
int E_SCPWTK = 5653221;
int T_SCPWTK = type3;
u_int64_t SCPWTK_RES = 55981589;
double SCPWTK_TIME = 0.123787;

//SOC-SINAWEIBO
std::string socweibo_path = "../../../datasets/social/soc-sinaweibo/soc-sinaweibo.mtx";
int N_SOCWEIBO = 58655849;
int E_SOCWEIBO = 261321071;
int T_SOCWEIBO = type3;
u_int64_t SOCWEIBO_RES = 30518752;
double SOCWEIBO_TIME = 288.545;
/////////////////////////////////////////////////////////////////////////

__device__ bool binary_search(T* arr, T l, T r, T x){
    while(l <= r){
        T m = l + (r - l + 1) / 2;
        if(arr[m] == x){
            return true;
        }
        if(arr[m] < x){
            l = m + 1;
        }
        else{
            r = m - 1;
        }
    }
    return false;
}

__global__ void tcgpusa(T N, T E, T *nodes, T *edges1, T*edges2, T *sum){
    u_int64_t tid = 1ULL * blockDim.x * blockIdx.x + threadIdx.x ;
    u_int64_t wid = tid / 32;
    T lid = threadIdx.x % 32;
    T u = edges1[wid];
    T v = edges2[wid];
    T u_size = nodes[u+1] - nodes[u];
    T v_size = nodes[v+1] - nodes[v];
    T *lu = &edges2[nodes[u]];
    T *lv = &edges2[nodes[v]];
    T count = 0;
    for(unsigned int i = lid; i < v_size; i+=32){
        count += binary_search(lu, 0, u_size-1, *(lv+i));
    }
    for(int offset = 16; offset > 0; offset /= 2){
        count += __shfl_down_sync(0xffffffff, count, offset);
    }
    sum[wid] = count;
}



int main(){
    unsigned long long int N = N_SOCWEIBO;
    unsigned long long int E = E_SOCWEIBO;
    unsigned int maxdeg = 0;
    std::vector<std::vector<unsigned int>> g;
    for(unsigned int i = 0; i < N; i++){
        std::vector<unsigned int> u;
        g.push_back(u);
    }
    std::ifstream INPUT(socweibo_path);
    unsigned int u,v;
    std::string w;
    unsigned int count = 0;
    auto readstart = std::chrono::steady_clock::now();
    while(INPUT >> u >> v){
        if(u < v){
            --u;
            --v;
            g.at(u).push_back(v);
        }
        else if(v < u){
            --u;
            --v;
            g.at(v).push_back(u);
        }
    }
    for(unsigned int i = 0; i < N; i++){
        std::set<unsigned int> s(g.at(i).begin(), g.at(i).end());
        g.at(i).assign(s.begin(),s.end());
        if(g.at(i).size() > maxdeg) maxdeg = g.at(i).size();
        count += g.at(i).size();
    }
    std::cout << "MAX DEGREE:   " << maxdeg << std::endl;
    auto readend = std::chrono::steady_clock::now();
    std::chrono::duration<double> readtime = readend-readstart;
    std::cout << "Read Time:    " << readtime.count() << "s" << std::endl;
    E = count;
    
    T *edges1;
    cudaMallocManaged(&edges1, E*sizeof(T));
    T *edges2;
    cudaMallocManaged(&edges2, E*sizeof(T));
    T *nodes;
    cudaMallocManaged(&nodes, (N+1)*sizeof(T));
    for(unsigned int i = 0; i < E; i++){
        edges1[i] = 0;
        edges2[i] = 0;
    }
    for(unsigned int i = 0; i < N; i++){
        nodes[i] = 0;
    }
    nodes[N] = E;

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

    cudaStream_t s1;
    cudaStreamCreate(&s1);
    cudaMemPrefetchAsync(edges2,E*sizeof(T),0,s1);
    u_int64_t tmp = 32ULL*E;
    unsigned long long int threadsPerBlock = 512;
    unsigned long long int blocksPerGrid = (tmp + threadsPerBlock - 1)/ threadsPerBlock;

    //launch and time kernel
    cudaEventRecord(sastart);
    tcgpusa<<<blocksPerGrid, threadsPerBlock>>>(N, E, nodes, edges1, edges2, sumgpusa); 
    cudaEventRecord(sastop);

    cudaDeviceSynchronize();
    cudaEventSynchronize(sastop);
    u_int64_t resgpusa = 0;
    for(T i = 0; i < E; i++){
        resgpusa += sumgpusa[i];
    }
    float sakerneltime = 0;
    cudaEventElapsedTime(&sakerneltime, sastart, sastop);
    
    
    std::cout << "GPU SA RES:   " << resgpusa << std::endl;
    std::cout << "GPU SA TIME:  " << sakerneltime/1000.0 << "s" << std::endl;
    
}
