#include "assert.h"
#include "graphreader.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <set>
#include <algorithm>
#include <fstream>

#define T u_int32_t


int type0 = 0;
int type1 = 1;
int type2 = 2;
int type3 = 3;
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



__device__ T smallSmallIntersectCount(T* lu, T* ru, T* lv, T* rv){
    T count = 0;
    while(lu != ru && lv != rv){
        if(*lu < *lv){
            ++lu;
        }
        else{
            if(!(*lv < *lu)){
                ++count;
                ++lu;
            }
            ++lv;
        }
    } 
    return count;
}

//1 warp per large-large edge
__global__ void llKernel(T *g, T n1, T n2, T *edges1, T *edges2, T *nodes, T *sum){
    T tid = blockDim.x * blockIdx.x + threadIdx.x;
    T wid = tid / 32;
    T lid = tid % 32;
    T u = edges1[wid];
    T v = edges2[wid];
    T u_neigh = nodes[u]*n2;
    T v_neigh = nodes[v]*n2;
    T count = 0;
    for(unsigned int i = lid; i < n2; i++){
        count += __popc(g[u_neigh+i] & g[v_neigh+i]);
    }
    for(int offset = 16; offset > 0; offset /= 2){
        count += __shfl_down_sync(0xffffffff, count, offset);
    }
    sum[u] = count;
}

//1 thread per edge for other edges
__global__ void otherKernel(T *g_large, T n2, T *edges1, T *edges2, bool *isLarge, T *largeNodeIndex, T *smallNodeIndex, T *sum){
    T tid = blockDim.x * blockIdx.x + threadIdx.x;
    T u = edges1[tid];
    T v = edges2[tid];
    if(isLarge[u]){
        T count = 0;
        T *lv = &edges2[smallNodeIndex[v]];
        T *rv = &edges2[smallNodeIndex[v+1]];
        T u_neigh = largeNodeIndex[u]*n2;
        T size = rv-lv;
        for(unsigned int i = 0; i < size; i++){
            T x = *(lv+i);
            T idx = x / 32;
            T shamt = x % 32;
            count += ((g_large[u_neigh+idx] & (1ULL << shamt)) == (1ULL << shamt));
        }
        sum[tid] = count;
    }
    else if(isLarge[v]){
        T count = 0;
        T *lu = &edges2[smallNodeIndex[u]];
        T *ru = &edges2[smallNodeIndex[u+1]];
        T v_neigh = largeNodeIndex[v]*n2;
        T size = ru-lu;
        for(unsigned int i = 0; i < size; i++){
            T x = *(lu+i);
            T idx = x / 32;
            T shamt = x % 32;
            count += ((g_large[v_neigh+idx] & (1ULL << shamt)) == (1ULL << shamt));
        }
        sum[tid] = count;
    }
    else{
        T count = 0;
        T *lu = &edges2[smallNodeIndex[u]];
        T *lv = &edges2[smallNodeIndex[v]];
        T *ru = &edges2[smallNodeIndex[u+1]];
        T *rv = &edges2[smallNodeIndex[v+1]];
        sum[tid] = smallSmallIntersectCount(lu,ru,lv,rv);
    }
}


int main(){
    unsigned int N = N_BIOSCGT;
    unsigned int E = E_BIOSCGT;
    std::vector<std::vector<unsigned int>> g;
    for(unsigned int i = 0; i < N; i++){
        std::vector<unsigned int> u;
        g.push_back(u);
    }
    std::vector<unsigned int> degs1 = std::vector<unsigned int>(N,0);
    std::vector<unsigned int> degs2 = std::vector<unsigned int>(N,0);
    std::ifstream INPUT(bioscgt_path);
    unsigned int u,v;
    std::string w;
    unsigned int count = 0;
    auto readstart = std::chrono::steady_clock::now();
    while(INPUT >> u >> v >> w){
        if(u < v){
            g.at(u).push_back(v);
            degs1.at(u)++;
            degs1.at(v)++;
            degs2.at(u)++;
            degs2.at(v)++;
        }
        else if(v < u){
            g.at(v).push_back(u);
            degs1.at(u)++;
            degs1.at(v)++;
            degs2.at(u)++;
            degs2.at(v)++;
        }
    }
    for(unsigned int i = 0; i < N; i++){
        std::set<unsigned int> s(g.at(i).begin(), g.at(i).end());
        g.at(i).assign(s.begin(),s.end());
        count += g.at(i).size();
    }
    auto readend = std::chrono::steady_clock::now();
    std::chrono::duration<double> readtime = readend-readstart;
    std::cout << "Read Time:    " << readtime.count() << "s" << std::endl;
    E = count;

    unsigned int threshold = N / 20;
    std::nth_element(degs1.begin(), degs1.begin()+threshold, degs1.end(), std::greater<unsigned int>());
    unsigned int cutoff = degs1[threshold];
    unsigned int n1Large = threshold-1;
    unsigned int n2Large;
    if(n1Large % 32 == 0){
        n2Large = n1Large / 32;
    }
    else{
        n2Large = (n1Large + 31) / 32;
    }
    T *g_large;
    cudaMallocManaged(&g_large, n1Large*n2Large*sizeof(T));
    for(unsigned int i = 0; i < n1Large*n2Large; i++){
        g_large[i] = 0UL;
    }
    bool *isLargeNode;
    cudaMallocManaged(&isLargeNode, N*sizeof(bool));
    T *largeNodeId;
    cudaMallocManaged(&largeNodeId, N*sizeof(T));
    unsigned int llEdgeCount = 0;
    unsigned int idx = 0;
    for(unsigned int i = 0; i < N; i++){
        if(degs2.at(i) > cutoff){
            if(idx < n1Large){
                isLargeNode[i] = true;
                largeNodeId[i] = idx;
                for(unsigned int j = 0; j < g.at(i).size(); j++){
                    unsigned int v = g.at(i).at(j);
                    if(degs2.at(v) > cutoff){
                        llEdgeCount++;
                    }
                    unsigned int denseIdx = v / 32;
                    unsigned int shamt = v % 32;
                    g_large[idx*n2Large + denseIdx] |= (1ULL << shamt);
                }   
                idx++;
            }
            
        }
        else{
            isLargeNode[i] = false;
        }
    }
    unsigned int edgeCount = E - llEdgeCount;
    T *llEdges1, *llEdges2;
    cudaMallocManaged(&llEdges1, llEdgeCount*sizeof(T));
    cudaMallocManaged(&llEdges2, llEdgeCount*sizeof(T));
    
    T *edges1, *edges2;
    cudaMallocManaged(&edges1, edgeCount*sizeof(T));
    cudaMallocManaged(&edges2, edgeCount*sizeof(T));
    T *allNodesIdx;
    cudaMallocManaged(&allNodesIdx, N*sizeof(T));
    for(unsigned int i = 0; i < N; i++){
        allNodesIdx[i] = 0;
    }
    unsigned int llit = 0;
    unsigned int it = 0;
    for(unsigned int i = 0; i < N; i++){
        allNodesIdx[i] = it;
        if(isLargeNode[i]){
            for(unsigned int j = 0; j < g.at(i).size(); j++){
                if(isLargeNode[g.at(i).at(j)]){
                    llEdges1[llit] = i;
                    llEdges2[llit] = g.at(i).at(j);
                    llit++;
                }
                else{
                    edges1[it] = i;
                    edges2[it] = g.at(i).at(j);
                    it++;
                }
            }
        }
        else{
            for(unsigned int j = 0; j < g.at(i).size(); j++){
                edges1[it] = i;
                edges2[it] = g.at(i).at(j);
                it++;
            }
        }
    }

    T *sumLarge, *sumSmall;
    cudaMallocManaged(&sumLarge, llEdgeCount*sizeof(T));
    cudaMallocManaged(&sumSmall, edgeCount*sizeof(T));
    for(unsigned int i = 0; i < edgeCount; i++){
        if(i < llEdgeCount){
            sumLarge[i] = 0;
        }
        sumSmall[i] = 0;
    }

    unsigned int threadsPerBlock = 128;
    unsigned int largeBPG = (32*llEdgeCount + threadsPerBlock - 1) / threadsPerBlock;
    unsigned int smallBPG = (edgeCount + threadsPerBlock - 1) / threadsPerBlock;


    auto start = std::chrono::steady_clock::now();
    llKernel<<<largeBPG, threadsPerBlock>>>(g_large, n1Large, n2Large, llEdges1, llEdges2, largeNodeId, sumLarge);

    otherKernel<<<smallBPG, threadsPerBlock>>>(g_large, n2Large, edges1, edges2, isLargeNode, largeNodeId, allNodesIdx, sumSmall);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> time = end-start;
    std::cout << "RUN TIME:       " << time.count() << "s" << std::endl;
    u_int64_t res = 0;

    for(unsigned int i = 0; i < llEdgeCount; i++){
        res += sumLarge[i];
    }
    for(unsigned int i = 0; i < edgeCount; i++){
        res += sumSmall[i];
    }

    std::cout << "TRIANGLE COUNT: " << res << std::endl;







}