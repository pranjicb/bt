#include "assert.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>

#define T u_int32_t

class Graph{
    public: 
        int N;
        int E;
        int kind;
        std::string path;
        Graph(int n, int e, int type, std::string filepath){
            N = n;
            E = e;
            kind = type;
            path = filepath;
        }
        void readGraph(bool *g){
            std::ifstream INPUT(path);
            T u,v;
            std::string c;
            switch(kind){
                case 0 : 
                    while(INPUT >> u >> v >> c){
                        g[N*u+v] = true;
                        g[N*v+u] = true;
                    }
                    break;
                case 1 : 
                    while(INPUT >> u >> v >> c){
                        u--;
                        v--;
                        g[N*u+v] = true;
                        g[N*v+u] = true;
                    }
                    break;
                case 2 : 
                    while(INPUT >> u >> v){
                        g[N*u+v] = true;
                        g[N*v+u] = true;
                    }
                    break;
                case 3 : 
                    while(INPUT >> u >> v){
                        u--;
                        v--;
                        g[N*u+v] = true;
                        g[N*v+u] = true;
                    }
                    break;

            }
        }

};

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

//SOC-SINAWEIBO
std::string socweibo_path = "../../../datasets/social/soc-sinaweibo/soc-sinaweibo.mtx";
int N_SOCWEIBO = 58655849;
int E_SOCWEIBO = 261321071;
/////////////////////////////////////////////////////////////////////////


__device__ T intersectCount(int n, T *u, T *v){
    T count = 0ULL;
    for(int i = 0; i < n; i++){
        count += __popcll(u[i] & v[i]);
    }
    return count;
}

__global__ void tcgpudb(int n1, int n2, T *sum, T *g, int numbits){
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
        if(tmp2 == tmp3 && u < v){
            T count = intersectCount(n2, &g[u_neigh], &g[v_neigh]);
            sum[u*n1+v] += count;
        }
        
    }
}

/*__global__ void tcgpudb(int n1, int n2, T *sum, T *g, int numbits){
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    T u_neigh = u*n2;
    if(u < n1){
        for(int i = 0; i < n2; i++){
            T tmp1 = g[u_neigh+i];
            for(int j = 0; j < numbits; j++){
                T tmp2 = (1ULL << j);
                T tmp3 = tmp1 & tmp2;
                T v = i*numbits+j;
                T v_neigh = v*n2;
                if(tmp2 == tmp3 && u < v){
                    T count = intersectCount(n2, &g[u_neigh], &g[v_neigh]);
                    sum[u*n1+v] += count;
                }
            }
        }
    }
    
}*/



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



int numbits = sizeof(T) * 8;
unsigned int N = N_SOCWEIBO;
unsigned int E = E_SOCWEIBO;

int main(){
    std::ifstream INPUT(socweibo_path);
    unsigned int u,v;
    unsigned int max = 0;
    while(INPUT >> u >> v){
        if(max < u) max = u;
        if(max < v) max = v;
    }
    std::cout << max << std::endl;
}