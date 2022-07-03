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

__device__ T largeLargeIntersectCount(unsigned int n, T *u, T *v){
    T count = 0;
    for(unsigned int i = 0; i < n; i++){
        count += __popcll(u[i] & v[i]);
    }
    return count;
}

__device__ T largeSmallIntersectCount(unsigned int n, T *u, T *lv, T *rv){
    
}

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