#include <iostream>
using namespace std;

int main(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Name: " << prop.name << endl;
    cout << "Compute capability: " << prop.major << "." << prop.minor << endl;
    cout << "clock frequency: " << prop.clockRate/1000000.0 << " GHz" << endl;
    cout << "L2 cache size in bytes: " << prop.l2CacheSize << endl;
    cout << "Number of SM: " << prop.multiProcessorCount << endl;
    cout << "Maximum number of threads per SM: " << prop.maxThreadsPerMultiProcessor << endl;
    cout << "Maximum number of threads per block: " << prop.maxThreadsPerBlock << endl;
    cout << "Maximum number of blocks per SM: " << prop.maxBlocksPerMultiProcessor << endl;
    cout << "Global memory bus bandwidth: " << prop.memoryBusWidth/1e9 << " Gb" << endl;
    cout << "Total global memory availale: " << prop.totalGlobalMem/1e9 << " GB" << endl;
    cout << "Warp size in threads: " << prop.warpSize << endl;
}