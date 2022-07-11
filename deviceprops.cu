#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main(){
    ofstream out("deviceprops.txt");
    streambuf *coutbuf = cout.rdbuf();
    cout.rdbuf(out.rdbuf());
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
    cout << "Reserved shared memory per block in bytes: " << prop.reservedSharedMemPerBlock << endl;
    cout << "Shared memory available per block in bytes: " << prop.sharedMemPerBlock << endl;
    cout << "Shared memory per multiprocessor in bytes: " << prop.sharedMemPerMultiprocessor << endl;
    cout << "L2 cache size in bytes: " << prop.l2CacheSize << endl;
}