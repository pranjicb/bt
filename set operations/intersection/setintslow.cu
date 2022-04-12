#include <iostream>
#include <stdlib.h>

__global__ void setint(int n, bool *x, bool *y){
    for(int i = 0; i < n; i++){
        x[i] = x[i] & y[i];
    }
}


int main(void){

    int N = 1<<20;
    bool *x, *y, *z;

    cudaMallocManaged(&x, N*sizeof(bool));
    cudaMallocManaged(&y, N*sizeof(bool));
    cudaMallocManaged(&z, N*sizeof(bool));

    for(int i = 0; i < N; i++){
        x[i] = rand() % 2;
        y[i] = rand() % 2;
        z[i] = x[i] & y[i];
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    setint<<<1,1>>>(N,x,y);
    cudaDeviceSynchronize();

    int count = 0;
    for(int i = 0; i < N; i++){
        count += x[i] ^ z[i];
    }

    std::cout << "Error: " << count << std::endl;

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

}