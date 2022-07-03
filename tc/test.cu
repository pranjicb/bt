#include "assert.h"
#include "graphreader.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <set>
#include <algorithm>
#include <fstream>

#define T u_int32_t

__device__ T mergeIntersectCount(T* lu, T* ru, T* lv, T* rv){
    T count = 0;
    while(lu != ru && lv != rv){
        if(*lu < *lv){
            ++lu;
        }
        else{
            if(!(*lv < *lu)){
               count++;
                *lu++;
            }
            ++lv;
        }
    } 
    return count;
}

__device__ bool binarySearch(T* arr, T l, T r, T x){
    while(l <= r){
        int m = l + (r - l + 1) / 2;
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

//launch 1 thread per edge, merge based intersection
__global__ void sa1(T N, T E, T *nodes, T *edges1, T*edges2, T *sum){
    T t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t < E){
        T u = edges1[t];
        T v = edges2[t];
        T* lu = &edges2[nodes[u]];
        T* lv = &edges2[nodes[v]];
        T* ru = &edges2[nodes[u+1]];
        T* rv = &edges2[nodes[v+1]];
        sum[t] = mergeIntersectCount(lu,ru,lv,rv);
    }
}

//launch 1 warp per node, merge based intersection
__global__ void sa2(T N, T E, T *nodes, T*edges, T *sum){
    T tid = blockIdx.x * blockDim.x + threadIdx.x;
    T wid = tid / 32;
    T lid = threadIdx.x % 32;
    if(wid < N){
        T u = wid;
        T sizeU = nodes[u+1]-nodes[u];
        T* lu = &edges[nodes[u]];
        T* ru = lu+sizeU;
        T count = 0;
        for(unsigned int i = lid; i < sizeU; i+=32){
            T v = *(lu+i);
            T sizeV = nodes[v+1]-nodes[v];
            T* lv = &edges[nodes[v]];
            T* rv = lv+sizeV;
            count += mergeIntersectCount(lu,ru,lv,rv);
        }
        __syncthreads();
        for(int offset = 16; offset > 0; offset /= 2){
            count += __shfl_down_sync(0xffffffff, count, offset);
        }
        sum[u] = count;
    }
}

//launch 1 warp per node, parallel binary search based intersection with strided access
__global__ void sa3(T N, T E, T *nodes, T*edges, T *sum){
    T tid = blockIdx.x * blockDim.x + threadIdx.x;
    T wid = tid / 32;
    T lid = threadIdx.x % 32;
    if(wid < N){
        T u = wid;
        T sizeU = nodes[u+1]-nodes[u];
        T* lu = &edges[nodes[u]];
        T* ru = lu+sizeU;
        T count = 0;
        for(unsigned int i = 0; i < sizeU; i++){
            T v = *(lu+i);
            T* lv = &edges[nodes[v]];
            T sizeV = nodes[v+1]-nodes[v];
            for(unsigned int j = lid; j < sizeV; j+=32){
                count += binarySearch(lu, 0, sizeU-1, *(lv+j));
            }
        }
        __syncthreads();
        for(int offset = 16; offset > 0; offset /= 2){
            count += __shfl_down_sync(0xffffffff, count, offset);
        }
        sum[u] = count;
    }
}

//1 warp per edge, parallel binary search based intersection with strided access
__global__ void sa4(T N, T E, T *nodes, T *edges1, T*edges2, T *sum){
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
        count += binarySearch(lu, 0, u_size-1, *(lv+i));
    }
    for(int offset = 16; offset > 0; offset /= 2){
        count += __shfl_down_sync(0xffffffff, count, offset);
    }
    sum[wid] = count;
}
std::vector<unsigned int> blockSizes = {32, 64, 96, 128, 256, 512, 1024};
std::vector<double> occupancies = {0.3333, 0.6667, 1.0, 1.0, 1.0, 1.0, 0.6667};
int main(){
    std::ofstream out("tc_results.txt");
    std::streambuf *coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(out.rdbuf());
    for(int i = 0; i < testGraphs.size(); i++){
        std::cout << "#############################################" << std::endl;
        std::cout << "#############################################" << std::endl;
        GraphReader graph = testGraphs.at(i);
        graph.read();
        
        std::cout << "# Graph: " << graph.name << std::endl;  
        std::cout << "# Nodes: " << graph.N << std::endl;
        std::cout << "# Edges: " << graph.E << std::endl;
        std::cout << "# CPU TIME: " << graph.time << "s" <<std::endl;

        T *edges1, *edges2, *nodes, *sum;
        cudaMallocManaged(&edges1, graph.E*sizeof(T));
        cudaMallocManaged(&edges2, graph.E*sizeof(T));
        cudaMallocManaged(&nodes, (graph.N+1)*sizeof(T));
        for(unsigned int i = 0; i < graph.E; i++){
            edges1[i] = 0;
            edges2[i] = 0;
        }
        for(unsigned int i = 0; i < graph.N; i++){
            nodes[i] = 0;
        }
        nodes[graph.N] = graph.E;
        T idx = 0;
        for(T i = 0; i < graph.N; i++){
            nodes[i] = idx;
            for(T j = 0; j < graph.g.at(i).size(); j++){
                edges1[idx] = i;
                edges2[idx] = graph.g.at(i).at(j);
                idx++;
            }
        }
        ///////////////////////////////////////////////////////////////////
        //SA1
        std::cout << "#" << std::endl;
        std::cout << "# 1 thread per edge, merge based intersection" << std::endl;
        for(int j = 0; j < blockSizes.size(); j++){
            double sa1_time = 0;
            cudaEvent_t sa1start, sa1stop;
            cudaEventCreate(&sa1start);
            cudaEventCreate(&sa1stop);

            cudaStream_t s1;
            cudaStreamCreate(&s1);
            cudaMemPrefetchAsync(edges2,graph.E*sizeof(T),0,s1);

            int threadsPerBlock = blockSizes.at(j);
            T blocksPerGrid = (graph.E + threadsPerBlock - 1) / threadsPerBlock;
            for(int k = 0; k < 5; k++){
                cudaMallocManaged(&sum, graph.E*sizeof(T));
                for(unsigned int i = 0; i < graph.E; i++){
                    sum[i] = 0;
                }
                cudaEventRecord(sa1start);
                sa1<<<blocksPerGrid, threadsPerBlock>>>(graph.N, graph.E, nodes, edges1, edges2, sum); 
                cudaEventRecord(sa1stop);
                cudaDeviceSynchronize();
                cudaEventSynchronize(sa1stop);
                u_int64_t resgpu = 0;
                for(T i = 0; i < graph.E; i++){
                    resgpu += sum[i];
                }
                assert(resgpu==graph.res);
                float sa1kerneltime = 0;
                cudaEventElapsedTime(&sa1kerneltime, sa1start, sa1stop);
                sa1kerneltime /= 1000.0;
                sa1_time += sa1kerneltime;
                cudaFree(sum);
            }
            sa1_time /= 5.0;
            std::cout << "# " << threadsPerBlock << " threads per block: " << sa1_time << "s, Speedup: " << graph.time/sa1_time <<
             "x, theoretical occupancy: " << occupancies.at(j) << std::endl;
        }

        ///////////////////////////////////////////////////////////////////
        //SA2
        std::cout << "#" << std::endl;
        std::cout << "# 1 warp per node, merge based intersection" << std::endl;
        for(int j = 0; j < blockSizes.size(); j++){
            double sa2_time = 0;
            cudaEvent_t sa2start, sa2stop;
            cudaEventCreate(&sa2start);
            cudaEventCreate(&sa2stop);

            cudaStream_t s2;
            cudaStreamCreate(&s2);
            cudaMemPrefetchAsync(edges2,graph.E*sizeof(T),0,s2);

            int threadsPerBlock = blockSizes.at(j);
            T blocksPerGrid = (graph.N*32 + threadsPerBlock - 1) / threadsPerBlock;
            for(int k = 0; k < 5; k++){
                cudaMallocManaged(&sum, graph.N*sizeof(T));
                for(unsigned int i = 0; i < graph.N; i++){
                    sum[i] = 0;
                }
                cudaEventRecord(sa2start);
                sa2<<<blocksPerGrid, threadsPerBlock>>>(graph.N, graph.E, nodes, edges2, sum); 
                cudaEventRecord(sa2stop);
                cudaDeviceSynchronize();
                cudaEventSynchronize(sa2stop);
                u_int64_t resgpu = 0;
                for(T i = 0; i < graph.N; i++){
                    resgpu += sum[i];
                }
                assert(resgpu==graph.res);
                float sa2kerneltime = 0;
                cudaEventElapsedTime(&sa2kerneltime, sa2start, sa2stop);
                sa2kerneltime /= 1000.0;
                sa2_time += sa2kerneltime;
                cudaFree(sum);
            }
            sa2_time /= 5.0;
            std::cout << "# " << threadsPerBlock << " threads per block: " << sa2_time << "s, Speedup: " << graph.time/sa2_time <<
             "x, theoretical occupancy: " << occupancies.at(j) << std::endl;
        }


        ///////////////////////////////////////////////////////////////////
        //SA3
        std::cout << "#" << std::endl;
        std::cout << "# 1 warp per node, parallel binary search based intersection" << std::endl;
        for(int j = 0; j < blockSizes.size(); j++){
            double sa3_time = 0;
            cudaEvent_t sa3start, sa3stop;
            cudaEventCreate(&sa3start);
            cudaEventCreate(&sa3stop);

            cudaStream_t s3;
            cudaStreamCreate(&s3);
            cudaMemPrefetchAsync(edges2,graph.E*sizeof(T),0,s3);

            int threadsPerBlock = blockSizes.at(j);
            T blocksPerGrid = (graph.N*32 + threadsPerBlock - 1) / threadsPerBlock;
            for(int k = 0; k < 5; k++){
                cudaMallocManaged(&sum, graph.N*sizeof(T));
                for(unsigned int i = 0; i < graph.N; i++){
                    sum[i] = 0;
                }
                cudaEventRecord(sa3start);
                sa3<<<blocksPerGrid, threadsPerBlock>>>(graph.N, graph.E, nodes, edges2, sum); 
                cudaEventRecord(sa3stop);
                cudaDeviceSynchronize();
                cudaEventSynchronize(sa3stop);
                u_int64_t resgpu = 0;
                for(T i = 0; i < graph.N; i++){
                    resgpu += sum[i];
                }
                assert(resgpu==graph.res);
                float sa3kerneltime = 0;
                cudaEventElapsedTime(&sa3kerneltime, sa3start, sa3stop);
                sa3kerneltime /= 1000.0;
                sa3_time += sa3kerneltime;
                cudaFree(sum);
            }
            sa3_time /= 5.0;
            std::cout << "# " << threadsPerBlock << " threads per block: " << sa3_time << "s, Speedup: " << graph.time/sa3_time <<
             "x, theoretical occupancy: " << occupancies.at(j) << std::endl;
        }


        ///////////////////////////////////////////////////////////////////
        //SA4
        std::cout << "#" << std::endl;
        std::cout << "# 1 warp per edge, parallel binary search based intersection" << std::endl;
        for(int j = 0; j < blockSizes.size(); j++){
            double sa4_time = 0;
            cudaEvent_t sa4start, sa4stop;
            cudaEventCreate(&sa4start);
            cudaEventCreate(&sa4stop);

            cudaStream_t s4;
            cudaStreamCreate(&s4);
            cudaMemPrefetchAsync(edges2,graph.E*sizeof(T),0,s4);

            int threadsPerBlock = blockSizes.at(j);
            T blocksPerGrid = (graph.E*32ULL + threadsPerBlock - 1) / threadsPerBlock;
            for(int k = 0; k < 5; k++){
                cudaMallocManaged(&sum, graph.E*sizeof(T));
                for(unsigned int i = 0; i < graph.E; i++){
                    sum[i] = 0;
                }
                cudaEventRecord(sa4start);
                sa1<<<blocksPerGrid, threadsPerBlock>>>(graph.N, graph.E, nodes, edges1, edges2, sum); 
                cudaEventRecord(sa4stop);
                cudaDeviceSynchronize();
                cudaEventSynchronize(sa4stop);
                u_int64_t resgpu = 0;
                for(T i = 0; i < graph.E; i++){
                    resgpu += sum[i];
                }
                assert(resgpu==graph.res);
                float sa4kerneltime = 0;
                cudaEventElapsedTime(&sa4kerneltime, sa4start, sa4stop);
                sa4kerneltime /= 1000.0;
                sa4_time += sa4kerneltime;
                cudaFree(sum);
            }
            sa4_time /= 5.0;
            std::cout << "# " << threadsPerBlock << " threads per block: " << sa4_time << "s, Speedup: " << graph.time/sa4_time <<
             "x, theoretical occupancy: " << occupancies.at(j) << std::endl;
        }

        ///////////////////////////////////////////////////////////////////

        cudaFree(edges1);
        cudaFree(edges2);
        cudaFree(nodes);
    }
}
