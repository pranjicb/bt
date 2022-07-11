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