#define T u_int32_t

__global__ void db(int n1, int n2, T *sum, T *g){
    T tid = blockDim.x * blockIdx.x + threadIdx.x;
    T u = tid / 32;
    T lid = tid % 32;
    T u_neigh = n2*u;
    T count = 0;
    for(unsigned int i = 0; i < n2; i++){
        T tmp1 = g[u_neigh+i];
        T tmp2 = (1ULL << lid);
        T tmp3 = tmp1 & tmp2;
        T v = 32*i+lid;
        T v_neigh = n2*v;
        if(tmp2 == tmp3){
            for(unsigned int k = 0; k < n2; k++){
                count += __popcll(g[u_neigh+k] & g[v_neigh+k]);
            }
        }
    }
    for(int offset = 16; offset > 0; offset /= 2){
        count += __shfl_down_sync(0xffffffff, count, offset);
    }
    sum[u] = count;
}

u_int64_t launchDense(GraphReader graph){
    T *sum;
    cudaMallocManaged(&sum, graph.n1*sizeof(T));
    for(unsigned int i = 0; i < graph.n1; i++){
        sum[i] = 0;
    }
    unsigned int threadsPerBlock = 128;
    unsigned int blocksPerGrid = (32*graph.n1 + threadsPerBlock - 1) / threadsPerBlock;
    db<<<blocksPerGrid, threadsPerBlock>>>(graph.n1, graph.n2, sum, graph.g_dense);
    cudaDeviceSynchronize();
    u_int64_t res = 0;
    for(unsigned int i = 0; i < graph.n1; i++){
        res += sum[i];
    }
    return res;
}