#include "assert.h"
#include "graphreader.h"
#include "sparse.cuh"
#include "dense.cuh"
#include <iostream>
#include <chrono>
#include <vector>
#include <set>
#include <algorithm>
#include <fstream>

#define T u_int32_t

int main(){
    auto readStart = std::chrono::steady_clock::now();
    GraphReader graph = BIOSCGT;
    graph.read();
    auto readEnd = std::chrono::steady_clock::now();
    std::chrono::duration<double> readTime = readEnd-readStart;
    std::cout << "READ TIME:      " << readTime.count() << "s" << std::endl;
    int alg = graph.alg;
    if(alg == 0){
        auto start = std::chrono::steady_clock::now();
        u_int64_t res = launchSa1(graph);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time = end-start;
        std::cout << "RUN TIME:       " << time.count() << "s" << std::endl;
        std::cout << "TRIANGLE COUNT: " << res << std::endl;
    }
    else if(alg == 1){
        auto start = std::chrono::steady_clock::now();
        u_int64_t res = launchSa2(graph);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time = end-start;
        std::cout << "RUN TIME:       " << time.count() << "s" << std::endl;
        std::cout << "TRIANGLE COUNT: " << res << std::endl;
    }
    else if(alg == 2){
        auto start = std::chrono::steady_clock::now();
        u_int64_t res = launchSa3(graph);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time = end-start;
        std::cout << "RUN TIME:       " << time.count() << "s" << std::endl;
        std::cout << "TRIANGLE COUNT: " << res << std::endl;
    }
    else if(alg == 3){
        auto start = std::chrono::steady_clock::now();
        u_int64_t res = launchSa4(graph);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time = end-start;
        std::cout << "RUN TIME:       " << time.count() << "s" << std::endl;
        std::cout << "TRIANGLE COUNT: " << res << std::endl;
    }
    else if(alg == 4){
        //hybrid
        auto start = std::chrono::steady_clock::now();
        u_int64_t res = launchSa1(graph);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time = end-start;
        std::cout << "RUN TIME:       " << time.count() << "s" << std::endl;
        std::cout << "TRIANGLE COUNT: " << res << std::endl;
    }
    else if(alg == 5){
        auto start = std::chrono::steady_clock::now();
        u_int64_t res = launchDense(graph);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time = end-start;
        std::cout << "RUN TIME:       " << time.count() << "s" << std::endl;
        std::cout << "TRIANGLE COUNT: " << res << std::endl;
    }
    else{
        std::cout << "error" << std::endl;
    }
}