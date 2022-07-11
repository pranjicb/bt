#include "assert.h"
#include "graphreader.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <set>
#include <algorithm>
#include <fstream>

#define T u_int32_t

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

int main(){
    
}