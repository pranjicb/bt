#include "assert.h"
#include "graphreader.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <set>
#include <algorithm>
#include <fstream>
#include <sparse.h>
#include <dense.h>
#define T u_int32_t

int main(){
    GraphReader graph = BIOHUMANGENE2;
    graph.read();
    if(graph.alg == 5){
        //DENSE IMPLEMENTATION
    }
    else{
        //SPARSE IMPLEMENTATION
    }
}