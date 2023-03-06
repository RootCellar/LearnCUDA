/*
 *
 * Darian Marvel
 * 3/06/2023
 * 
 * Benchmarking different Nvidia graphics card operations
 *
 *
*/



// Includes

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>

#include "debug.h"

#define SECONDS_PER_RUN 1

__global__ void benchFloats(float* floats) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = 0; i < 100; i++) {
        floats[j] *= 6.7;
        floats[j] *= 6.7;
        floats[j] *= 6.7;
        floats[j] *= 6.7;
        floats[j] *= 6.7;
        floats[j] *= 6.7;
        floats[j] *= 6.7;
        floats[j] *= 6.7;
        floats[j] *= 6.7;
        floats[j] *= 6.7;
    }

}

int main(int argc, char** argv) {
    int blocks = 128*1000;

    clock_t start_time, time_now;
    long runCount = 0;

    float* floats = (float*) malloc(sizeof(float) * blocks);
    float* gpu_floats;

    cudaMalloc(&gpu_floats, sizeof(float) * blocks);

    if(gpu_floats == 0 || floats == 0) {
        perror("Could not allocate space for floats\n");
        exit(1);
    }

    for(int i = 0; i < blocks; i++) {
        floats[i] = (float) i;
    }

    start_time = clock();
    while(clock() - start_time < CLOCKS_PER_SEC * SECONDS_PER_RUN) {
        benchFloats<<<blocks/128, 128>>>(gpu_floats);
        runCount++;
    }

    debug_printf("benchFloats: %ld \n", runCount);

}