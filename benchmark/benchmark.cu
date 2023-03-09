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
#define BLOCKS 128*1000

/*
 *
 * BENCHMARK(function, values, "bench") will expand to the following:
 *

    do {
        int runCount = 0;
        clock_t start_time = clock();
        clock_t time_now;
        while( (time_now = clock() ) - start_time < CLOCKS_PER_SEC * SECONDS_PER_RUN) {
            function<<<BLOCKS/128, 128>>>(values);
            cudaDeviceSynchronize();
            runCount++;
        }

        float seconds = (float) (time_now - start_time) / CLOCKS_PER_SEC;
        debug_printf("bench: %d over %f seconds\n", runCount, seconds);
    } while(0)

 *
 * This makes benchmarking much easier. The do-while loop surrounds the whole thing just make sure
 * it is in it's own code block and won't behave weirdly inside of other blocks (for loops, etc)
 *
*/

#define BENCHMARK(x, vals, name) do { int runCount = 0; clock_t start_time = clock(); clock_t time_now;\
    while( ( time_now = clock() ) - start_time < CLOCKS_PER_SEC * SECONDS_PER_RUN) { x<<<BLOCKS/128, 128>>>(vals); cudaDeviceSynchronize(); runCount++; }\
    float seconds = (float) (time_now - start_time) / CLOCKS_PER_SEC; debug_printf(name ": %d over %f seconds\n", runCount, seconds); } while(0)

__global__ void benchFloats(float* floats) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for(float i = 0; i < 1000000; i+=1) {
        float x = 5.5 * i * i;
        float y = 5.5;
        float z = x * y;
        floats[j] = z;
    }

}

__global__ void benchInts(int* ints) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = 0; i < 1000000; i+=1) {
        int x = 5 * i * i;
        int y = 5;
        int z = x * y;
        ints[j] = z;
    }

}

__global__ void benchDoubles(double* doubles) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for(double i = 0; i < 1000000; i+=1) {
        double x = 5 * i * i;
        double y = 5;
        double z = x * y;
        doubles[j] = z;
    }

}

/*
void benchmark(void* func, void* values, char* name) {
    int runCount;
    clock_t start_time, time_now;
    float seconds;

    runCount = 0;
    start_time = clock();
    while( (time_now = clock() ) - start_time < CLOCKS_PER_SEC * SECONDS_PER_RUN) {
        (**func)<<<BLOCKS/128, 128>>>(values);
        runCount++;
    }

    seconds = (float) (time_now - start_time) / CLOCKS_PER_SEC;
    debug_printf("benchFloats: %d over %f seconds\n", runCount, seconds);
}
*/

int main(int argc, char** argv) {
    clock_t start_time, time_now;
    int runCount;
    float seconds;

    // For both mallocs, we allocate twice the space needed for an array of floats
    // because we will also benchmark doubles, which are twice the size

    float* floats = (float*) malloc(sizeof(float) * BLOCKS * 2);
    float* gpu_floats;

    cudaMalloc(&gpu_floats, sizeof(float) * BLOCKS * 2);

    if(gpu_floats == 0 || floats == 0) {
        perror("Could not allocate space for floats\n");
        exit(1);
    }

    /*
    for(int i = 0; i < BLOCKS; i++) {
        floats[i] = (float) i;
    }
    */

    runCount = 0;
    start_time = clock();
    while( (time_now = clock() ) - start_time < CLOCKS_PER_SEC * SECONDS_PER_RUN) {
        cudaMemcpy(gpu_floats, floats, sizeof(float) * BLOCKS, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        runCount++;
    }

    seconds = (float) (time_now - start_time) / CLOCKS_PER_SEC;
    debug_printf("cudaMemcpy: %d over %f seconds\n", runCount, seconds);

    debug_printf("Copied %ld bytes\n", sizeof(float) * BLOCKS);

    /*
    runCount = 0;
    start_time = clock();
    while( (time_now = clock() ) - start_time < CLOCKS_PER_SEC * SECONDS_PER_RUN) {
        benchFloats<<<BLOCKS/128, 128>>>(gpu_floats);
        runCount++;
    }

    seconds = (float) (time_now - start_time) / CLOCKS_PER_SEC;
    debug_printf("benchFloats: %d over %f seconds\n", runCount, seconds);
    */

    for(int i = 0; i < 10; i++) {
        //cudaMemcpy(gpu_floats, floats, sizeof(float) * BLOCKS, cudaMemcpyHostToDevice);
        BENCHMARK(benchFloats, gpu_floats, "benchFloats");
    }

    for(int i = 0; i < 10; i++) {
        BENCHMARK(benchInts, (int*) gpu_floats, "benchInts");
    }

    for(int i = 0; i < 10; i++) {
        BENCHMARK(benchDoubles, (double*) gpu_floats, "benchDoubles");
    }

}