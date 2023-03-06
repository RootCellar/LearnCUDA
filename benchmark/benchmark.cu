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

#define BENCHMARK(x, vals) do { int runCount = 0; clock_t start_time = clock();\
    while( (time_now = clock() ) - start_time < CLOCKS_PER_SEC * SECONDS_PER_RUN) { x<<<BLOCKS/128, 128>>>(vals); runCount++; } } while(0)

__global__ void benchFloats(float* floats) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = 0; i < 1000; i++) {
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

    float* floats = (float*) malloc(sizeof(float) * BLOCKS);
    float* gpu_floats;

    cudaMalloc(&gpu_floats, sizeof(float) * BLOCKS);

    if(gpu_floats == 0 || floats == 0) {
        perror("Could not allocate space for floats\n");
        exit(1);
    }

    for(int i = 0; i < BLOCKS; i++) {
        floats[i] = (float) i;
    }

    runCount = 0;
    start_time = clock();
    while( (time_now = clock() ) - start_time < CLOCKS_PER_SEC * SECONDS_PER_RUN) {
        cudaMemcpy(gpu_floats, floats, sizeof(float) * BLOCKS, cudaMemcpyHostToDevice);
        runCount++;
    }

    seconds = (float) (time_now - start_time) / CLOCKS_PER_SEC;
    debug_printf("cudaMemcpy: %d over %f seconds\n", runCount, seconds);

    debug_printf("Copied %ld bytes\n", sizeof(float) * BLOCKS);

    runCount = 0;
    start_time = clock();
    while( (time_now = clock() ) - start_time < CLOCKS_PER_SEC * SECONDS_PER_RUN) {
        benchFloats<<<BLOCKS/128, 128>>>(gpu_floats);
        runCount++;
    }

    seconds = (float) (time_now - start_time) / CLOCKS_PER_SEC;
    debug_printf("benchFloats: %d over %f seconds\n", runCount, seconds);

    BENCHMARK(benchFloats, gpu_floats);

}