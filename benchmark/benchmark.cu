/*
 *
 * Darian Marvel
 * 3/06/2023
 * 
 * Benchmarking different Nvidia graphics card operations
 * for CS241 Computer Hardware Concepts
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
#include <locale.h>

#include "debug.h"

#define BLOCKS (128*1000)
#define ITERATIONS_PER_OP (1000000)

#define OPS_PER_RUN ( (long) BLOCKS * ITERATIONS_PER_OP * 2 )

#define BENCHMARK_TIMES (10)
#define SECONDS_PER_RUN (1)

/*
 *
 * BENCHMARK(function, values, "bench") will expand to the following:
 *

    do {
        // How many times the function was run in the elapsed time
        int runCount = 0;

        // For keeping track of how long we've spent running this
        clock_t start_time = clock();
        clock_t time_now;

        // Call the function over and over until at least one second has elapsed
        while( (time_now = clock() ) - start_time < CLOCKS_PER_SEC * SECONDS_PER_RUN) {

            function<<<BLOCKS/128, 128>>>(values); // Run the function we are benchmarking

            cudaDeviceSynchronize(); // Wait for the device to finish that before we move on

            runCount++;

        }

        // Calculate how long this actually took and print the results
        float seconds = (float) (time_now - start_time) / CLOCKS_PER_SEC;
        debug_printf("bench: %d over %f seconds (%'.2f operations per second)\n", runCount, seconds, (float) OPS_PER_RUN * ((float)runCount/seconds));
    } while(0)

 *
 * This makes benchmarking much easier. The do-while loop surrounds the whole thing just make sure
 * it is in its own code block and won't behave weirdly inside of other blocks (for loops, etc)
 *
*/

#define BENCHMARK(x, vals, name) do { int runCount = 0; clock_t start_time = clock(); clock_t time_now;\
    while( ( time_now = clock() ) - start_time < CLOCKS_PER_SEC * SECONDS_PER_RUN) { x<<<BLOCKS/128, 128>>>(vals); cudaDeviceSynchronize(); runCount++; }\
    float seconds = (float) (time_now - start_time) / CLOCKS_PER_SEC; \
    debug_printf(name ": %'d over %'f seconds (%'.2f operations per second)\n", runCount, seconds, (float) OPS_PER_RUN * ((float)runCount/seconds)); } while(0)


// Do whatever x is the given number of times
#define TIMES(x, times) do { for(int i = 0; i < times; i++) x; } while(0)


/*
 * Performs 1 million iterations per thread, 4 floating point operations each iteration (theoretically)
 * and of course checks to see if the loop is done yet each iteration
 *
 * On my GTX 1660 Ti, this runs 23 times in 1.035 seconds on average.
 * (23/1.035) times per second * 1,000,000 iterations * 4 operations per iteration * 128*1000 threads
 * = 11,377,777,777,778 FLOPS (theoretically)
 * = 11.378 TFLOPS (theoretically)
 * 
 * The interesting thing to note is that websites such as techpowerup which calculate theoretical TFLOPS
 * come up with only about 4.9 TFLOPS.
 * 
 * This indicates significant compiler optimization. Also, my GPU actually operates at a higher clock rate than
 * techpowerup claims.
 *
*/

/*
 * Each of these benchmarking functions looks just a little bit different from the others - why?
 * 
 * The compiler seems to optimize the code out in different circumstances for each data type.
*/

__global__ void benchFloats(float* floats) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for(float i = 0; i < ITERATIONS_PER_OP; i+=1) {
        float x = 5.5 * i * i;
        float y = 5.5;
        float z = x * y;
        if(j < 0) z *= 2.0;
        floats[j] = z;
    }

}

__global__ void benchInts(int* ints) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = 0; i < ITERATIONS_PER_OP; i+=1) {
        int x = 5 * i * i;
        int y = 5;
        int z = x * y;
        if(j < 0) z *= 2;
        if(j < 0) ints[j] = z;
    }

}

/*
 * Performs 1 million iterations per thread, 4 double-precision floating point operations each iteration (theoretically)
 * and of course checks to see if the loop is done yet each iteration
 *
 * On my GTX 1660 Ti, this runs 1 time in 1.7 seconds on average.
 * (1/1.7) times per second * 1,000,000 iterations * 4 operations per iteration * 128*1000 threads
 * = 301,176,470,588 FLOPS (theoretically)
 * = 301.176 GFLOPS (theoretically)
 * 
 * Techpowerup claims a GTX 1660 Ti mobile can do 152 GFLOPS
 *
 * This is likely compiler optimization at work again.
 * However, I still notice that this is WAAAY slower than regular single-precision floating-point math operations.
 *
*/
__global__ void benchDoubles(double* doubles) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for(double i = 0; i < ITERATIONS_PER_OP; i+=1) {
        double x = 5.5 * i * i;
        double y = 5.5;
        double z = x * y;
        doubles[j] = z;
    }

}

__global__ void benchLongs(long* longs) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for(double i = 0; i < ITERATIONS_PER_OP; i+=1) {
        long x = 5 * i * i;
        long y = 5;
        long z = x * y;
        longs[j] = z;
    }

}

int main(int argc, char** argv) {

    // set the locale so that printed numbers can be formatted
    // with commas separating the digits
    setlocale(LC_NUMERIC, "");

    clock_t start_time, time_now;
    int runCount;
    float seconds;

    debug_printf("%'d blocks, %'d iterations per operation\n", BLOCKS, ITERATIONS_PER_OP);
    debug_printf("Each run of each operation performs %'ld computations\n", OPS_PER_RUN);

    // For both mallocs, we allocate twice the space needed for an array of floats
    // because we will also benchmark doubles and longs, which are twice the size

    float* floats = (float*) malloc(sizeof(float) * BLOCKS * 2);
    float* gpu_floats;

    cudaMalloc(&gpu_floats, sizeof(float) * BLOCKS * 2);

    if(gpu_floats == 0 || floats == 0) {
        perror("Could not allocate space for floats\n");
        exit(1);
    }

    // Benchmark RAM to VRAM memory transfer speeds

    runCount = 0;
    start_time = clock();
    while( (time_now = clock() ) - start_time < CLOCKS_PER_SEC * SECONDS_PER_RUN * 3) {
        cudaMemcpy(gpu_floats, floats, sizeof(float) * BLOCKS, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        runCount++;
    }

    seconds = (float) (time_now - start_time) / CLOCKS_PER_SEC;
    float bytes_per_sec = (float) runCount * (sizeof(float) * BLOCKS) / seconds;
    printf("\n");
    debug_printf("cudaMemcpy: %'d over %'f seconds\n", runCount, seconds);

    debug_printf("%'ld bytes * %'d times in %'f seconds = \n", sizeof(float) * BLOCKS, runCount, seconds);
    debug_printf("%'.2f bytes per second\n\n", bytes_per_sec);

    // End memory transfer benchmark

    // Benchmark GPU operations

    TIMES( BENCHMARK(benchFloats, gpu_floats, "benchFloats"), BENCHMARK_TIMES);
    TIMES( BENCHMARK(benchInts, (int*) gpu_floats, "benchInts"), BENCHMARK_TIMES);
    TIMES( BENCHMARK(benchDoubles, (double*) gpu_floats, "benchDoubles"), BENCHMARK_TIMES);
    TIMES( BENCHMARK(benchLongs, (long*) gpu_floats, "benchLongs"), BENCHMARK_TIMES);

}

