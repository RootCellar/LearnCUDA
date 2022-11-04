/*
 * RootCellar
 * 10/17/2022
*/

#include <stdio.h>
#include <math.h>

#define DEBUG 1

#define debug_print(fmt, ...) \
        do { if (DEBUG) fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, \
                                __LINE__, __func__, __VA_ARGS__); } while (0)

__global__
void isPrime(int n, float a, int *x, int *nums)
{
  // Find which number we are checking
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  // Assume it is prime
  x[j]=1;

  if(nums[j] < 2) {
    x[j] = 0;
    return;
  }

  // Find a case that means it isn't prime
  for(int i=2; i <= sqrtf(nums[j]); i++) {
        if(nums[j]%i == 0) {
          x[j]=0;
          return;
        }
  }

}

// TODO: for large number of primes, cycle through sets of them at a set size
// so that any number of primes can be found (assuming large hard drive space)
// on any RAM or VRAM size

// PSEUDOCODE:

/*

  for(each set)
    setup array in RAM
    copy array to VRAM
    find the primes
    copy back to RAM
    write to file

*/

int main(void)
{
  // Just over half a billion primes at 1<<29, consumes ~2.5 GB VRAM
  // if done in one pass
  int N_total = 1<<30; // the number we will search up to
  int N = N_total/(1<<4); // how many per pass
  int previous_max = 0;

  debug_print("%d primes total, %d primes per pass\n", N_total, N);

  // Pointers
  int *x, *d_x;
  int *nums, *gpu_nums;

  // List of primes in RAM
  x = (int*)malloc(N*sizeof(int));
  nums = (int*) malloc(N*sizeof(int));

  // Same list in VRAM
  cudaMalloc(&d_x, N*sizeof(int));
  cudaMalloc(&gpu_nums, N*sizeof(int));

  while(previous_max < N_total) {

    debug_print("Handling %d to %d\n", previous_max, previous_max + N);

    /*

    // initialize list
    for (int i = 0; i < N; i++) {
      x[i] = 1;
      nums[i] = i+previous_max;
    }

    // Copy host list to VRAM list
    cudaMemcpy(d_x, x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_nums, nums, N*sizeof(int), cudaMemcpyHostToDevice);

    // Run the calculation
    isPrime<<<N/128, 128>>>(N, 2.0f, d_x, gpu_nums);

    // Copy results back to host RAM
    cudaMemcpy(x, d_x, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nums, gpu_nums, N*sizeof(int), cudaMemcpyDeviceToHost);

    // Display results
    for(int i=0; i < N; i++) {
      if(x[i] == 1) printf("%d\n", i);
    }

    */

    previous_max = previous_max + N;

  }

  // Cleanup
  cudaFree(d_x);
  cudaFree(gpu_nums);
  free(x);
  free(nums);
}
