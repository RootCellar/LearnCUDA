/*
 * RootCellar
 * 10/17/2022
*/

#include <stdio.h>
#include <math.h>

#define DEBUG 1

/*
 * Useful debug function define I found online
*/

#define debug_printf(fmt, ...) \
        do { if (DEBUG) fprintf(stderr, "%s:%d:%s(): " fmt, __FILE__, \
                                __LINE__, __func__, __VA_ARGS__); } while (0)

#define debug_print(fmt) \
        do { if (DEBUG) fprintf(stderr, "%s:%d:%s(): " fmt, __FILE__, \
                                __LINE__, __func__); } while (0)


void findPrimes(int countScale, int sectionScale);

/*
 * nums: pointer to the array of numbers in graphics card memory
 * x: array of trues and falses for each number (whether or not they are prime)
*/
__global__ void isPrime(int* x, int* nums)
{
  // Find which number we are checking
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  // Assume it is prime
  x[j] = 1;

  // Find a case that means it isn't prime

  if(nums[j] < 2) {
    x[j] = 0;
    return;
  }

  if(nums[j]%2 == 0) {
    x[j] = 0;
    return;
  }

  int num_stop = sqrtf(nums[j]);

  for(int i = 3; i <= num_stop; i += 2) {
        if( nums[j] % i == 0 ) {
          // The number this thread is checking is not prime
          x[j] = 0;
          return;
        }
  }

}

// PSEUDOCODE:

/*

  Divide up the numbers to check for prime-ness into smaller sets
  based on command line arguments
  (This allows GPUs with low VRAM to be able to check just as many
  numbers as GPUs with large VRAM)

  for(each set)
    setup array in RAM
    copy array to VRAM
    find the primes
    copy back to RAM
    print

*/

int main(int argc, char** argv)
{
  int param1 = 5, param2 = 3;
  if(argc >= 2) {
    sscanf(argv[1], "%d", &param1);
  }
  if(argc >= 3) {
    sscanf(argv[2], "%d", &param2);
  }

  debug_printf("%d %d\n", param1, param2);

  if(param2 > param1) {
    printf("2nd parameter cannot be larger than 1st parameter\n");
    exit(1);
  }

  findPrimes(param1, param2);
}

void findPrimes(int countScale, int sectionScale)
{
  // Just over half a billion primes at 1<<29, consumes ~5 GB VRAM
  // if done in one pass
  int N_total = 1<<countScale; // the number we will search up to
  //int N_total = 100000000;
  int N = N_total / (1<<sectionScale); // how many per pass
  int previous_max = 0;

  debug_printf("%d numbers total, %d numbers per pass\n", N_total, N);

  // Pointers
  int *x, *d_x;
  int *nums, *gpu_nums;

  debug_print("Deciding block size...\n");
  int blockSize = 64;
  if( N < 4096 ) {
    blockSize = 1;
  }

  // List of primes in RAM
  x = (int*) malloc(N * sizeof(int));
  nums = (int*) malloc(N * sizeof(int));

  if(x == 0 || nums == 0) {
    debug_print("Could not allocate RAM!\n");
    exit(1);
  }

  // Same list in VRAM
  cudaError_t error1 = cudaMalloc(&d_x, N * sizeof(int));
  cudaError_t error2 = cudaMalloc(&gpu_nums, N * sizeof(int));

  if(error1 != cudaSuccess || error2 != cudaSuccess) {
    debug_print("Could not allocate VRAM!\n");
    exit(1);
  }

  // Loop over each subset that the whole range to check is divided into
  while(previous_max < N_total) {

    debug_printf("Handling %d to %d\n", previous_max, previous_max + N);

    // initialize list
    debug_print("Making list...\n");
    for (int i = 0; i < N; i++) {
      nums[i] = i + previous_max;
    }

    // Copy host list to VRAM list
    debug_print("Copying to VRAM..\n");
    cudaMemcpy(gpu_nums, nums, N * sizeof(int), cudaMemcpyHostToDevice);

    // Run the calculation
    debug_print("Calculating...\n");
    isPrime<<<N/blockSize, blockSize>>>(d_x, gpu_nums);

    // Copy results back to host RAM
    cudaMemcpy(x, d_x, N * sizeof(int), cudaMemcpyDeviceToHost);
    debug_print("Copied back to RAM\n");

    // Display results
    debug_print("Displaying results...\n");
    for(int i = 0; i < N; i++) {
      if(x[i] == 1) printf("%d\n", nums[i]);
    }

    previous_max = previous_max + N;

  }

  // Cleanup
  cudaFree(d_x);
  cudaFree(gpu_nums);
  free(x);
  free(nums);
}
