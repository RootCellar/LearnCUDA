#include <stdio.h>
#include <math.h>

__global__
void isPrime(int n, float a, int *x)
{
  // Find which number we are checking
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  // Assume it is prime
  x[j]=1;

  // Find a case that means it isn't prime
  for(int i=2; i <= sqrtf(j); i++) {
        if(j%i == 0) {
          x[j]=0;
          return;
        }
  }

}

int main(void)
{
  //Just over half a billion primes, consumes ~2.5 GB VRAM
  int N = 1<<29;

  // Pointers
  int *x, *d_x;

  // List of primes in RAM
  x = (int*)malloc(N*sizeof(int));

  // Same list in VRAM
  cudaMalloc(&d_x, N*sizeof(int));

  // initialize list
  for (int i = 0; i < N; i++) {
    x[i] = 1;
  }

  // Copy host list to VRAM list
  cudaMemcpy(d_x, x, N*sizeof(int), cudaMemcpyHostToDevice);

  // Run the calculation
  isPrime<<<N/128, 128>>>(N, 2.0f, d_x);

  // Copy results back to host RAM
  cudaMemcpy(x, d_x, N*sizeof(int), cudaMemcpyDeviceToHost);

  // Display results
  for(int i=0; i < N; i++) {
    if(x[i] == 1) printf("%d\n", i);
  }

  // Cleanup
  cudaFree(d_x);
  free(x);
}
