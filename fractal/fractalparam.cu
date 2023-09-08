/*
 * RootCellar
 * Based on code by Dr. Orion Lawlor
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define DEBUG 1

#define debug_print(fmt, ...) \
        do { if (DEBUG) fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, \
                                __LINE__, __func__, __VA_ARGS__); } while (0)

/* Class to represent one image pixel */
struct pixel {
	unsigned char r,g,b;
};

/* GPU Code to render mandelbrot set fractal */
__global__ void draw_image(pixel *image,int wid,int ht, float param1, float param2) {
	//for (int i=0;i<wid*ht;i++) // loop over all the pixels
	long i = threadIdx.x + blockIdx.x * blockDim.x;
	{
		int x=i%wid, y=i/wid;
		float fx=x*(1.0/wid), fy=y*(1.0/ht);
		float scale=1; // amount of the mandelbrot set to draw
		fx*=scale; fy*=scale;

		float ci=fy, cr=fx; // complex constant: x,y coordinates
		float zi=ci, zr=cr; // complex number to iterate
		int iter;
		float zr_new, zi_new;
		for (iter=0;iter<2500;iter++) {
			if (zi*zi+zr*zr>4.0) break; // number too big--stop iterating
			// z = z*z + c
			zr_new=zr*zr*param1-zi*zi*param2+cr;
			zi_new=2*zr*zi+ci;
			zr=zr_new; zi=zi_new;
		}

		image[i].r=zr*255/4.0;
		image[i].g=zi*255/4.0;
		image[i].b=iter;
	}
}

int makeImage(char*, float, float);

int main(int argc, char** argv) {

	if(argc < 4) {
		printf("Usage: %s <number> <number> <file name>\n", argv[0]);
		return -1;
	}

	float param1 = 0, param2 = 0;
	sscanf(argv[1], "%f", &param1);
	sscanf(argv[2], "%f", &param2);

	makeImage(argv[3], param1, param2);
}

/* Run on CPU */
int makeImage(char* fileName, float param1, float param2) {
	debug_print("Param 1: %f\n", param1);
	debug_print("Param 2: %f\n", param2);

	int wid=16384,ht=16384;

	pixel* pixels;
	pixels = (pixel*) malloc(wid*ht*sizeof(pixel));
	if(pixels == 0) {
        printf("Could not allocate memory for pixels!\n");
        exit(1);
    }

	pixel* gpu_pixels;
	cudaMalloc(&gpu_pixels, wid*ht*sizeof(pixel));
	if(gpu_pixels == 0) {
        printf("Could not allocate GPU memory for pixels!\n");
        exit(1);
    }

	printf("Beginning to draw...\n");
	draw_image<<<(wid*ht)/512,512>>>(gpu_pixels,wid,ht,param1, param2);
	printf("Drawing called.\n");

	// Note: It turns out that any code in this space CAN be executed while
	// the graphics card is still working - this section won't be stopped
	// until we need the GPU again
	//printf("Test\n");

	cudaMemcpy(pixels, gpu_pixels, wid*ht*sizeof(pixel), cudaMemcpyDeviceToHost);
	printf("Drawing finished.\n");

	FILE *file = fopen(fileName,"wb");
	fprintf(file, "P6\n%d %d\n255\n",wid, ht);
	static unsigned char color[3];
	for(int i=0; i<wid*ht; i++) {
    color[0] = pixels[i].r;
    color[1] = pixels[i].g;
    color[2] = pixels[i].b;
    fwrite(color, 1, 3, file);
	}
	fclose(file);
	return 1;
}
