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
__global__ void draw_image(pixel *image,int wid,int ht, float param) {
	//for (int i=0;i<wid*ht;i++) // loop over all the pixels
	long i = threadIdx.x + blockIdx.x * blockDim.x;
	{
		int x=i%wid, y=i/wid;
		float fx=x*(1.0/wid), fy=y*(1.0/ht);
		float scale=0.8; // amount of the mandelbrot set to draw
		fx*=scale; fy*=scale;

		float ci=fy, cr=fx; // complex constant: x,y coordinates
		float zi=ci, zr=cr; // complex number to iterate
		int iter;
		float zr_new, zi_new;
		for (iter=0;iter<2500;iter++) {
			if (zi*zi+zr*zr>4.0) break; // number too big--stop iterating
			// z = z*z + c
			zr_new=zr*zr*param-zi*zi*param+cr;
			zi_new=2*zr*zi+ci;
			zr=zr_new; zi=zi_new;
		}

		image[i].r=zr*255/4.0;
		image[i].g=zi*255/4.0;
		image[i].b=iter;
	}
}

int makeImage(char*, float);

int main(int argc, char** argv) {

	if(argc < 3) {
		printf("Usage: %s <number> <file name>\n", argv[0]);
		return -1;
	}

	float param = 0;
	sscanf(argv[1], "%f", &param);

	makeImage(argv[2], param);
}

/* Run on CPU */
int makeImage(char* fileName, float param) {
	debug_print("Param: %f\n", param);

	int wid=16384,ht=16384;

	pixel* pixels;
	pixels = (pixel*) malloc(wid*ht*sizeof(pixel));

	pixel* gpu_pixels;
	cudaMalloc(&gpu_pixels, wid*ht*sizeof(pixel));

	printf("Beginning to draw...\n");
	draw_image<<<(wid*ht)/512,512>>>(gpu_pixels,wid,ht,param);
	printf("Drawing called.\n");

	cudaMemcpy(pixels, gpu_pixels, wid*ht*sizeof(pixel), cudaMemcpyDeviceToHost);
	printf("Drawing finished.\n");

	FILE *file = fopen(fileName,"wb");
	fprintf(file, "P6\n%d %d\n255\n",wid, ht);
	for(int i=0; i<wid*ht; i++) {
		static unsigned char color[3];
    color[0] = pixels[i].r;
    color[1] = pixels[i].g;
    color[2] = pixels[i].b;
    fwrite(color, 1, 3, file);
	}
	fclose(file);
return 1;
}
