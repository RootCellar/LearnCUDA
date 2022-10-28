#include <stdlib.h>
#include <stdio.h>

/* Class to represent one image pixel */
struct pixel {
	unsigned char r,g,b;
};

/* GPU Code to render mandelbrot set fractal */
__global__ void draw_image(pixel *image,int wid,int ht) {
	//for (int i=0;i<wid*ht;i++) // loop over all the pixels
	long i = threadIdx.x + blockIdx.x * blockDim.x;
	{
		int x=i%wid, y=i/wid;
		float fx=x*(1.0/wid), fy=y*(1.0/ht);
		float scale=1.0; // amount of the mandelbrot set to draw
		fx*=scale; fy*=scale;

		float ci=fy, cr=fx; // complex constant: x,y coordinates
		float zi=ci, zr=cr; // complex number to iterate
		int iter;
		for (iter=0;iter<100;iter++) {
			if (zi*zi+zr*zr>4.0) break; // number too big--stop iterating
			// z = z*z + c
			float zr_new=zr*zr-zi*zi+cr;
			float zi_new=2*zr*zi+ci;
			zr=zr_new; zi=zi_new;
		}

		image[i].r=zr*255/4.0;
		image[i].g=zi*255/4.0;
		image[i].b=iter;
	}
}

/* Run on CPU */
int main(void) {
	int wid=512,ht=512;

	pixel* pixels;
	pixels = (pixel*) malloc(wid*ht*sizeof(pixel));

	pixel* gpu_pixels;
	cudaMalloc(&gpu_pixels, wid*ht*sizeof(pixel));

	draw_image<<<wid,ht>>>(gpu_pixels,wid,ht);

	cudaMemcpy(pixels, gpu_pixels, wid*ht*sizeof(pixel), cudaMemcpyDeviceToHost);

	FILE *file = fopen("image.ppm","wb");
	fprintf(file, "P6\n%d %d\n255\n",wid, ht);
	for(int i=0; i<wid*ht; i++) {
		static unsigned char color[3];
    color[0] = pixels[i].r;
    color[1] = pixels[i].g;
    color[2] = pixels[i].b;
    fwrite(color, 1, 3, file);
	}
	fclose(file);

}
