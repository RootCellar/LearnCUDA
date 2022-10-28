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
void foo(void) {
	int wid=512,ht=512;
	
	// Make space for output data
	gpu_vec<pixel> gpu_image(wid*ht); 

	// Render on GPU
	double start=time_in_seconds();
	
	draw_image<<<wid,ht>>>(gpu_image,wid,ht);
	gpu_check(cudaGetLastError());
	gpu_check(cudaDeviceSynchronize());

	double elapsed=time_in_seconds()-start;
	printf("render: %.4f ns/pixel\n", elapsed*1.0e9/(wid*ht));
	
	// Copy rendered image back to CPU
	std::vector<pixel> img; 
	gpu_image.copy_to(img);

	// Copy the image data back to the CPU, and write to file
	std::ofstream imgfile("out.ppm",std::ios_base::binary);
	imgfile<<"P6\n"<<wid<<" "<<ht<<" 255\n";
	imgfile.write((char *)&img[0],wid*ht*sizeof(pixel));
}
