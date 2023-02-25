/*
 *
 * Darian Marvel
 * 2/24/2023
 * 
 * Building an n-particle/n-body simulator that runs on the graphics card
 *
 * Uses some code from Ryan Brune's Orbital-Simulator
 *
*/



// Includes

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <time.h>
#include <GL/glu.h>
#include <unistd.h>
#include <stdbool.h>

#include "debug.h"
#include "physics.h"

#define WIDTH 1600
#define HEIGHT 800

#define PI 3.14159265359

void DrawCircle(float cx, float cy, float r) {
	glBegin(GL_LINE_LOOP);
	for(float i = 0; i < 2 * PI ; i += 0.1)
	{
		float x = r * cosf(i);
        x /= WIDTH/HEIGHT;

		float y = r * sinf(i);
		glVertex2f(x + cx, y + cy);
	}
	glEnd();
}

__device__ void calcDistance(float* distance, struct particle one, struct particle two) {
    (*distance) = sqrtf( pow(one.x-two.x,2) + pow(one.y-two.y,2) );
}

__global__ void calcAcceleration(struct particle* particles, struct particle center_of_mass) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    particles[j].v_x = 1;
    particles[j].v_y = 1;

    float distance;
    calcDistance(&distance, particles[j], center_of_mass);

    particles[j].x += particles[j].v_x;
    particles[j].y += particles[j].v_y;
}

int main( int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE);    // Use single color buffer and no depth buffer.
    glutInitWindowSize(WIDTH,HEIGHT);         // Size of display area, in pixels.
    glutInitWindowPosition(0,0);     // Location of window in screen coordinates.
    glutCreateWindow("N-Particle Simulator"); // Parameter is window title.

    int particle_count = 128*20;
    struct particle* particles = (struct particle*) malloc(sizeof(particle) * particle_count);
    if(particles == 0) {
        printf("Could not allocate memory for particles!\n");
        exit(1);
    }

    struct particle* gpu_particles;
    cudaMalloc(&gpu_particles, sizeof(struct particle) * particle_count);
    if(gpu_particles == 0) {
        printf("Could not allocate GPU memory for particles!\n");
        exit(1);
    }

    struct particle center_of_mass;
    float scaled_x;
    float scaled_y;

    // Position Particles
    for(int i = 0; i < particle_count; i++) {
        particles[i].x = i % 100;
        particles[i].x *= 15;
        particles[i].y = i / 100;
        particles[i].y *= 30;
    }

    while(1) {

        // Do the physics

        calculate_center_of_mass(&center_of_mass, particles, particle_count);

        cudaMemcpy(gpu_particles, particles, particle_count * sizeof(struct particle), cudaMemcpyHostToDevice);

        calcAcceleration<<<particle_count/128,128>>>(gpu_particles, center_of_mass);

        cudaMemcpy(particles, gpu_particles, particle_count * sizeof(struct particle), cudaMemcpyDeviceToHost);

        // Draw it

        glClear(GL_COLOR_BUFFER_BIT);
        //DrawCircle(-0.8, 0.1, 0.01);

        for(int i = 0; i < particle_count; i++) {
            scaled_x = (particles[i].x - WIDTH/2) / (WIDTH/2);
            scaled_y = (particles[i].y - HEIGHT/2) / (HEIGHT/2);
            DrawCircle(scaled_x, scaled_y, 0.0025);
        }

        glutSwapBuffers();

        usleep(1000);
    }
}