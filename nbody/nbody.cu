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
	for(float i = 0; i < 2 * PI ; i += 1.0)
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

    float distance;
    calcDistance(&distance, particles[j], center_of_mass);

    distance /= 5;
    if(distance < 2) distance = 2;
    
    float force = 1 / powf(distance, 2);
    force /= 1;

    float x_distance = particles[j].x - center_of_mass.x;
    float y_distance = particles[j].y - center_of_mass.y;

    float angle = atanf(y_distance/x_distance);

    float v_x_add = cosf(angle) * force;
    float v_y_add = sinf(angle) * force;

    if(particles[j].x > center_of_mass.x) {
        v_x_add *= -1;
        v_y_add *= -1;
    }

    /*
    if(particles[j].y > center_of_mass.y) {
        v_y_add *= -1;
    }
    */

    particles[j].v_x += v_x_add;
    particles[j].v_y += v_y_add;


    particles[j].x += particles[j].v_x;
    particles[j].y += particles[j].v_y;

    particles[j].v_x *= 0.99999;
    particles[j].v_y *= 0.99999;

    if(particles[j].x > WIDTH) particles[j].x = WIDTH;
    if(particles[j].x < 0) particles[j].x = 0;

    if(particles[j].y > HEIGHT) particles[j].y = HEIGHT;
    if(particles[j].y < 0) particles[j].y = 0;


    if(particles[j].x + particles[j].v_x > WIDTH) particles[j].v_x *= -1;
    if(particles[j].x + particles[j].v_x < 0) particles[j].v_x *= -1;

    if(particles[j].y + particles[j].v_y > HEIGHT) particles[j].v_y *= -1;
    if(particles[j].y + particles[j].v_y < 0) particles[j].v_y *= -1;
}

int main( int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE);    // Use single color buffer and no depth buffer.
    glutInitWindowSize(WIDTH,HEIGHT);         // Size of display area, in pixels.
    glutInitWindowPosition(0,0);     // Location of window in screen coordinates.
    glutCreateWindow("N-Particle Simulator"); // Parameter is window title.

    int particle_count = 128*100;
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
        particles[i].x = i % 128;
        particles[i].x *= 4;
        particles[i].y = i / 128;
        particles[i].y *= 4;

        particles[i].v_x = 0;
        particles[i].v_y = 0;
    }

    //calculate_center_of_mass(&center_of_mass, particles, particle_count);
    center_of_mass.x = WIDTH/2;
    center_of_mass.y = HEIGHT/2;

    while(1) {

        // Do the physics

        //calculate_center_of_mass(&center_of_mass, particles, particle_count);

        cudaMemcpy(gpu_particles, particles, particle_count * sizeof(struct particle), cudaMemcpyHostToDevice);

        calcAcceleration<<<particle_count/128,128>>>(gpu_particles, center_of_mass);

        cudaMemcpy(particles, gpu_particles, particle_count * sizeof(struct particle), cudaMemcpyDeviceToHost);

        // Draw it

        glClear(GL_COLOR_BUFFER_BIT);

        scaled_x = (center_of_mass.x - WIDTH/2) / (WIDTH/2);
        scaled_y = (center_of_mass.y - HEIGHT/2) / (HEIGHT/2);
        DrawCircle(scaled_x, scaled_y, 0.01);

        for(int i = 0; i < particle_count; i++) {
            scaled_x = (particles[i].x - WIDTH/2) / (WIDTH/2);
            scaled_y = (particles[i].y - HEIGHT/2) / (HEIGHT/2);
            DrawCircle(scaled_x, scaled_y, 0.0025);
        }

        glutSwapBuffers();

        usleep(1000);
    }
}