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

#define PI 3.14159265359

void DrawCircle(float cx, float cy, float r) {
	glBegin(GL_LINE_LOOP);
	for(float i = 0; i < 2 * PI ; i += 0.1)
	{
		float x = r * cosf(i);
		float y = r * sinf(i);
		glVertex2f(x + cx, y + cy);
	}
	glEnd();
}

int main( int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE);    // Use single color buffer and no depth buffer.
    glutInitWindowSize(1600,800);         // Size of display area, in pixels.
    glutInitWindowPosition(0,0);     // Location of window in screen coordinates.
    glutCreateWindow("N-Particle Simulator"); // Parameter is window title.

    int particle_count = 1000;
    struct particle* particles = (struct particle*) malloc(sizeof(particle) * particle_count);
    if(particles == 0) {
        printf("Could not allocate memory for particles!\n");
        exit(1);
    }

    struct particle center_of_mass;

    while(1) {

        // Do the physics

        calculate_center_of_mass(&center_of_mass, particles, particle_count);


        // Draw it
        glClear(GL_COLOR_BUFFER_BIT);
        DrawCircle(-0.8, 0.1, 0.01);
        glutSwapBuffers();

        sleep(1000);
    }
}