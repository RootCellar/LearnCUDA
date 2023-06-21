/*
 *
 * Darian Marvel
 * 2/24/2023
 *
 * Personal Project, and submitted for CS241 Computer Hardware Concepts
 * 
 * Small gravity simulator that runs on the graphics card
 *
 * A bunch of particles are placed with an initial velocity,
 * and they are pulled by some object at the center of the screen
 *
 * Uses some OpenGL code from Ryan Brune's Orbital-Simulator
 *
*/



// Includes

#include <GL/glut.h>

#include <stdio.h>
#include <time.h>

#include "debug.h"


// Window size
#define WIDTH 1600
#define HEIGHT 900

// Tick Rate
#define TICKS_PER_SECOND (10000.0)
#define TIME_PER_TICK (1/TICKS_PER_SECOND)

// Display Frame Rate
#define DRAW_FRAMES_PER_SECOND (50.0)
#define TIME_PER_DRAW (1.0/DRAW_FRAMES_PER_SECOND) // Seconds

/* 
 *
 * Particle Count
 * Note - turning this up too high causes display drawing times to become really long!
 * This may cause the GPU to run the simulation much slower than it actually can.
 * In cases where you want lots of particles, I recommend turning down the frame rate
 * by turning down DRAW_FRAMES_PER_SECOND above.
 *
 * Also, make sure this is divisible by 128, the GPU thread block size.
 *
*/ 
#define PARTICLE_COUNT (128*500)

// Whether or not the particles are forced to slow down over time
#define SLOW_DOWN 0

// Whether or not particles will bounce off the edges of the screen 
// and stay in bounds.
#define KEEP_ON_SCREEN 0

// Used to set the scale
// Make sure this evaluates as a float!
#define DISTANCE_MULTIPLIER (1.0/1.5)

// Set how strong gravity is
#define FORCE_MULTIPLIER (0.02)

#define PI 3.14159265359



struct particle {

    // Position
    float x;
    float y;

    // Velocity
    float v_x;
    float v_y;

};

/*
 * Calculates the center of mass of all of the given particles, and sets the given particle
 * "center_of_mass" as the answer
 *
 * Not currently used
*/
void calculate_center_of_mass(struct particle* center_of_mass, struct particle* particles, int count) {
    float x = 0;
    float y = 0;

    for(int i = 0; i < count; i++) {
        x += particles[i].x;
        y += particles[i].y;
    }

    x /= count;
    y /= count;

    center_of_mass->x = x;
    center_of_mass->y = y;


}


float calcSpeed(struct particle* p) {
    return sqrtf( powf(p->v_x,2.0f) + powf(p->v_y,2.0f) );
}

void DrawParticle(struct particle* p) {
    glBegin(GL_POINTS);

    float speed = calcSpeed(p);

    // "Redness" is used to color particles based on speed
    float redness = speed*40; // TODO: find a better way to calculate this

    glColor3f(redness, 1-redness, 1-redness);

    float scaled_x = (p->x - WIDTH/2) / (WIDTH/2);
    float scaled_y = (p->y - HEIGHT/2) / (HEIGHT/2);

    glVertex2f(scaled_x, scaled_y);

    glEnd();
}

void DrawCircle(float cx, float cy, float r) {
	glBegin(GL_LINE_LOOP);
    glColor3f(1.0, 1.0, 1.0);
	for(float i = 0; i < 2 * PI ; i += 0.1) {
		float x = r * cosf(i);
        x /= WIDTH/HEIGHT;

		float y = r * sinf(i);
        
		glVertex2f(x + cx, y + cy);
	}
	glEnd();
}

// Why is this function void?
// Because functions running on the graphics card cannot return a value!
// The value we want from this function must be passed as a parameter and set inside the function instead.
__device__ void calcDistance(float* distance, struct particle one, struct particle two) {
    (*distance) = sqrtf( powf(one.x - two.x, 2.0f) + powf(one.y - two.y, 2.0f) );
}

__global__ void calcAcceleration(struct particle* particles, struct particle center_of_mass) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float distance;
    calcDistance(&distance, particles[j], center_of_mass);

    distance *= DISTANCE_MULTIPLIER;
    if(distance < 1) distance = 1;
    
    float force = 1 / powf(distance, 2);
    force *= FORCE_MULTIPLIER;

    float x_distance = particles[j].x - center_of_mass.x;
    float y_distance = particles[j].y - center_of_mass.y;

    float angle = atanf(y_distance/x_distance);

    float v_x_add = cosf(angle) * force;
    float v_y_add = sinf(angle) * force;

    if(particles[j].x > center_of_mass.x) {
        v_x_add *= -1;
        v_y_add *= -1;
    }

    particles[j].v_x += v_x_add;
    particles[j].v_y += v_y_add;


    particles[j].x += particles[j].v_x;
    particles[j].y += particles[j].v_y;

    if( SLOW_DOWN ) {
      particles[j].v_x *= 0.9999;
      particles[j].v_y *= 0.9999;
    }

    if(KEEP_ON_SCREEN) {

        // Bring back inside screen bounds if the particle is somehow
        // off screen
        if(particles[j].x > WIDTH) particles[j].x = WIDTH;
        if(particles[j].x < 0) particles[j].x = 0;

        if(particles[j].y > HEIGHT) particles[j].y = HEIGHT;
        if(particles[j].y < 0) particles[j].y = 0;

        // If the particle would go off screen,
        // bounce off the edge
        if(particles[j].x + particles[j].v_x > WIDTH) particles[j].v_x *= -1;
        if(particles[j].x + particles[j].v_x < 0) particles[j].v_x *= -1;

        if(particles[j].y + particles[j].v_y > HEIGHT) particles[j].v_y *= -1;
        if(particles[j].y + particles[j].v_y < 0) particles[j].v_y *= -1;
    }
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE);    // Use single color buffer and no depth buffer.
    glutInitWindowSize(WIDTH,HEIGHT);         // Size of display area, in pixels.
    glutInitWindowPosition(0,0);     // Location of window in screen coordinates.
    glutCreateWindow("Gravity Simulator"); // Parameter is window title.


    // Time Stuff


    // Timing for displaying frames
    clock_t start_time;
    start_time = clock();

    // Timing for physics frames
    clock_t physics_start;
    physics_start = clock();

    // Timing for printing physics simulation rate
    int tick_count = 0;
    clock_t physics_time_start;
    physics_time_start = clock();

    float seconds;
    clock_t time_now;


    // End Time Stuff


    // Allocating and positioning particles


    int particle_count = PARTICLE_COUNT;
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
        particles[i].x = rand() % (WIDTH/6) + (WIDTH/4);
        particles[i].y = rand() % (HEIGHT/6) + (HEIGHT/4);

        particles[i].x += 5.0 * ( (float) rand() / RAND_MAX );
        particles[i].y += 5.0 * ( (float) rand() / RAND_MAX );

        particles[i].v_x = 0.01;
        if(particles[i].x > WIDTH/2) particles[i].v_x *= -1;
        particles[i].v_y = 0;
    }

    //calculate_center_of_mass(&center_of_mass, particles, particle_count);
    center_of_mass.x = WIDTH/2;
    center_of_mass.y = HEIGHT/2;

    cudaMemcpy(gpu_particles, particles, particle_count * sizeof(struct particle), cudaMemcpyHostToDevice);


    // End Allocating and positioning particles


    debug_printf("Simulating %d particles\n", particle_count);
    debug_printf("Time per tick: %f s\n", TIME_PER_TICK);
    debug_printf("Clocks per second: %li\n", CLOCKS_PER_SEC);


    while(1) {


        // Do the physics

        time_now = clock();
        seconds = (float) (time_now - physics_start) / CLOCKS_PER_SEC;
        if(seconds >= TIME_PER_TICK) {
            physics_start = clock();

            // The three commented-out lines here were used when the center of mass' position was calculated,
            // but they are now unused because copying the particle data back and forth makes the simulation *really* slow
            // and the "center of mass" / object with gravity has been put in a static position.

            //calculate_center_of_mass(&center_of_mass, particles, particle_count);

            //cudaMemcpy(gpu_particles, particles, particle_count * sizeof(struct particle), cudaMemcpyHostToDevice);

            calcAcceleration<<<particle_count/128,128>>>(gpu_particles, center_of_mass);

            //cudaMemcpy(particles, gpu_particles, particle_count * sizeof(struct particle), cudaMemcpyDeviceToHost);

            tick_count++;
        }


        // Print physics frames each second

        time_now = clock();
        seconds = (float) (time_now - physics_time_start) / CLOCKS_PER_SEC;
        if(seconds >= 1.0) {
            physics_time_start = clock();
            debug_printf("%d frames over %f seconds\n", tick_count, seconds);
            tick_count = 0;
        }

        // Draw it

        time_now = clock();
        if((float) (time_now - start_time)/CLOCKS_PER_SEC < TIME_PER_DRAW) continue;

        cudaMemcpy(particles, gpu_particles, particle_count * sizeof(struct particle), cudaMemcpyDeviceToHost);

        start_time = clock();

        glClear(GL_COLOR_BUFFER_BIT);

        scaled_x = (center_of_mass.x - WIDTH/2) / (WIDTH/2);
        scaled_y = (center_of_mass.y - HEIGHT/2) / (HEIGHT/2);
        DrawCircle(scaled_x, scaled_y, 0.01);

        for(int i = 0; i < particle_count; i++) {
            DrawParticle(&particles[i]);
        }

        glutSwapBuffers();
    }
}