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

#include <time.h>

#include <locale.h>

#include "debug.h"

// Constants, Options

// Window size
#define WIDTH 1600
#define HEIGHT 900

// Tick Rate
#define TICKS_PER_SECOND (10000.0)
#define TIME_PER_TICK (1/TICKS_PER_SECOND)

/*
 * Multi-Tick
 * Each time the main() function ticks (calls calcAcceleration() on the GPU), 
 * the GPU will perform the operation MULTI_TICK_COUNT times if enabled.
 * This may reduce any bottlenecks, including GPU thread creation
 * and system calls
*/
#define MULTI_TICK_ENABLED 1
#define MULTI_TICK_COUNT 100

// Display Frame Rate
#define DRAW_FRAMES_PER_SECOND (40.0)
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

// Functions

float calcSpeed(struct particle*);



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



float calcSpeed(struct particle* p) {
    return sqrtf( powf(p->v_x,2.0f) + powf(p->v_y,2.0f) );
}



// Why is this function void?
// Because functions running on the graphics card cannot return a value!
// The value we want from this function must be passed as a parameter and set inside the function instead.
__device__ void calcDistance(float* distance, struct particle one, struct particle two) {
    (*distance) = sqrtf( powf(one.x - two.x, 2.0f) + powf(one.y - two.y, 2.0f) );
}



__global__ void calcAcceleration(struct particle* particles, struct particle center_of_mass) {
    // The number of the particle that this thread is working on
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate distance from this particle to the large object

    float distance;

    #if MULTI_TICK_ENABLED
    for(int i = 0; i < MULTI_TICK_COUNT; i++) {
    #endif

    calcDistance(&distance, particles[j], center_of_mass);

    distance *= DISTANCE_MULTIPLIER;
    if(distance < 1) distance = 1; // Stop particles from rocketting off the screen
    
    // Calculate gravitational force between this particle and the large object

    float force = 1 / powf(distance, 2);
    force *= FORCE_MULTIPLIER;

    // Calculate how the force should change the particle's x-axis velocity and y-axis velocity

    float x_distance = particles[j].x - center_of_mass.x;
    float y_distance = particles[j].y - center_of_mass.y;

    float angle = atanf(y_distance/x_distance);

    float v_x_add = cosf(angle) * force;
    float v_y_add = sinf(angle) * force;

    if(particles[j].x > center_of_mass.x) {
        v_x_add *= -1;
        v_y_add *= -1;
    }

    // Add the calculated velocity changes to the particle's current velocity

    particles[j].v_x += v_x_add;
    particles[j].v_y += v_y_add;

    // Move the particle according to it's current velocity

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

    #if MULTI_TICK_ENABLED
    }
    #endif
    
}



int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE);
    glutInitWindowSize(WIDTH,HEIGHT);
    glutInitWindowPosition(0,0);
    glutCreateWindow("Gravity Simulator");

    // set the locale so that printed numbers can be formatted
    // with commas separating the digits
    setlocale(LC_NUMERIC, "");


    // Time Stuff


    // Timing for displaying frames
    clock_t frames_start;
    frames_start = clock();

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


    struct particle* particles = (struct particle*) malloc(sizeof(particle) * PARTICLE_COUNT);
    if(particles == 0) {
        printf("Could not allocate memory for particles!\n");
        exit(1);
    }

    struct particle* gpu_particles;
    cudaMalloc(&gpu_particles, sizeof(struct particle) * PARTICLE_COUNT);
    if(gpu_particles == 0) {
        printf("Could not allocate GPU memory for particles!\n");
        exit(1);
    }

    struct particle center_of_mass;
    float scaled_x;
    float scaled_y;

    // Position Particles
    for(int i = 0; i < PARTICLE_COUNT; i++) {
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

    cudaMemcpy(gpu_particles, particles, PARTICLE_COUNT * sizeof(struct particle), cudaMemcpyHostToDevice);


    // End Allocating and positioning particles


    debug_printf("Simulating %'d particles\n", PARTICLE_COUNT);
    debug_printf("Time per tick: %f s\n", TIME_PER_TICK);
    debug_printf("Clocks per second: %'li\n", CLOCKS_PER_SEC);

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

            calcAcceleration<<<PARTICLE_COUNT/128,128>>>(gpu_particles, center_of_mass);

            //cudaMemcpy(particles, gpu_particles, particle_count * sizeof(struct particle), cudaMemcpyDeviceToHost);

            #if MULTI_TICK_ENABLED
            tick_count+=MULTI_TICK_COUNT;
            #else
            tick_count++;
            #endif
        }

        #if MULTI_TICK_ENABLED
        cudaDeviceSynchronize();
        #endif

        // Print physics frames each second

        time_now = clock();
        seconds = (float) (time_now - physics_time_start) / CLOCKS_PER_SEC;
        if(seconds >= 1.0) {
            physics_time_start = clock();
            debug_printf("%'d frames over %'f seconds\n", tick_count, seconds);
            tick_count = 0;
        }

        // Draw it

        time_now = clock();
        if((float) (time_now - frames_start)/CLOCKS_PER_SEC < TIME_PER_DRAW) continue;

        frames_start = clock();
        
        cudaMemcpy(particles, gpu_particles, PARTICLE_COUNT * sizeof(struct particle), cudaMemcpyDeviceToHost);

        glClear(GL_COLOR_BUFFER_BIT);

        scaled_x = (center_of_mass.x - WIDTH/2) / (WIDTH/2);
        scaled_y = (center_of_mass.y - HEIGHT/2) / (HEIGHT/2);
        DrawCircle(scaled_x, scaled_y, 0.01);

        for(int i = 0; i < PARTICLE_COUNT; i++) {
            DrawParticle(&particles[i]);
        }

        glutSwapBuffers();
    }
}
